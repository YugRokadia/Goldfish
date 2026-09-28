from __future__ import annotations

import json
import re
import threading
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from typing import Any

import keyring
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVICE_NAME = "RecallX"

ACCOUNT_NAME = "gmail"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
]

ENGINE_DIR = Path(__file__).resolve().parents[2]

SECRETS_DIR = ENGINE_DIR / "secrets"

CREDENTIALS_PATH = SECRETS_DIR / "credentials.json"

KEYRING_USERNAME = "gmail"

MAX_RESULTS = 20


# ---------------------------------------------------------------------------
# OAuth state
# ---------------------------------------------------------------------------

_auth_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _credentials_exist() -> bool:
    return CREDENTIALS_PATH.exists()


def _load_credentials_from_keyring() -> Credentials | None:
    token_json = keyring.get_password(
        SERVICE_NAME,
        KEYRING_USERNAME,
    )

    if not token_json:
        return None

    try:
        return Credentials.from_authorized_user_info(
            json.loads(token_json),
            SCOPES,
        )
    except Exception:
        return None


def _save_credentials(credentials: Credentials) -> None:
    keyring.set_password(
        SERVICE_NAME,
        KEYRING_USERNAME,
        credentials.to_json(),
    )


def _delete_credentials() -> None:
    try:
        keyring.delete_password(
            SERVICE_NAME,
            KEYRING_USERNAME,
        )
    except keyring.errors.PasswordDeleteError:
        pass


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------


def connect_gmail() -> dict[str, Any]:
    """
    Start the Google OAuth flow.

    Google opens the user's browser. After authorization, the OAuth
    credentials are stored in Windows Credential Manager through keyring.
    """

    if not _credentials_exist():
        raise FileNotFoundError(
            f"Gmail OAuth credentials not found at: {CREDENTIALS_PATH}"
        )

    with _auth_lock:
        credentials = _load_credentials_from_keyring()

        # Existing valid credentials.
        if credentials and credentials.valid:
            return {
                "connected": True,
                "email": get_gmail_profile()["email"],
            }

        # Existing credentials with a refresh token.
        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
                _save_credentials(credentials)

                return {
                    "connected": True,
                    "email": get_gmail_profile()["email"],
                }

            except Exception:
                _delete_credentials()

        # First-time OAuth.
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_PATH),
            SCOPES,
        )

        credentials = flow.run_local_server(
            host="127.0.0.1",
            port=0,
            open_browser=True,
        )

        _save_credentials(credentials)

        profile = get_gmail_profile()

        return {
            "connected": True,
            "email": profile["email"],
        }


def disconnect_gmail() -> dict[str, Any]:
    """Remove locally stored Gmail OAuth credentials."""

    _delete_credentials()

    return {
        "connected": False,
    }


# ---------------------------------------------------------------------------
# Gmail service
# ---------------------------------------------------------------------------


def get_credentials() -> Credentials | None:
    """
    Return valid Gmail credentials.

    Refreshes expired credentials when a refresh token exists.
    """

    credentials = _load_credentials_from_keyring()

    if not credentials:
        return None

    if credentials.valid:
        return credentials

    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
            _save_credentials(credentials)
            return credentials
        except Exception:
            _delete_credentials()
            return None

    return None


def get_gmail_service() -> Resource | None:
    """Create an authenticated Gmail API service."""

    credentials = get_credentials()

    if not credentials:
        return None

    return build(
        "gmail",
        "v1",
        credentials=credentials,
        cache_discovery=False,
    )


# ---------------------------------------------------------------------------
# Account status
# ---------------------------------------------------------------------------


def is_gmail_connected() -> bool:
    """Return whether Gmail is currently connected."""

    return get_credentials() is not None


def get_gmail_profile() -> dict[str, Any]:
    """Return the authenticated Gmail profile."""

    service = get_gmail_service()

    if service is None:
        raise RuntimeError("Gmail is not connected.")

    profile = (
        service.users()
        .getProfile(userId="me")
        .execute()
    )

    return {
        "email": profile.get("emailAddress", ""),
        "messages_total": profile.get("messagesTotal", 0),
        "threads_total": profile.get("threadsTotal", 0),
    }


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------


def _header_value(
    headers: list[dict[str, str]],
    name: str,
) -> str:
    target = name.lower()

    for header in headers:
        if header.get("name", "").lower() == target:
            return header.get("value", "")

    return ""


def _build_message_url(
    message_id: str,
    thread_id: str,
    email: str = "",
) -> str:
    """
    Build a Gmail URL that targets the authenticated account.

    Gmail's multi-account email-based /u/<email> and authuser
    URL behavior is not reliable in current Gmail. Use Google's
    AccountChooser with the authenticated email, then redirect
    to the requested message/thread.
    """

    target_id = thread_id or message_id

    gmail_url = (
        "https://mail.google.com/mail/u/0/"
        f"#all/{target_id}"
    )

    if not email:
        return gmail_url

    return (
        "https://accounts.google.com/AccountChooser"
        f"?Email={quote(email.strip(), safe='')}"
        f"&continue={quote(gmail_url, safe='')}"
    )


# ---------------------------------------------------------------------------
# Search query helpers
# ---------------------------------------------------------------------------


_EMAIL_WORD_RE = re.compile(
    r"\b(?:email|emails|e-mail|e-mails|mail|mails|message|messages|gmail)\b",
    flags=re.IGNORECASE,
)


def _format_gmail_date(value: datetime) -> str:
    """Format a local date for Gmail's after:/before: operators."""
    return value.strftime("%Y/%m/%d")


def build_gmail_query(query: str) -> str:
    """
    Convert common natural-language email searches into Gmail search syntax.

    This is intentionally lightweight. Gmail remains responsible for the
    actual search, ranking, and filtering.

    Examples:
        "mail from yug" -> "from:yug"
        "emails from kpmg about internship" -> "from:kpmg internship"
        "unread mail from yug" -> "is:unread from:yug"
        "emails from kpmg yesterday" -> "from:kpmg after:... before:..."
        "emails with attachments" -> "has:attachment"

    Existing Gmail operators such as from:, subject:, after:, before:, etc.
    are preserved as-is.
    """
    cleaned = query.strip()

    if not cleaned:
        return ""

    # Preserve native Gmail syntax. We only add natural-language filters
    # around it rather than trying to rewrite an existing Gmail query.
    # Gmail operators are case-insensitive, so this check is intentionally
    # simple.
    # ---------------------------------------------------------------
    # Natural-language sender / recipient / subject
    # ---------------------------------------------------------------

    match = re.search(
        r"\b(?:email|emails|e-mail|e-mails|mail|mails|"
        r"message|messages)\s+from\s+([^\s]+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match and not re.search(r"(^|\s)from:", cleaned, flags=re.IGNORECASE):
        sender = match.group(1).strip(".,;")
        cleaned = (
            cleaned[:match.start()]
            + f"from:{sender}"
            + cleaned[match.end():]
        )

    match = re.search(
        r"\b(?:email|emails|e-mail|e-mails|mail|mails|"
        r"message|messages)\s+to\s+([^\s]+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match and not re.search(r"(^|\s)to:", cleaned, flags=re.IGNORECASE):
        recipient = match.group(1).strip(".,;")
        cleaned = (
            cleaned[:match.start()]
            + f"to:{recipient}"
            + cleaned[match.end():]
        )

    match = re.search(
        r"\b(?:email|emails|e-mail|e-mails|mail|mails|"
        r"message|messages)\s+subject\s+([^\s]+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match and not re.search(r"(^|\s)subject:", cleaned, flags=re.IGNORECASE):
        subject = match.group(1).strip(".,;")
        cleaned = (
            cleaned[:match.start()]
            + f"subject:{subject}"
            + cleaned[match.end():]
        )

    # ---------------------------------------------------------------
    # Natural-language state / attachment filters
    # ---------------------------------------------------------------

    extra_terms: list[str] = []

    if re.search(r"\bunread\b", cleaned, flags=re.IGNORECASE):
        extra_terms.append("is:unread")

    if re.search(r"\bread\b", cleaned, flags=re.IGNORECASE):
        extra_terms.append("is:read")

    if re.search(
        r"\b(?:with|has|containing)\s+(?:an?\s+)?attachments?\b"
        r"|\battachments?\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        extra_terms.append("has:attachment")

    # Remove only the natural-language forms we translated. Native Gmail
    # operators remain untouched.
    cleaned = re.sub(
        r"\b(?:unread|read)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:with|has|containing)\s+(?:an?\s+)?attachments?\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # ---------------------------------------------------------------
    # Relative date filters
    # ---------------------------------------------------------------

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    date_filter: list[str] = []

    if re.search(r"\byesterday\b", cleaned, flags=re.IGNORECASE):
        start = today - timedelta(days=1)
        end = today
        date_filter.extend(
            [f"after:{_format_gmail_date(start)}",
             f"before:{_format_gmail_date(end)}"]
        )
        cleaned = re.sub(r"\byesterday\b", " ", cleaned, flags=re.IGNORECASE)

    elif re.search(r"\btoday\b", cleaned, flags=re.IGNORECASE):
        tomorrow = today + timedelta(days=1)
        date_filter.extend(
            [f"after:{_format_gmail_date(today)}",
             f"before:{_format_gmail_date(tomorrow)}"]
        )
        cleaned = re.sub(r"\btoday\b", " ", cleaned, flags=re.IGNORECASE)

    elif re.search(r"\blast\s+week\b", cleaned, flags=re.IGNORECASE):
        # Monday-to-Monday week range.
        this_monday = today - timedelta(days=today.weekday())
        last_monday = this_monday - timedelta(days=7)
        date_filter.extend(
            [f"after:{_format_gmail_date(last_monday)}",
             f"before:{_format_gmail_date(this_monday)}"]
        )
        cleaned = re.sub(
            r"\blast\s+week\b", " ", cleaned, flags=re.IGNORECASE
        )

    elif re.search(r"\bthis\s+week\b", cleaned, flags=re.IGNORECASE):
        this_monday = today - timedelta(days=today.weekday())
        next_monday = this_monday + timedelta(days=7)
        date_filter.extend(
            [f"after:{_format_gmail_date(this_monday)}",
             f"before:{_format_gmail_date(next_monday)}"]
        )
        cleaned = re.sub(
            r"\bthis\s+week\b", " ", cleaned, flags=re.IGNORECASE
        )

    # ---------------------------------------------------------------
    # Remove routing words.
    # ---------------------------------------------------------------

    cleaned = _EMAIL_WORD_RE.sub(" ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    parts = []
    if extra_terms:
        parts.extend(extra_terms)
    if date_filter:
        parts.extend(date_filter)
    if cleaned:
        parts.append(cleaned)

    result = " ".join(parts).strip()

    # If the input was only routing words and no translation was possible,
    # preserve the original query instead of accidentally searching for
    # nothing.
    if not result:
        return query.strip()

    return result


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_gmail(
    query: str,
    limit: int = MAX_RESULTS,
) -> list[dict[str, Any]]:
    """
    Search Gmail live using Gmail's native search syntax.

    Examples:

        KPMG internship
        from:example@gmail.com
        subject:internship
        after:2026/09/01 internship
        "research paper"
    """

    query = build_gmail_query(query)

    if not query:
        return []

    limit = max(1, min(limit, MAX_RESULTS))

    service = get_gmail_service()

    if service is None:
        return []

    # Get the account actually authenticated with RecallX.
    profile = get_gmail_profile()
    account_email = profile["email"]

    try:
        response = (
            service.users()
            .messages()
            .list(
                userId="me",
                q=query,
                maxResults=limit,
            )
            .execute()
        )

        messages = response.get("messages", [])

        results: list[dict[str, Any]] = []

        for message in messages:
            message_id = message.get("id", "")
            thread_id = message.get("threadId", "")

            if not message_id:
                continue

            detail = (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=message_id,
                    format="metadata",
                    metadataHeaders=[
                        "Subject",
                        "From",
                        "To",
                        "Date",
                    ],
                )
                .execute()
            )

            payload = detail.get("payload", {})
            headers = payload.get("headers", [])

            subject = _header_value(
                headers,
                "Subject",
            )

            sender = _header_value(
                headers,
                "From",
            )

            recipient = _header_value(
                headers,
                "To",
            )

            date = _header_value(
                headers,
                "Date",
            )

            snippet = detail.get(
                "snippet",
                "",
            )

            results.append(
                {
                    "id": message_id,
                    "thread_id": thread_id,
                    "source_type": "gmail",
                    "content_type": "email",
                    "title": subject or "(No subject)",
                    "sender": sender,
                    "recipient": recipient,
                    "date": date,
                    "content": snippet,
                    "snippet": snippet,
                    "path": (
                        f"gmail:{message_id}"
                    ),
                    "url": _build_message_url(
                        message_id,
                        thread_id,
                        account_email,
                    ),
                }
            )

        return results

    except HttpError as error:
        status = getattr(
            error.resp,
            "status",
            None,
        )

        if status == 401:
            _delete_credentials()

            raise RuntimeError(
                "Gmail authorization has expired or "
                "been revoked."
            ) from error

        if status == 403:
            raise RuntimeError(
                "Gmail API access was denied."
            ) from error

        raise RuntimeError(
            f"Gmail API error: {error}"
        ) from error


# ---------------------------------------------------------------------------
# Open Gmail message
# ---------------------------------------------------------------------------


def open_gmail_message(
    message_id: str,
    thread_id: str = "",
) -> dict[str, Any]:
    """Open the original Gmail message in the user's browser."""

    profile = get_gmail_profile()
    account_email = profile["email"]

    url = _build_message_url(
        message_id,
        thread_id,
        account_email,
    )

    webbrowser.open(url)

    return {
        "opened": True,
        "url": url,
    }