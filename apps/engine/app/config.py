import os
from pathlib import Path


# apps/engine/
ENGINE_DIR = Path(__file__).resolve().parent.parent

def _default_data_dir() -> Path:
	"""Return a writable per-user data directory for the current platform."""
	if os.name == "nt":
		base_dir = os.getenv("LOCALAPPDATA")
	else:
		base_dir = os.getenv("XDG_DATA_HOME")

	if base_dir:
		return Path(base_dir) / "RecallX"

	return Path.home() / ".local" / "share" / "RecallX"


# Packaged builds must not write inside the installed application directory.
DATA_DIR = Path(
	os.getenv("RECALLX_DATA_DIR", str(_default_data_dir()))
).expanduser()

# Local RecallX database.
DATABASE_PATH = DATA_DIR / "recallx.db"

# FAISS index will eventually live here.
INDEX_DIR = DATA_DIR / "index"


# Make sure our local data directories exist.
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)