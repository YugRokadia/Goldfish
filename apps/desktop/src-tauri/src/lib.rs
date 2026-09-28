#[cfg(target_os = "windows")]
use std::ffi::c_void;

#[cfg(target_os = "windows")]
use windows::Win32::Foundation::HWND;

#[cfg(target_os = "windows")]
use windows::Win32::Graphics::Gdi::{
    CreateRoundRectRgn,
    SetWindowRgn,
};

use std::process::{Child, Command, Stdio};

use std::sync::Mutex;

struct EngineProcess(Mutex<Option<Child>>);

#[cfg(not(debug_assertions))]
fn start_engine(app: &tauri::AppHandle) -> Result<(), String> {
    use tauri::Manager;

    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|error| error.to_string())?;

    let executable_name = if cfg!(target_os = "windows") {
        "recallx-engine.exe"
    } else {
        "recallx-engine"
    };

    let executable = resource_dir
        .join("recallx-engine")
        .join(executable_name);

    let child = Command::new(&executable)
        .current_dir(
            executable
                .parent()
                .ok_or_else(|| "Engine directory not found".to_string())?,
        )
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| {
            format!(
                "Failed to start RecallX engine at {}: {}",
                executable.display(),
                error
            )
        })?;

    if let Some(state) = app.try_state::<EngineProcess>() {
        *state
            .0
            .lock()
            .map_err(|_| "Engine process lock is poisoned".to_string())? =
            Some(child);
    }

    Ok(())
}

#[cfg(target_os = "windows")]
fn apply_rounded_region(window: &tauri::WebviewWindow) {
    let Ok(raw_hwnd) = window.hwnd() else {
        eprintln!("Failed to get RecallX HWND");
        return;
    };

    let hwnd = HWND(raw_hwnd.0 as *mut c_void);

    // Get the actual physical/native window size.
    let Ok(size) = window.outer_size() else {
        eprintln!("Failed to get RecallX window size");
        return;
    };

    let width = size.width as i32;
    let height = size.height as i32;

    // Fixed radius so the expanded results window
    // does not become a giant pill/circle.
    let radius = 36;

    let region = unsafe {
        CreateRoundRectRgn(
            0,
            0,
            width + 1,
            height + 1,
            radius,
            radius,
        )
    };

    if region.is_invalid() {
        eprintln!("Failed to create rounded window region");
        return;
    }

    unsafe {
        SetWindowRgn(hwnd, Some(region), true);
    }
}

#[tauri::command]
fn set_window_height(
    app: tauri::AppHandle,
    height: f64,
) -> Result<(), String> {
    use tauri::{LogicalSize, Manager};

    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "RecallX window not found".to_string())?;

    // Keep the current position so the window
    // expands downward rather than jumping around.
    let position = window
        .outer_position()
        .map_err(|error| error.to_string())?;

    // Change only the height.
    // Width remains 680 logical pixels.
    window
        .set_size(LogicalSize::new(680.0, height))
        .map_err(|error| error.to_string())?;

    // Restore the same position after resizing.
    window
        .set_position(position)
        .map_err(|error| error.to_string())?;

    // Reapply the native rounded region because
    // the window height has changed.
    #[cfg(target_os = "windows")]
    apply_rounded_region(&window);

    Ok(())
}

// ---------------------------------------------------------------------------
// Open a file and hide RecallX
// ---------------------------------------------------------------------------

#[tauri::command]
fn open_file(
    app: tauri::AppHandle,
    path: String,
) -> Result<(), String> {
    use std::process::Command;
    use tauri::Manager;

    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "RecallX window not found".to_string())?;

    // Open the file using the Windows default application.
    //
    // `start "" "path"` lets Windows choose the application
    // associated with the file extension.
    Command::new("cmd")
        .args(["/C", "start", "", &path])
        .spawn()
        .map_err(|error| format!("Failed to open file: {}", error))?;

    // Hide RecallX immediately after launching the file.
    window
        .hide()
        .map_err(|error| format!("Failed to hide RecallX: {}", error))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Hide RecallX natively
// ---------------------------------------------------------------------------

#[tauri::command]
fn hide_window(
    app: tauri::AppHandle,
) -> Result<(), String> {
    use tauri::Manager;

    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "RecallX window not found".to_string())?;

    window
        .hide()
        .map_err(|error| format!("Failed to hide RecallX: {}", error))?;

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()

    .manage(EngineProcess(Mutex::new(None)))

        // ------------------------------------------------------------
        // Existing opener plugin
        // ------------------------------------------------------------
        .plugin(tauri_plugin_opener::init())

        // ------------------------------------------------------------
        // Global Ctrl + Space
        // ------------------------------------------------------------
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    use tauri::Manager;
                    use tauri_plugin_global_shortcut::{
                        Code,
                        Modifiers,
                        ShortcutState,
                    };

                    // Make sure this handler is only responding
                    // to Ctrl + Space.
                    if !shortcut.matches(Modifiers::CONTROL, Code::Space) {
                        return;
                    }

                    // Only react to the key press.
                    // Ignore key release.
                    if event.state() != ShortcutState::Pressed {
                        return;
                    }

                    let Some(window) =
                        app.get_webview_window("main")
                    else {
                        eprintln!(
                            "RecallX: main window not found"
                        );
                        return;
                    };

                    match window.is_visible() {
                        // ------------------------------------------------
                        // Currently visible -> hide
                        // ------------------------------------------------
                        Ok(true) => {
                            if let Err(error) = window.hide() {
                                eprintln!(
                                    "RecallX: failed to hide window: {}",
                                    error
                                );
                            } else {
                                println!(
                                    "RecallX: window hidden"
                                );
                            }
                        }

                        // ------------------------------------------------
                        // Currently hidden -> show + focus
                        // ------------------------------------------------
                        Ok(false) => {
                            if let Err(error) = window.show() {
                                eprintln!(
                                    "RecallX: failed to show window: {}",
                                    error
                                );
                                return;
                            }

                            if let Err(error) = window.set_focus() {
                                eprintln!(
                                    "RecallX: failed to focus window: {}",
                                    error
                                );
                            } else {
                                println!(
                                    "RecallX: window shown"
                                );
                            }
                        }

                        Err(error) => {
                            eprintln!(
                                "RecallX: failed to determine window visibility: {}",
                                error
                            );
                        }
                    }
                })
                .build(),
        )

        // ------------------------------------------------------------
        // React -> Rust commands
        // ------------------------------------------------------------
        .invoke_handler(tauri::generate_handler![
            set_window_height,
            open_file,
            hide_window
        ])

        // ------------------------------------------------------------
        // Window setup
        // ------------------------------------------------------------
        .setup(|app| {
            #[cfg(not(debug_assertions))]
            start_engine(app.handle())?;

            #[cfg(desktop)]
            {
                use tauri::webview::WebviewWindowBuilder;
                use tauri::WebviewUrl;

                // --------------------------------------------------------
                // Register Ctrl + Space
                // --------------------------------------------------------

                use tauri_plugin_global_shortcut::{
                    Code,
                    GlobalShortcutExt,
                    Modifiers,
                    Shortcut,
                };

                let ctrl_space = Shortcut::new(
                    Some(Modifiers::CONTROL),
                    Code::Space,
                );

                app.global_shortcut()
                    .register(ctrl_space)?;

                // --------------------------------------------------------
                // Create RecallX window
                // --------------------------------------------------------

                let window = WebviewWindowBuilder::new(
                    app,
                    "main",
                    WebviewUrl::App("index.html".into()),
                )
                .title("RecallX")

                // --------------------------------------------------------
                // Base / idle size
                // --------------------------------------------------------

                .inner_size(680.0, 70.0)

                .min_inner_size(500.0, 70.0)

                .resizable(false)

                // --------------------------------------------------------
                // Frameless native window
                // --------------------------------------------------------

                .decorations(false)

                // --------------------------------------------------------
                // Transparent native background
                // --------------------------------------------------------

                .transparent(true)

                .shadow(false)

                // --------------------------------------------------------
                // Keep RecallX above other windows
                // --------------------------------------------------------

                .always_on_top(true)

                // Keep your current behavior for now.
                .visible(true)

                // --------------------------------------------------------
                // Windows transparency workaround
                // --------------------------------------------------------

                .no_redirection_bitmap(true)

                // --------------------------------------------------------
                // Center on active monitor
                // --------------------------------------------------------

                .center()

                .build()?;

                // --------------------------------------------------------
                // Move the centered window 150 physical pixels upward.
                // --------------------------------------------------------

                let position = window.outer_position()?;

                window.set_position(
                    tauri::PhysicalPosition::new(
                        position.x,
                        position.y - 150,
                    ),
                )?;

                // --------------------------------------------------------
                // Apply native Windows rounded corners
                // --------------------------------------------------------

                #[cfg(target_os = "windows")]
                apply_rounded_region(&window);
            }

            Ok(())
        })

        .run(tauri::generate_context!())
        .expect("error while running RecallX");
}