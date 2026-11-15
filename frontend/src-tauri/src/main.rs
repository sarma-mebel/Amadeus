use std::{
    io::{Read, Write},
    net::{TcpStream, ToSocketAddrs},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use base64::Engine as _;
use keyring::Entry;
use tauri::{
    async_runtime, AppHandle, Emitter, Manager, Runtime, Url, WebviewUrl, WebviewWindow,
    WebviewWindowBuilder,
};
use tauri_plugin_shell::{process::CommandChild, process::CommandEvent, ShellExt};

const KEYCHAIN_SERVICE: &str = "amadeus-kurisu";
const GEMINI_KEY_USER: &str = "gemini-api-key";
const PET_WINDOW_LABEL: &str = "pet";

/// Foreground-window fullscreen detection, polled ~every 5s by the pet
/// frontend so it can quiet/hide itself during games, movies, or screen
/// shares. Windows-only (the only platform the pet overlay targets in
/// practice); every other target always reports "not fullscreen" rather than
/// erroring, since the pet's auto-hide is a nicety, not a feature the rest of
/// the app depends on.
#[cfg(target_os = "windows")]
mod fullscreen_detect {
    use windows_sys::Win32::Foundation::{HWND, RECT};
    use windows_sys::Win32::Graphics::Gdi::{
        GetMonitorInfoW, MonitorFromWindow, HMONITOR, MONITORINFO, MONITOR_DEFAULTTONEAREST,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetClassNameW, GetForegroundWindow, GetWindowRect, GetWindowThreadProcessId,
    };

    /// Slack for the "covers the whole monitor" comparison. Some apps'
    /// borderless-fullscreen windows are a pixel or two off true monitor
    /// bounds (DPI rounding, a hairline border) and would otherwise never
    /// register as fullscreen.
    const TOLERANCE_PX: i32 = 2;

    fn approx_eq(a: i32, b: i32) -> bool {
        (a - b).abs() <= TOLERANCE_PX
    }

    /// Reads a window's class name (e.g. "Progman", "WorkerW") via the
    /// win32 API's fixed-buffer-plus-length-out convention.
    fn window_class_name(hwnd: HWND) -> String {
        let mut buf = [0u16; 256];
        // SAFETY: `buf` is a valid, appropriately-sized UTF-16 buffer for the
        // duration of the call; GetClassNameW writes at most buf.len() code
        // units (including the terminator budget) and returns the count
        // written, excluding the terminator.
        let len = unsafe { GetClassNameW(hwnd, buf.as_mut_ptr(), buf.len() as i32) };
        if len <= 0 {
            return String::new();
        }
        String::from_utf16_lossy(&buf[..len as usize])
    }

    /// True when `hwnd`'s rect covers the ENTIRE bounds (not just the work
    /// area — true fullscreen draws over the taskbar) of the monitor it's
    /// mostly on, within `TOLERANCE_PX`. Excludes our own process's windows
    /// (the pet/main windows are always-on-top and could otherwise
    /// self-trigger) and the desktop shell (Progman / WorkerW class —
    /// maximizing "nothing" onto an empty desktop must not read as
    /// fullscreen).
    pub fn is_foreground_fullscreen() -> bool {
        // SAFETY: GetForegroundWindow takes no arguments and can return null
        // (no window has focus, e.g. between logins) — checked immediately
        // below before any further calls that require a valid handle.
        let hwnd = unsafe { GetForegroundWindow() };
        if hwnd.is_null() {
            return false;
        }

        // Exclude our own process's windows (pet + main) so an always-on-top
        // overlay can never mistake itself for a fullscreen app underneath it.
        let mut owner_pid: u32 = 0;
        // SAFETY: `hwnd` was just returned non-null by GetForegroundWindow;
        // `owner_pid` is a valid out-pointer for the duration of the call.
        unsafe { GetWindowThreadProcessId(hwnd, &mut owner_pid) };
        // SAFETY: no preconditions beyond being callable from any thread.
        let current_pid = unsafe { GetCurrentProcessId() };
        if owner_pid != 0 && owner_pid == current_pid {
            return false;
        }

        // Exclude the desktop shell itself. Progman is the classic desktop
        // window; WorkerW is the (sometimes multiple) helper window Explorer
        // creates alongside it for desktop icons/wallpaper — either can end
        // up "foreground" with no real app focused, and either can span the
        // full monitor without that meaning anything is actually fullscreen.
        let class_name = window_class_name(hwnd);
        if class_name == "Progman" || class_name == "WorkerW" {
            return false;
        }

        let mut window_rect: RECT = unsafe { std::mem::zeroed() };
        // SAFETY: `hwnd` is a still-valid handle from just above (foreground
        // windows aren't destroyed mid-poll in practice, and a failed call
        // here just leaves window_rect zeroed, caught by the return check).
        if unsafe { GetWindowRect(hwnd, &mut window_rect) } == 0 {
            return false;
        }

        // MONITOR_DEFAULTTONEAREST: a window is never considered "on no
        // monitor" — pick whichever it's closest to/mostly overlapping,
        // matching how Windows itself decides which monitor "owns" a window.
        let hmonitor: HMONITOR =
            unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) };
        if hmonitor.is_null() {
            return false;
        }

        let mut info: MONITORINFO = unsafe { std::mem::zeroed() };
        info.cbSize = std::mem::size_of::<MONITORINFO>() as u32;
        // SAFETY: `info.cbSize` is set to the struct size as the API
        // requires before the call; `hmonitor` is a still-valid handle from
        // just above.
        if unsafe { GetMonitorInfoW(hmonitor, &mut info) } == 0 {
            return false;
        }

        // rcMonitor (not rcWork): the FULL monitor bounds, taskbar included —
        // true fullscreen covers the taskbar, so comparing against the work
        // area would make legitimate fullscreen apps fail this check.
        let m = info.rcMonitor;
        approx_eq(window_rect.left, m.left)
            && approx_eq(window_rect.top, m.top)
            && approx_eq(window_rect.right, m.right)
            && approx_eq(window_rect.bottom, m.bottom)
    }
}

/// Workstation lock/unlock detection, polled ~every 10-15s by a background
/// thread spawned in `.setup()` so the pet can perk up the moment Windows
/// unlocks. Windows-only, same rationale as `fullscreen_detect` above (the
/// pet overlay is a Windows-first feature; every other target simply never
/// starts the poller).
#[cfg(target_os = "windows")]
mod lock_detect {
    use windows_sys::Win32::Foundation::BOOL;
    use windows_sys::Win32::System::StationsAndDesktops::{
        CloseDesktop, OpenInputDesktop, DESKTOP_SWITCHDESKTOP,
    };

    /// True when the workstation is CURRENTLY locked (secure desktop active).
    ///
    /// Well-documented Win32 idiom (used by Chromium's idle-detection code
    /// and various screensaver/lock detectors): `OpenInputDesktop` opens a
    /// handle to whichever desktop is currently receiving input. On an
    /// unlocked session that's the ordinary interactive "Default" desktop,
    /// which the calling (unprivileged, same-session) process can always
    /// open. Once the workstation locks, Winlogon switches the input desktop
    /// to a secure desktop that a normal user-session process's token has no
    /// rights to open — so the call fails and returns NULL. We request only
    /// `DESKTOP_SWITCHDESKTOP` access (the minimum this check needs) rather
    /// than `GENERIC_ALL`/a full access mask, since this thread never does
    /// anything with the desktop object besides immediately closing it.
    ///
    /// NOT tried-and-rejected: `WTSRegisterSessionNotification` (delivers
    /// WM_WTSSESSION_CHANGE via the window message loop) is the "proper"
    /// event-driven alternative, but it requires subclassing a window
    /// procedure and pumping WM_WTSSESSION_CHANGE through it — Tauri/Wry
    /// owns the message loop and doesn't expose a supported hook to splice
    /// a custom WndProc branch into it. A cheap poll avoids that surgery
    /// entirely at the cost of a bounded (~10-15s) detection latency, which
    /// is fine for a cosmetic "welcome back" reaction.
    ///
    /// `OpenInputDesktop` returning NULL is, per the doc comment above, both
    /// the lock signal itself and the only way this call fails for an
    /// unprivileged same-session caller (no name to mis-resolve, no buffer
    /// to under-size) — so there is no separate "couldn't tell" outcome to
    /// distinguish from "locked" at this layer. The one genuinely-separate
    /// failure mode is `CloseDesktop` on an already-obtained non-null
    /// handle failing to close cleanly; that's a harmless handle leak, not
    /// a mis-detected state, and is logged once (not every poll) by the
    /// caller below rather than changing what this function reports.
    fn is_locked() -> bool {
        // SAFETY: no preconditions beyond being callable from any thread;
        // flags=0/inherit=FALSE mirror the documented typical usage. The
        // returned handle (if non-null) is closed below before returning.
        let desktop = unsafe { OpenInputDesktop(0, 0 as BOOL, DESKTOP_SWITCHDESKTOP) };
        if desktop.is_null() {
            return true;
        }
        // SAFETY: `desktop` was just confirmed non-null; the handle isn't
        // used again after this in this process.
        let closed = unsafe { CloseDesktop(desktop) };
        if closed == 0 {
            log_close_failure_once();
        }
        false
    }

    use std::sync::atomic::{AtomicBool, Ordering};
    /// Ensures a `CloseDesktop` failure (see `is_locked`'s doc comment) is
    /// reported at most once for this process's lifetime rather than once
    /// per ~12s poll tick for as long as it keeps failing — the task's
    /// "must never log-spam" contract.
    static CLOSE_FAILURE_LOGGED: AtomicBool = AtomicBool::new(false);
    fn log_close_failure_once() {
        if !CLOSE_FAILURE_LOGGED.swap(true, Ordering::SeqCst) {
            eprintln!("[lock-detect] CloseDesktop failed (handle leak; detection itself is unaffected)");
        }
    }

    /// Poll-loop body: returns the current locked state. `is_locked` itself
    /// has no failure path left to degrade (see its doc comment) — this
    /// thin wrapper exists so the caller has one obvious entry point and so
    /// a future genuinely-fallible detection path could be added here
    /// without changing the poll loop's shape.
    pub fn poll_locked() -> bool {
        is_locked()
    }
}

/// IPC entry point for the pet frontend's ~5s poll. Plain `bool` return
/// (not `Result`) is deliberate: every internal failure path inside
/// `fullscreen_detect` already degrades to `false` rather than propagating
/// an error, so making the command signature itself infallible is the
/// simplest way to guarantee a flaky detection call can never surface an
/// `Err` to JS and flap the pet's auto-hide on and off — worse than just
/// occasionally under-reporting fullscreen for one poll.
#[tauri::command]
fn is_foreground_fullscreen() -> bool {
    #[cfg(target_os = "windows")]
    {
        fullscreen_detect::is_foreground_fullscreen()
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

/// Set before any deliberate backend kill (tray "Quit", main-window close) so
/// the exit watcher below can tell "we killed it" from "it died on its own"
/// and stay quiet in the former case.
static SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);

/// Bundled-app origin the main window boots from (see `WebviewUrl::App` /
/// `tauri.conf.json`'s implicit "index.html"). Navigating back here after an
/// unexpected backend death re-lands on `DesktopBackendWait`
/// (`isWaitingForDesktopBackend()` in App.tsx keys off this exact host).
/// Mirrors tauri's own internal `tauri_protocol_url` (not public API):
/// `http://tauri.localhost` on Windows/Android, `tauri://localhost` elsewhere.
fn bundled_app_url() -> Url {
    let raw = if cfg!(windows) || cfg!(target_os = "android") {
        "http://tauri.localhost"
    } else {
        "tauri://localhost"
    };
    Url::parse(raw).expect("bundled app URL is valid")
}

struct BackendProcess {
    child: Arc<Mutex<Option<CommandChild>>>,
}

/// The sidecar's port, recorded once its /healthz passes — the pet window
/// (a second webview onto the same origin) can only be created after that.
struct BackendState {
    port: Arc<Mutex<Option<u16>>>,
}

/// Mirrors the pet window's click-through flag so the global shortcut (which
/// has no direct read of window state, and shouldn't need one) can flip it.
/// `set_pet_click_through` is the single writer of both the window property
/// and this flag, so the two never drift apart.
struct PetClickThroughState {
    ignoring: AtomicBool,
}

fn kill_backend_child(child: Arc<Mutex<Option<CommandChild>>>) {
    // Flip this before the kill (not after): the exit watcher task races this
    // thread and must see "deliberate" before it observes the Terminated
    // event, or it will mistake our own kill for a crash.
    SHUTTING_DOWN.store(true, Ordering::SeqCst);
    if let Some(child) = child
        .lock()
        .expect("backend process mutex poisoned")
        .take()
    {
        let _ = child.kill();
    }
}

impl Drop for BackendProcess {
    fn drop(&mut self) {
        kill_backend_child(Arc::clone(&self.child));
    }
}

#[derive(serde::Serialize)]
struct GeminiKeyStatus {
    configured: bool,
}

fn gemini_entry() -> Result<Entry, String> {
    Entry::new(KEYCHAIN_SERVICE, GEMINI_KEY_USER).map_err(|err| err.to_string())
}

#[tauri::command]
fn get_gemini_key_status() -> Result<GeminiKeyStatus, String> {
    let configured = gemini_entry()?.get_password().map(|value| !value.is_empty()).unwrap_or(false);
    Ok(GeminiKeyStatus { configured })
}

#[tauri::command]
fn set_gemini_key(value: String) -> Result<GeminiKeyStatus, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("Gemini API key cannot be empty.".to_string());
    }
    gemini_entry()?.set_password(trimmed).map_err(|err| err.to_string())?;
    Ok(GeminiKeyStatus { configured: true })
}

/// Marker file that makes the desktop app launch straight to the pet
/// overlay (main chat window hidden until asked for). Written by the
/// frontend's `set_pet_default` invoke: onboarding completion turns it on,
/// the "Start as desktop pet" settings toggle controls it afterwards. A
/// plain file (not localStorage) because the startup decision happens in
/// Rust before any webview has run a line of JS.
fn pet_default_flag_path(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .app_data_dir()
        .ok()
        .map(|dir| dir.join("pet_default"))
}

fn pet_default_enabled(app: &AppHandle) -> bool {
    pet_default_flag_path(app)
        .map(|path| path.exists())
        .unwrap_or(false)
}

#[tauri::command]
fn set_pet_default(app: AppHandle, enabled: bool) -> Result<(), String> {
    let path = pet_default_flag_path(&app).ok_or_else(|| "No app data dir".to_string())?;
    if enabled {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        std::fs::write(&path, b"1").map_err(|err| err.to_string())
    } else {
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err.to_string()),
        }
    }
}

/// "Launch at login" toggle (Settings). Deliberately a thin Rust wrapper over
/// `tauri_plugin_autostart`'s `ManagerExt` rather than the plugin's own JS
/// guest bindings (`@tauri-apps/plugin-autostart`) — this repo's convention
/// (see the the project notes gotcha this whole file lives by) is every JS-invoked
/// command going through the same #[tauri::command] + build.rs manifest +
/// capabilities-file ACL path, and using the guest bindings here would add an
/// npm dependency and a second, differently-shaped permission story just for
/// this one feature. Failures (OS registry/launch-agent write denied,
/// sandboxed install, etc.) come back as `Err(String)` for the UI to show —
/// never a panic, since a failed autostart toggle is a UX inconvenience, not
/// something that should take the app down.
#[tauri::command]
fn set_autostart(app: AppHandle, enabled: bool) -> Result<(), String> {
    use tauri_plugin_autostart::ManagerExt;
    let manager = app.autolaunch();
    if enabled {
        manager.enable()
    } else {
        manager.disable()
    }
    .map_err(|err| err.to_string())
}

#[tauri::command]
fn get_autostart(app: AppHandle) -> Result<bool, String> {
    use tauri_plugin_autostart::ManagerExt;
    app.autolaunch().is_enabled().map_err(|err| err.to_string())
}

/// Portable builds ship a `portable.marker` file next to the exe (written by
/// the packaging step). The NSIS auto-updater must never run for them — it
/// would install into Program Files and leave the portable copy stale — so
/// both updater commands check this and the UI falls back to the
/// releases-page nudge.
fn is_portable_install() -> bool {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("portable.marker").exists()))
        .unwrap_or(false)
}

#[derive(serde::Serialize)]
struct AppUpdateStatus {
    available: bool,
    version: Option<String>,
    notes: Option<String>,
    portable: bool,
}

/// Signed-update check against the latest.json endpoint in tauri.conf.json.
/// Same convention as autostart: the updater plugin's Rust API behind our own
/// command, no npm guest bindings. Errors come back as `Err(String)` for the
/// UI to swallow — an unreachable endpoint (offline, private repo) must never
/// be louder than "no update".
#[tauri::command]
async fn check_app_update(app: AppHandle) -> Result<AppUpdateStatus, String> {
    use tauri_plugin_updater::UpdaterExt;
    let portable = is_portable_install();
    let updater = app.updater().map_err(|err| err.to_string())?;
    match updater.check().await.map_err(|err| err.to_string())? {
        Some(update) => Ok(AppUpdateStatus {
            available: true,
            version: Some(update.version.clone()),
            notes: update.body.clone(),
            portable,
        }),
        None => Ok(AppUpdateStatus {
            available: false,
            version: None,
            notes: None,
            portable,
        }),
    }
}

#[derive(serde::Serialize, Clone)]
struct AppUpdateProgress {
    downloaded: u64,
    /// Content-length when the server sent one; the UI shows an
    /// indeterminate bar otherwise.
    total: Option<u64>,
    done: bool,
}

/// Download the signed update (progress via `app-update-progress` events),
/// kill the backend sidecar (the installer swaps files under it otherwise —
/// same kill the tray "Quit" path uses), then install. On Windows install()
/// exits the process and the NSIS passive installer relaunches the app; on
/// any platform where install() returns, restart explicitly.
#[tauri::command]
async fn install_app_update(app: AppHandle) -> Result<(), String> {
    use tauri_plugin_updater::UpdaterExt;
    if is_portable_install() {
        return Err(
            "portable build: download the new zip from the releases page instead".into(),
        );
    }
    let update = app
        .updater()
        .map_err(|err| err.to_string())?
        .check()
        .await
        .map_err(|err| err.to_string())?
        .ok_or_else(|| "no update available".to_string())?;

    let progress_app = app.clone();
    let mut downloaded: u64 = 0;
    let bytes = update
        .download(
            move |chunk, total| {
                downloaded += chunk as u64;
                let _ = progress_app.emit(
                    "app-update-progress",
                    AppUpdateProgress {
                        downloaded,
                        total,
                        done: false,
                    },
                );
            },
            || {},
        )
        .await
        .map_err(|err| err.to_string())?;
    let _ = app.emit(
        "app-update-progress",
        AppUpdateProgress {
            downloaded: bytes.len() as u64,
            total: Some(bytes.len() as u64),
            done: true,
        },
    );

    if let Some(state) = app.try_state::<BackendProcess>() {
        kill_backend_child(Arc::clone(&state.child));
    }
    update.install(bytes).map_err(|err| err.to_string())?;
    app.restart()
}

fn show_main_window_impl(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

/// Pet window's route into the full chat window (settings, transcript,
/// memory panel). Showing an existing window is safe from the IPC pool —
/// only webview CREATION needs the main thread (see toggle_pet_window).
#[tauri::command]
fn show_main_window(app: AppHandle) -> Result<(), String> {
    show_main_window_impl(&app);
    Ok(())
}

/// Saved outer position + inner size for the pet window, physical pixels.
/// The window is undecorated (no OS chrome), so outer and inner size differ
/// only by whatever border the compositor still draws — close enough for the
/// `pet_rect_is_visible` reachability check below, which only needs a rough
/// top strip, not pixel-exact bounds.
/// The pet is deliberately kept OFF the window-state plugin's tracked set
/// (see the denylist comment on the plugin builder below) because that
/// plugin restores blindly — a saved position that's since gone off-screen
/// (monitor unplugged, resolution changed) would strand an undecorated,
/// always-on-top window with no chrome to drag it back by. This bespoke
/// version restores the same convenience but only after `pet_rect_is_visible`
/// confirms the saved spot still lands somewhere reachable.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug)]
struct PetRect {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

fn pet_window_state_path(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .app_data_dir()
        .ok()
        .map(|dir| dir.join("pet_window.json"))
}

fn load_pet_rect(app: &AppHandle) -> Option<PetRect> {
    let path = pet_window_state_path(app)?;
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn save_pet_rect(app: &AppHandle, rect: PetRect) {
    let Some(path) = pet_window_state_path(app) else {
        return;
    };
    if let Some(parent) = path.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return;
        }
    }
    if let Ok(bytes) = serde_json::to_vec(&rect) {
        // Best-effort: a failed write just means next launch falls back to
        // the default spot, which is the same outcome as never having saved.
        let _ = std::fs::write(path, bytes);
    }
}

/// Latest observed pet window rect plus a generation counter, shared between
/// the window-event handler (writer, every Moved/Resized) and the debounce
/// timer thread (reader, fires ~800ms after the last write). Moves/resizes
/// fire continuously during a drag — writing the JSON file on every one of
/// them would hammer disk for no benefit, so only the LAST rect in a burst
/// ever gets persisted. `generation` lets the timer notice "nothing changed
/// since I was scheduled" and skip a redundant write instead of comparing
/// full struct equality.
struct PetRectDebounce {
    latest: Mutex<Option<PetRect>>,
    generation: Arc<std::sync::atomic::AtomicU64>,
}

const PET_RECT_DEBOUNCE_MS: u64 = 800;

/// Record the pet window's current outer rect and (re)arm the debounce timer.
/// Called from Moved/Resized window events — cheap (a mutex store + spawning
/// a sleep thread that usually finds itself superseded and exits without
/// touching disk).
fn schedule_pet_rect_save(app: &AppHandle, rect: PetRect) {
    let Some(state) = app.try_state::<PetRectDebounce>() else {
        return;
    };
    *state.latest.lock().expect("pet rect mutex poisoned") = Some(rect);
    let my_generation = state.generation.fetch_add(1, Ordering::SeqCst) + 1;
    let generation = Arc::clone(&state.generation);
    let handle = app.clone();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(PET_RECT_DEBOUNCE_MS));
        // A later move/resize bumped the generation while we slept — that
        // call's own timer will do the flush, so bail out and let it.
        if generation.load(Ordering::SeqCst) != my_generation {
            return;
        }
        flush_pet_rect(&handle);
    });
}

/// Write whatever rect is currently pending, if any. Safe to call redundantly
/// (e.g. once from the debounce timer and again from Focused(false)) — it
/// just re-saves the same value the second time. This is also THE moment a
/// drag is considered "over" (the debounce timer only reaches here once
/// nothing else has moved the window for `PET_RECT_DEBOUNCE_MS`), so the
/// edge-perch snap runs here too, before persistence — the snapped rect is
/// what gets saved, not the pre-snap drop point, so a relaunch restores the
/// perched position rather than the spot the user's cursor happened to
/// release at.
fn flush_pet_rect(app: &AppHandle) {
    let Some(state) = app.try_state::<PetRectDebounce>() else {
        return;
    };
    let rect = *state.latest.lock().expect("pet rect mutex poisoned");
    if let Some(rect) = rect {
        let rect = apply_perch_snap(app, rect);
        save_pet_rect(app, rect);
    }
}

/// True when a meaningful part of the window's TOP strip — the only place an
/// undecorated pet window can be grabbed and dragged — lands within some
/// monitor's work area. Requires at least `MIN_STRIP` px of overlap in both
/// axes so a rect that's only barely clipping a monitor corner (a pixel or
/// two, effectively unreachable) doesn't count as valid. All arithmetic is
/// in physical pixels, matching both `PetRect` and `Monitor::work_area()`.
const MIN_STRIP: i32 = 40;

fn pet_rect_is_visible(rect: &PetRect, monitors: &[tauri::Monitor]) -> bool {
    // Candidate drag-bar strip: the top MIN_STRIP px of the window. Height
    // only — the overlap check below uses the window's FULL width so a wide
    // window straddling a monitor edge still counts once enough of that top
    // strip clears MIN_STRIP horizontally.
    let strip_h = rect.height.min(MIN_STRIP as u32) as i32;
    let top_right_x = rect.x + rect.width as i32;
    let strip_y1 = rect.y + strip_h;
    monitors.iter().any(|monitor| {
        // work_area() rather than position()/size(): excludes the taskbar
        // and similar reserved OS chrome, so a saved rect that's technically
        // "on" a monitor but hidden behind the taskbar still fails this
        // check and falls back to the default spot instead of restoring
        // somewhere the user can't actually see or grab it.
        let area = monitor.work_area();
        let wx0 = area.position.x;
        let wy0 = area.position.y;
        let wx1 = area.position.x + area.size.width as i32;
        let wy1 = area.position.y + area.size.height as i32;
        // Horizontal overlap between the window's full width and the
        // monitor's work area, clamped — then require at least MIN_STRIP.
        let overlap_x0 = rect.x.max(wx0);
        let overlap_x1 = top_right_x.min(wx1);
        let overlap_w = (overlap_x1 - overlap_x0).max(0);
        // Vertical overlap between the window's top strip and the work area.
        let overlap_y0 = rect.y.max(wy0);
        let overlap_y1 = strip_y1.min(wy1);
        let overlap_h = (overlap_y1 - overlap_y0).max(0);
        overlap_w >= MIN_STRIP && overlap_h >= MIN_STRIP
    })
}

/// Which screen edge the pet window is magnetically snapped to. Top is
/// deliberately not a variant: the drag bar lives along the window's top
/// edge, so snapping there would fight the user every time they pick the pet
/// up again. Serializes as a lowercase string so the frontend's poll of
/// `pet_perch_state` gets a plain "bottom"/"left"/"right" without a manual
/// match on the Rust side.
#[derive(serde::Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum PerchEdge {
    Bottom,
    Left,
    Right,
}

/// How close (physical px) a drag has to end to a work-area edge before it
/// magnetically snaps. Generous enough to catch "drag roughly hovering the
/// taskbar" without being so wide that it snaps drags that were only ever
/// passing near an edge on their way somewhere else.
const PERCH_SNAP_PX: i32 = 28;

/// Current perch, shared between the drag-settle snap logic (writer) and the
/// `pet_perch_state` poll command (reader). `None` means "not perched" —
/// the pet is free-floating somewhere in the middle of a monitor.
struct PerchState(Mutex<Option<PerchEdge>>);

/// Finds whichever monitor's work area contains the window's current
/// top-left corner, falling back to the first available monitor. Mirrors the
/// matching logic `build_pet_window` already uses for its scale-factor
/// lookup on restore, factored out here so the perch snap (which runs on
/// every drag-settle, not just at startup) shares the same "which monitor is
/// this rect actually on" rule instead of drifting from it over time.
fn containing_monitor<'a>(
    rect: &PetRect,
    monitors: &'a [tauri::Monitor],
) -> Option<&'a tauri::Monitor> {
    monitors
        .iter()
        .find(|m| {
            let p = m.position();
            let s = m.size();
            rect.x >= p.x
                && rect.x < p.x + s.width as i32
                && rect.y >= p.y
                && rect.y < p.y + s.height as i32
        })
        .or_else(|| monitors.first())
}

/// Given the window's resting rect and the work area of the monitor it's on,
/// decides whether it should snap and to which edge, returning the
/// snapped rect when it does. Pure function (no window/IO side effects) so
/// the snap decision and the "what rect do we actually persist" question
/// share one source of truth instead of the caller recomputing edges twice.
fn compute_perch_snap(
    rect: &PetRect,
    work_area: &tauri::PhysicalRect<i32, u32>,
) -> Option<(PerchEdge, PetRect)> {
    // No wy0 (work area top): top is not a snap edge (see PerchEdge's doc
    // comment — the drag bar lives there), so the top boundary of the work
    // area never enters this comparison.
    let wx0 = work_area.position.x;
    let wx1 = work_area.position.x + work_area.size.width as i32;
    let wy1 = work_area.position.y + work_area.size.height as i32;

    let rect_right = rect.x + rect.width as i32;
    let rect_bottom = rect.y + rect.height as i32;

    let dist_bottom = (wy1 - rect_bottom).abs();
    let dist_left = (rect.x - wx0).abs();
    let dist_right = (wx1 - rect_right).abs();

    // Bottom takes priority over left/right when a corner is close to both
    // (e.g. dragged into the bottom-left corner) — sitting "on the taskbar"
    // reads as the more natural resting pose than flush against a side wall.
    if dist_bottom <= PERCH_SNAP_PX {
        let mut snapped = *rect;
        snapped.y = wy1 - rect.height as i32;
        return Some((PerchEdge::Bottom, snapped));
    }
    if dist_left <= PERCH_SNAP_PX {
        let mut snapped = *rect;
        snapped.x = wx0;
        return Some((PerchEdge::Left, snapped));
    }
    if dist_right <= PERCH_SNAP_PX {
        let mut snapped = *rect;
        snapped.x = wx1 - rect.width as i32;
        return Some((PerchEdge::Right, snapped));
    }
    None
}

/// Runs at the same moment a drag is judged "over" — called from
/// `flush_pet_rect` just before the rect is persisted, so a snap changes
/// what actually gets saved instead of racing a separate write. Moves the
/// live window to the snapped position (best-effort; a failed `set_position`
/// just means the pet stays where it visually was one frame while state
/// still reports it perched, negligible) and updates `PerchState`. Returns
/// the rect that should be persisted: the snapped one when perching,
/// otherwise the input unchanged.
fn apply_perch_snap(app: &AppHandle, rect: PetRect) -> PetRect {
    let Some(perch_state) = app.try_state::<PerchState>() else {
        return rect;
    };
    let monitors = app.available_monitors().unwrap_or_default();
    let Some(monitor) = containing_monitor(&rect, &monitors) else {
        *perch_state.0.lock().expect("perch state mutex poisoned") = None;
        return rect;
    };
    let work_area = monitor.work_area();

    match compute_perch_snap(&rect, work_area) {
        Some((edge, snapped)) => {
            *perch_state.0.lock().expect("perch state mutex poisoned") = Some(edge);
            if let Some(window) = app.get_webview_window(PET_WINDOW_LABEL) {
                let _ = window.set_position(tauri::Position::Physical(tauri::PhysicalPosition {
                    x: snapped.x,
                    y: snapped.y,
                }));
            }
            snapped
        }
        None => {
            *perch_state.0.lock().expect("perch state mutex poisoned") = None;
            rect
        }
    }
}

#[derive(serde::Serialize)]
struct PetPerchStateResponse {
    perched: bool,
    edge: Option<PerchEdge>,
}

/// Polled by the frontend right after its own drag-settle detection fires,
/// so it knows whether to visually pose the avatar as sitting on an edge
/// (legs dangling off the taskbar, leaning on a side wall) versus the normal
/// floating idle pose. Read-only — the snap decision itself always happens
/// server-side inside `apply_perch_snap`, this just reports the outcome.
#[tauri::command]
fn pet_perch_state(app: AppHandle) -> PetPerchStateResponse {
    let edge = app
        .try_state::<PerchState>()
        .and_then(|state| *state.0.lock().expect("perch state mutex poisoned"));
    PetPerchStateResponse {
        perched: edge.is_some(),
        edge,
    }
}

/// Cancellation token for the roam tween (`animate_pet_window_to`): a bare
/// generation counter, same idea as `PetRectDebounce.generation`. Starting a
/// new tween bumps it and captures the new value; the tween thread bails the
/// moment the stored value no longer matches (either a newer tween started or
/// `cancel_pet_window_animation` bumped it). Arc'd because the tween loop
/// runs on a blocking-pool thread that outlives the command's borrow of
/// managed state.
struct PetRoamAnim {
    generation: Arc<std::sync::atomic::AtomicU64>,
}

/// Geometry snapshot the pet frontend's roam scheduler plans a stroll from:
/// the window's current rect plus the work area of the monitor it's on, all
/// physical pixels (matching `PetRect` / `Monitor::work_area()` — no unit
/// conversion on either side of the IPC boundary, same convention as
/// `pet_cursor_context`). The JS side picks the perch target; the Rust side
/// (`animate_pet_window_to`) re-clamps it anyway, so a stale snapshot can
/// never push the window off-screen.
/// One monitor's work area, for the multi-monitor roam extension.
#[derive(serde::Serialize)]
struct RoamMonitorArea {
    x: i32,
    y: i32,
    w: u32,
    h: u32,
}

#[derive(serde::Serialize)]
struct PetRoamContext {
    win_x: i32,
    win_y: i32,
    win_w: u32,
    win_h: u32,
    work_x: i32,
    work_y: i32,
    work_w: u32,
    work_h: u32,
    /// Work areas of ALL monitors (the containing one included), so the JS
    /// roam scheduler can occasionally wander to a different screen.
    /// Additive: older frontends simply never read it.
    monitors: Vec<RoamMonitorArea>,
}

#[tauri::command]
fn pet_roam_context(app: AppHandle) -> Result<PetRoamContext, String> {
    let window = app
        .get_webview_window(PET_WINDOW_LABEL)
        .ok_or_else(|| "Pet window is not open".to_string())?;
    let position = window.outer_position().map_err(|err| err.to_string())?;
    // inner_size, matching PetRect's convention (outer position + inner
    // size — see PetRect's doc comment on why that mix is fine for an
    // undecorated window).
    let size = window.inner_size().map_err(|err| err.to_string())?;
    let rect = PetRect {
        x: position.x,
        y: position.y,
        width: size.width,
        height: size.height,
    };
    let monitors = app.available_monitors().unwrap_or_default();
    // Same "which monitor is this rect on" rule as the rect-restore and the
    // perch snap — shared via containing_monitor rather than a third copy.
    let monitor = containing_monitor(&rect, &monitors)
        .ok_or_else(|| "No monitor found".to_string())?;
    let area = monitor.work_area();
    let monitor_areas = monitors
        .iter()
        .map(|m| {
            let a = m.work_area();
            RoamMonitorArea {
                x: a.position.x,
                y: a.position.y,
                w: a.size.width,
                h: a.size.height,
            }
        })
        .collect();
    Ok(PetRoamContext {
        win_x: rect.x,
        win_y: rect.y,
        win_w: rect.width,
        win_h: rect.height,
        work_x: area.position.x,
        work_y: area.position.y,
        work_w: area.size.width,
        work_h: area.size.height,
        monitors: monitor_areas,
    })
}

/// Clamp with a degenerate-range guard: a window larger than the work area
/// makes `max < min`, in which case pinning to `min` (the work-area origin)
/// beats a `clamp` panic or an off-screen coordinate.
fn clamp_axis(value: i32, min: i32, max: i32) -> i32 {
    if max < min {
        min
    } else {
        value.clamp(min, max)
    }
}

/// Clamps a candidate top-left so the WHOLE window stays inside the work
/// area (taskbar excluded — same reachability reasoning as
/// `pet_rect_is_visible`). Pure, so the roam tween's target sanitation is
/// testable without a window.
fn clamp_to_work_area(
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    work_area: &tauri::PhysicalRect<i32, u32>,
) -> (i32, i32) {
    let wx0 = work_area.position.x;
    let wy0 = work_area.position.y;
    let wx1 = wx0 + work_area.size.width as i32;
    let wy1 = wy0 + work_area.size.height as i32;
    (
        clamp_axis(x, wx0, wx1 - width as i32),
        clamp_axis(y, wy0, wy1 - height as i32),
    )
}

/// Smoothstep ease-in-out: a stroll should read as "sets off and settles",
/// not a linear conveyor-belt slide.
fn ease_in_out(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Tween cadence (~30Hz) and duration bounds for `animate_pet_window_to`.
/// The duration clamp is a server-side sanity bound, not the pacing policy —
/// the frontend's roam module owns how slow a stroll should feel.
const ROAM_TICK_MS: u64 = 33;
const ROAM_MIN_DURATION_MS: u64 = 200;
const ROAM_MAX_DURATION_MS: u64 = 60_000;
/// If the window's observed position drifts more than this from where the
/// tween last put it, someone else moved it (the user grabbed the drag bar
/// mid-stroll, or the OS relocated it) — the tween must yield instantly
/// rather than fight the drag. A couple px of slack covers DPI rounding.
const ROAM_EXTERNAL_MOVE_SLACK_PX: i32 = 3;

/// Tween the pet window to (x, y) over `duration_ms`, easing in/out at
/// ~30Hz. Returns Ok(true) when the stroll completed, Ok(false) when it was
/// cancelled (newer tween, `cancel_pet_window_animation`, an external move,
/// or the window closing mid-stroll). The target is clamped to the work area
/// of the monitor the window starts on, reusing the perch/rect-restore
/// monitor logic — a bad target from JS degrades to a shorter stroll, never
/// an off-screen window.
///
/// Persistence rides the EXISTING path for free: every `set_position` tick
/// fires a Moved event, whose handler debounces into `flush_pet_rect` —
/// which edge-snaps (`apply_perch_snap`) and saves ~800ms after the last
/// tick, exactly as if the user had dragged the window there. No separate
/// save path exists for roaming, so a restart always restores the same kind
/// of snapped, validated perch a manual drag would have left.
///
/// `async` for the same reason as `toggle_pet_window`: this holds the IPC
/// call open for the stroll's whole duration, and the loop itself runs on
/// the blocking pool (set_position just dispatches to the event loop, safe
/// off-thread — `apply_perch_snap` already calls it from a timer thread).
#[tauri::command]
async fn animate_pet_window_to(
    app: AppHandle,
    x: i32,
    y: i32,
    duration_ms: u64,
) -> Result<bool, String> {
    let window = app
        .get_webview_window(PET_WINDOW_LABEL)
        .ok_or_else(|| "Pet window is not open".to_string())?;
    let start = window.outer_position().map_err(|err| err.to_string())?;
    let size = window.inner_size().map_err(|err| err.to_string())?;
    let rect = PetRect {
        x: start.x,
        y: start.y,
        width: size.width,
        height: size.height,
    };
    let monitors = app.available_monitors().unwrap_or_default();
    let monitor = containing_monitor(&rect, &monitors)
        .ok_or_else(|| "No monitor found".to_string())?;
    // Clamp against the monitor the TARGET lands on — cross-monitor strolls
    // are legal since the multi-monitor roam pass; the old current-monitor
    // clamp silently bounced every off-screen target back. A target in the
    // dead zone between monitors falls back to the current monitor's area.
    let target_rect = PetRect {
        x,
        y,
        width: size.width,
        height: size.height,
    };
    let target_monitor = containing_monitor(&target_rect, &monitors).unwrap_or(monitor);
    let (target_x, target_y) =
        clamp_to_work_area(x, y, size.width, size.height, target_monitor.work_area());
    let duration = duration_ms.clamp(ROAM_MIN_DURATION_MS, ROAM_MAX_DURATION_MS);

    let generation = Arc::clone(&app.state::<PetRoamAnim>().generation);
    // Starting a tween implicitly cancels any previous one: bump-and-capture
    // means the older loop's next generation check fails.
    let my_generation = generation.fetch_add(1, Ordering::SeqCst) + 1;

    let completed = async_runtime::spawn_blocking(move || {
        let started_at = Instant::now();
        let mut last_set: Option<(i32, i32)> = None;
        loop {
            if generation.load(Ordering::SeqCst) != my_generation {
                return false;
            }
            // Yield to a user drag: if the window isn't where this tween
            // last put it, someone else is moving it.
            if let Some((lx, ly)) = last_set {
                match window.outer_position() {
                    Ok(pos)
                        if (pos.x - lx).abs() > ROAM_EXTERNAL_MOVE_SLACK_PX
                            || (pos.y - ly).abs() > ROAM_EXTERNAL_MOVE_SLACK_PX =>
                    {
                        return false;
                    }
                    Ok(_) => {}
                    // Window gone (closed mid-stroll) — nothing left to move.
                    Err(_) => return false,
                }
            }
            let t = started_at.elapsed().as_millis() as f64 / duration as f64;
            let eased = ease_in_out(t);
            let nx = start.x + ((target_x - start.x) as f64 * eased).round() as i32;
            let ny = start.y + ((target_y - start.y) as f64 * eased).round() as i32;
            if window
                .set_position(tauri::Position::Physical(tauri::PhysicalPosition {
                    x: nx,
                    y: ny,
                }))
                .is_err()
            {
                return false;
            }
            last_set = Some((nx, ny));
            if t >= 1.0 {
                return true;
            }
            thread::sleep(Duration::from_millis(ROAM_TICK_MS));
        }
    })
    .await
    .unwrap_or(false);
    Ok(completed)
}

/// Cancels an in-flight roam tween (no-op when none is running). The tween
/// notices within one ~33ms tick; the partially-travelled position then
/// settles through the normal Moved-event debounce → snap → persist path
/// like any interrupted drag would.
#[tauri::command]
fn cancel_pet_window_animation(app: AppHandle) {
    if let Some(state) = app.try_state::<PetRoamAnim>() {
        state.generation.fetch_add(1, Ordering::SeqCst);
    }
}

/// Inline splash shown the instant the pet window opens on a pet-default
/// launch, before the backend sidecar has even been spawned. Everything is
/// self-contained (no external resources, no fetch) — it ships as a `data:`
/// URL (see `data_html_url`), so it must survive with zero network/filesystem
/// access. Visual language deliberately echoes `.pet-drag-bar` /
/// `.pet-drag-title` in PetView.css (monospace, rgba(255,255,255,0.7),
/// generous letter-spacing) so the splash reads as "the same app, still
/// booting" rather than a different, jarring screen. `overflow: hidden` and a
/// fully transparent `html`/`body` background matter as much as the pet
/// window's own `.transparent(true)` builder flag — either half missing
/// paints a white/black rectangle for the whole boot instead of the
/// see-through overlay the pet is supposed to be.
const SPLASH_HTML: &str = r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    background: transparent;
    overflow: hidden;
  }
  body {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    -webkit-user-select: none;
    user-select: none;
  }
  .dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.7);
    box-shadow: 0 0 18px 4px rgba(255, 255, 255, 0.25);
    animation: pulse 2.2s ease-in-out infinite;
  }
  .label {
    font-size: 10px;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.7);
  }
  @keyframes pulse {
    0%, 100% { transform: scale(0.85); opacity: 0.45; }
    50% { transform: scale(1.15); opacity: 1; }
  }
</style>
</head>
<body>
  <div class="dot"></div>
  <div class="label">waking up&hellip;</div>
</body>
</html>"#;

/// Shown only if the health-wait deadline (`wait_for_health`) is reached
/// without the backend ever answering — see `launch_backend`'s pet-default
/// branch. Same aesthetic as `SPLASH_HTML` (no separate visual language for
/// "it's broken" — just different words and no pulse, since animating a dead
/// state is misleading) so the window never sits on "waking up..." forever
/// with no way for the user to tell the boot actually failed.
const SPLASH_ERROR_HTML: &str = r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    background: transparent;
    overflow: hidden;
  }
  body {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    text-align: center;
    padding: 0 24px;
    -webkit-user-select: none;
    user-select: none;
  }
  .dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: rgba(239, 71, 111, 0.75);
    box-shadow: 0 0 18px 4px rgba(239, 71, 111, 0.25);
  }
  .label {
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.7);
  }
  .sub {
    font-size: 9px;
    letter-spacing: 0.08em;
    color: rgba(255, 255, 255, 0.45);
  }
</style>
</head>
<body>
  <div class="dot"></div>
  <div class="label">backend failed to start</div>
  <div class="sub">check the logs, or reinstall</div>
</body>
</html>"#;

/// Encodes inline HTML as a `data:text/html;base64,...` URL. Base64 (not raw
/// percent-encoding) because it has no character-escaping edge cases to get
/// wrong for arbitrary HTML/CSS content — the whole payload round-trips
/// byte-for-byte regardless of what punctuation the splash markup contains.
/// Requires the `tauri`/`webview-data-url` feature (Cargo.toml) — without it
/// `WebviewUrl::External`/`navigate()` reject `data:` scheme URLs at runtime
/// with `InvalidWebviewUrl`, not a compile error, so this would fail silently
/// in a build that forgot the feature flag.
fn data_html_url(html: &str) -> Url {
    let encoded = base64::engine::general_purpose::STANDARD.encode(html.as_bytes());
    Url::parse(&format!("data:text/html;base64,{encoded}")).expect("data URL is valid")
}

fn pet_live_url(port: u16) -> Url {
    format!("http://127.0.0.1:{port}/?view=pet")
        .parse()
        .expect("pet URL is valid")
}

/// Shared window-shell builder for the pet overlay: title/size/transparency/
/// always-on-top/browser-args mirroring/rect-restore/persistence handlers are
/// identical whether the window opens on the live backend URL (the
/// already-healthy `toggle_pet_window` path) or on the splash URL first (the
/// pet-default instant-on path in `launch_backend`, navigated to the live URL
/// once health passes). Only the initial `WebviewUrl` differs between
/// callers, so it's the one thing factored out as a parameter instead of
/// being baked into this function.
fn build_pet_window_with_url(app: &AppHandle, url: WebviewUrl) -> tauri::Result<WebviewWindow> {
    let mut builder = WebviewWindowBuilder::new(app, PET_WINDOW_LABEL, url)
    .title("Kurisu")
    .inner_size(380.0, 560.0)
    .transparent(true)
    .decorations(false)
    .shadow(false)
    .always_on_top(true)
    .skip_taskbar(true)
    .resizable(true);
    // WebView2 runs ONE browser environment per app, created with the FIRST
    // webview's browser arguments — and every later webview must request the
    // exact same ones, or its controller creation fails silently (the window
    // handle exists but the webview never attaches; every method on it
    // errors with FailedToReceiveMessage). The main window sets
    // additionalBrowserArgs in tauri.conf.json (the WebView2 mic-permission
    // fix), so mirror those args here from config instead of hardcoding a
    // copy that would drift.
    if let Some(args) = app
        .config()
        .app
        .windows
        .iter()
        .find(|w| w.label == "main")
        .and_then(|w| w.additional_browser_args.clone())
    {
        builder = builder.additional_browser_args(&args);
    } else {
        // A miss here means the pet webview gets WebView2's default args —
        // which re-enable the gesture-gated autoplay policy, so every pet
        // line plays silently. Loud log so a config restructure that breaks
        // the label/field lookup doesn't ship as "voice sometimes off".
        eprintln!(
            "[pet] additionalBrowserArgs not found on the 'main' window config; \
             pet webview will use WebView2 defaults (autoplay likely blocked)"
        );
    }
    // Restore the last saved position/size, but only when it still lands
    // somewhere reachable — see `pet_rect_is_visible`'s doc comment for why
    // this bespoke check exists instead of just un-denylisting the pet from
    // tauri-plugin-window-state. `inner_size`/`position` below take LOGICAL
    // units, so a physical-pixel saved rect must be divided by a scale
    // factor first; the window doesn't exist yet to ask its own, so use
    // whichever monitor's bounds contain the saved top-left corner (falling
    // back to the first available monitor if none match, e.g. saved
    // top-left sat exactly on a boundary due to prior rounding).
    if let Some(rect) = load_pet_rect(app) {
        let monitors = app.available_monitors().unwrap_or_default();
        if pet_rect_is_visible(&rect, &monitors) {
            // `containing_monitor` also backs the edge-perch snap
            // (`apply_perch_snap`) — sharing it here means both "which
            // monitor is this rect on" questions answer identically instead
            // of two independently-maintained copies of the same match.
            let scale = containing_monitor(&rect, &monitors)
                .map(|m| m.scale_factor())
                .unwrap_or(1.0);
            builder = builder
                .position(rect.x as f64 / scale, rect.y as f64 / scale)
                .inner_size(rect.width as f64 / scale, rect.height as f64 / scale);
        }
    }
    let window = builder
        .build()
        .inspect_err(|err| eprintln!("[pet] window creation failed: {err}"))?;
    // Debounced persistence for future launches: Moved and Resized both fire
    // continuously while the user drags/drags-to-resize, so each handler call
    // only records the latest rect and (re)arms a timer — see
    // `schedule_pet_rect_save`'s doc comment. Focused(false) and
    // CloseRequested flush immediately so a quick drag-then-close within the
    // debounce window still gets saved instead of losing the last move.
    let persist_window = window.clone();
    window.on_window_event(move |event| match event {
        tauri::WindowEvent::Moved(position) => {
            if let Ok(size) = persist_window.inner_size() {
                schedule_pet_rect_save(
                    persist_window.app_handle(),
                    PetRect {
                        x: position.x,
                        y: position.y,
                        width: size.width,
                        height: size.height,
                    },
                );
            }
        }
        tauri::WindowEvent::Resized(_size) => {
            // Use inner_size() rather than the event's own payload: Resized
            // reports the new INNER size already, but re-querying keeps this
            // arm symmetric with Moved above and immune to any future Tauri
            // change in what the event payload represents.
            if let (Ok(position), Ok(size)) =
                (persist_window.outer_position(), persist_window.inner_size())
            {
                schedule_pet_rect_save(
                    persist_window.app_handle(),
                    PetRect {
                        x: position.x,
                        y: position.y,
                        width: size.width,
                        height: size.height,
                    },
                );
            }
        }
        tauri::WindowEvent::Focused(false) | tauri::WindowEvent::CloseRequested { .. } => {
            flush_pet_rect(persist_window.app_handle());
        }
        _ => {}
    });
    Ok(window)
}

/// The ordinary, post-health entry point (tray toggle, HUD button): the
/// backend is already known-healthy (its port is in `BackendState`) by the
/// time anything can call this, so there is nothing to splash — build
/// straight on the live URL, same as before this file grew a splash path.
fn build_pet_window(app: &AppHandle, port: u16) -> tauri::Result<WebviewWindow> {
    build_pet_window_with_url(app, WebviewUrl::External(pet_live_url(port)))
}

/// Instant-on entry point for a pet-default launch: creates the pet window
/// immediately, before the backend sidecar has even been spawned, showing
/// `SPLASH_HTML` so the app never appears to do nothing but sit in the tray
/// for the whole boot. Caller (`launch_backend`) is responsible for calling
/// `window.navigate(...)` once health resolves either way — this function
/// only ever produces the splash-first window, never the live one, so there
/// is exactly one place (the health-wait continuation) that decides what the
/// splash turns into.
fn build_pet_splash_window(app: &AppHandle) -> tauri::Result<WebviewWindow> {
    build_pet_window_with_url(app, WebviewUrl::External(data_html_url(SPLASH_HTML)))
}

/// Open or close the desktop-pet window. Returns true when it is now open.
/// Shared by the tray handler (event-loop thread, fine to call directly) and
/// the async IPC command below.
fn toggle_pet_window_impl(app: &AppHandle) -> Result<bool, String> {
    if let Some(window) = app.get_webview_window(PET_WINDOW_LABEL) {
        window.close().map_err(|err| err.to_string())?;
        return Ok(false);
    }
    let port = app
        .state::<BackendState>()
        .port
        .lock()
        .expect("backend port mutex poisoned")
        .ok_or_else(|| "Backend is not ready yet".to_string())?;
    build_pet_window(app, port).map_err(|err| err.to_string())?;
    Ok(true)
}

/// IPC entry point for the HUD pet button. Two thread constraints, both hit
/// in the wild on Windows:
/// - MUST stay `async`: a synchronous command runs inside the webview's IPC
///   dispatch, and creating a webview window from there deadlocks — the
///   invoke promise never settles.
/// - The creation itself must run on the event-loop thread: done straight
///   from the async pool, the OS window appears but its webview never
///   attaches (empty transparent shell). So dispatch to the main thread and
///   wait for the result over a channel.
#[tauri::command]
async fn toggle_pet_window(app: AppHandle) -> Result<bool, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = app.clone();
    app.run_on_main_thread(move || {
        let _ = tx.send(toggle_pet_window_impl(&handle));
    })
    .map_err(|err| err.to_string())?;
    rx.recv().map_err(|err| err.to_string())?
}

/// Click-through: while enabled the pet window is unclickable, so recovery
/// lives in the tray menu ("Disable click-through") and the global shortcut
/// (CmdOrCtrl+Shift+K, see `register_click_through_shortcut`) — both call
/// this same function so the window property and `PetClickThroughState`
/// (the shortcut's only way to know the CURRENT state, since it has no
/// window to query) never disagree about which state is active.
#[tauri::command]
fn set_pet_click_through(app: AppHandle, enabled: bool) -> Result<(), String> {
    let window = app
        .get_webview_window(PET_WINDOW_LABEL)
        .ok_or_else(|| "Pet window is not open".to_string())?;
    window
        .set_ignore_cursor_events(enabled)
        .map_err(|err| err.to_string())?;
    if let Some(state) = app.try_state::<PetClickThroughState>() {
        state.ignoring.store(enabled, Ordering::SeqCst);
    }
    Ok(())
}

/// Toggle click-through off a fixed shortcut rather than a command arg: the
/// shortcut has no JS caller to hand it the desired new state, so it reads
/// `PetClickThroughState` and flips it. No-ops (does not register a state
/// flip) when the pet window doesn't exist — nothing to toggle.
fn toggle_pet_click_through_impl(app: &AppHandle) {
    let Some(_window) = app.get_webview_window(PET_WINDOW_LABEL) else {
        return;
    };
    let Some(state) = app.try_state::<PetClickThroughState>() else {
        return;
    };
    let currently_ignoring = state.ignoring.load(Ordering::SeqCst);
    if let Err(err) = set_pet_click_through(app.clone(), !currently_ignoring) {
        eprintln!("[pet] click-through shortcut toggle failed: {err}");
    }
}

/// Ambient desktop signals for presence: focused-window title, battery
/// state, and now-playing metadata — the zero-cost awareness layer. The
/// frontend polls this every ~15s, decides which TRANSITIONS are
/// noteworthy (long focus, low battery, track change), and asks the
/// backend for at most an occasional line; this command itself never
/// talks to the LLM. Windows-only like `fullscreen_detect`; other
/// platforms answer all-None and the feature quietly doesn't exist.
#[cfg(target_os = "windows")]
mod ambient_detect {
    use windows_sys::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
    use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowTextW};

    pub fn foreground_title() -> Option<String> {
        // SAFETY: GetForegroundWindow takes no arguments and can return
        // null (no focused window) — checked before any use.
        let hwnd = unsafe { GetForegroundWindow() };
        if hwnd.is_null() {
            return None;
        }
        let mut buf = [0u16; 512];
        // SAFETY: `buf` is a valid, appropriately-sized UTF-16 buffer for
        // the duration of the call; GetWindowTextW writes at most buf.len()
        // code units and returns the count written, sans terminator.
        let len = unsafe { GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32) };
        if len <= 0 {
            return None;
        }
        Some(String::from_utf16_lossy(&buf[..len as usize]))
    }

    pub fn battery() -> (Option<u8>, Option<bool>) {
        let mut status: SYSTEM_POWER_STATUS = unsafe { std::mem::zeroed() };
        // SAFETY: `status` is a valid out-pointer for the duration of the
        // call; a zero return means the struct is unreliable and is
        // reported as unknown rather than read.
        if unsafe { GetSystemPowerStatus(&mut status) } == 0 {
            return (None, None);
        }
        // 255 = "unknown" per the API contract for both fields below.
        let percent =
            (status.BatteryLifePercent != 255).then_some(status.BatteryLifePercent);
        let charging = match status.ACLineStatus {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        };
        (percent, charging)
    }

    /// Now-playing via the System Media Transport Controls session manager
    /// (covers Spotify, browsers, most players). Resolves the WinRT async
    /// ops synchronously — milliseconds in practice — which is why the
    /// `ambient_signals` command is async (worker thread), never a
    /// main-thread sync one. "No active media session" is the common case
    /// and reads as all-None, not an error.
    pub fn now_playing() -> (Option<String>, Option<String>, Option<bool>) {
        use windows::Media::Control::{
            GlobalSystemMediaTransportControlsSessionManager as SessionManager,
            GlobalSystemMediaTransportControlsSessionPlaybackStatus as PlaybackStatus,
        };
        let result: windows::core::Result<(String, String, bool)> = (|| {
            let manager = SessionManager::RequestAsync()?.get()?;
            let session = manager.GetCurrentSession()?;
            let playing =
                session.GetPlaybackInfo()?.PlaybackStatus()? == PlaybackStatus::Playing;
            let props = session.TryGetMediaPropertiesAsync()?.get()?;
            let title = props.Title().map(|s| s.to_string()).unwrap_or_default();
            let artist = props.Artist().map(|s| s.to_string()).unwrap_or_default();
            Ok((title, artist, playing))
        })();
        match result {
            Ok((title, artist, playing)) => (
                (!title.is_empty()).then_some(title),
                (!artist.is_empty()).then_some(artist),
                Some(playing),
            ),
            Err(_) => (None, None, None),
        }
    }
}

#[derive(serde::Serialize, Default)]
struct AmbientSignals {
    window_title: Option<String>,
    battery_percent: Option<u8>,
    battery_charging: Option<bool>,
    media_title: Option<String>,
    media_artist: Option<String>,
    media_playing: Option<bool>,
}

#[tauri::command]
async fn ambient_signals() -> AmbientSignals {
    #[cfg(target_os = "windows")]
    {
        let (battery_percent, battery_charging) = ambient_detect::battery();
        let (media_title, media_artist, media_playing) = ambient_detect::now_playing();
        AmbientSignals {
            window_title: ambient_detect::foreground_title(),
            battery_percent,
            battery_charging,
            media_title,
            media_artist,
            media_playing,
        }
    }
    #[cfg(not(target_os = "windows"))]
    AmbientSignals::default()
}

#[derive(serde::Serialize)]
struct PetCursorContext {
    cursor_x: f64,
    cursor_y: f64,
    win_x: i32,
    win_y: i32,
    win_w: u32,
    win_h: u32,
    scale: f64,
}

/// Polled a few times a second by the pet window's gaze-follow JS — keep this
/// cheap and quiet on the happy path (no logging; errors are returned to the
/// caller, which is expected to just skip that tick). All position/size
/// fields are physical pixels, matching `AppHandle::cursor_position()` and
/// window geometry getters directly (no unit conversion needed on either
/// side of the IPC boundary).
#[tauri::command]
fn pet_cursor_context(app: AppHandle) -> Result<PetCursorContext, String> {
    let window = app
        .get_webview_window(PET_WINDOW_LABEL)
        .ok_or_else(|| "Pet window is not open".to_string())?;
    let cursor = app.cursor_position().map_err(|err| err.to_string())?;
    let position = window.outer_position().map_err(|err| err.to_string())?;
    let size = window.outer_size().map_err(|err| err.to_string())?;
    let scale = window.scale_factor().map_err(|err| err.to_string())?;
    Ok(PetCursorContext {
        cursor_x: cursor.x,
        cursor_y: cursor.y,
        win_x: position.x,
        win_y: position.y,
        win_w: size.width,
        win_h: size.height,
        scale,
    })
}

/// Named layout presets for the pet window, keeping current position.
/// Logical units (matches `inner_size`'s builder-time counterpart above and
/// the DPI-independent sizes the frontend already reasons in).
#[tauri::command]
fn set_pet_window_preset(app: AppHandle, preset: String) -> Result<(), String> {
    let window = app
        .get_webview_window(PET_WINDOW_LABEL)
        .ok_or_else(|| "Pet window is not open".to_string())?;
    let (width, height) = match preset.as_str() {
        "full" => (380.0, 560.0),
        "compact" => (320.0, 380.0),
        other => return Err(format!("Unknown pet window preset: {other}")),
    };
    window
        .set_size(tauri::Size::Logical(tauri::LogicalSize { width, height }))
        .map_err(|err| err.to_string())
}

#[derive(serde::Serialize)]
struct Screenshot {
    data: String, // base64 PNG
    width: u32,
    height: u32,
}

/// One-shot screen capture of the monitor under the cursor (vision feature).
/// Physical pixels; the frontend downscales/JPEG-encodes before upload.
#[tauri::command]
fn capture_screenshot(app: AppHandle) -> Result<Screenshot, String> {
    let monitor = match app.cursor_position() {
        Ok(pos) => xcap::Monitor::from_point(pos.x as i32, pos.y as i32)
            .or_else(|_| primary_monitor())?,
        Err(_) => primary_monitor()?,
    };
    let image = monitor.capture_image().map_err(|err| err.to_string())?;
    let (width, height) = (image.width(), image.height());
    let mut png_bytes: Vec<u8> = Vec::new();
    image
        .write_to(
            &mut std::io::Cursor::new(&mut png_bytes),
            image::ImageFormat::Png,
        )
        .map_err(|err| err.to_string())?;
    Ok(Screenshot {
        data: base64::engine::general_purpose::STANDARD.encode(png_bytes),
        width,
        height,
    })
}

fn primary_monitor() -> Result<xcap::Monitor, String> {
    let mut monitors = xcap::Monitor::all().map_err(|err| err.to_string())?;
    if let Some(index) = monitors
        .iter()
        .position(|m| m.is_primary().unwrap_or(false))
    {
        return Ok(monitors.swap_remove(index));
    }
    // No monitor claims primary (or the query failed) — any monitor beats
    // erroring out of a screenshot request.
    monitors
        .into_iter()
        .next()
        .ok_or_else(|| "No monitor found".to_string())
}

#[tauri::command]
fn delete_gemini_key() -> Result<GeminiKeyStatus, String> {
    match gemini_entry()?.delete_credential() {
        Ok(()) => Ok(GeminiKeyStatus { configured: false }),
        Err(err) => {
            let message = err.to_string();
            if message.to_lowercase().contains("no entry") || message.to_lowercase().contains("not found") {
                Ok(GeminiKeyStatus { configured: false })
            } else {
                Err(message)
            }
        }
    }
}

fn wait_for_health<A: ToSocketAddrs>(addr: A, deadline: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < deadline {
        if let Ok(mut stream) = TcpStream::connect(&addr) {
            let request = b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
            if stream.write_all(request).is_ok() {
                let mut response = String::new();
                if stream.read_to_string(&mut response).is_ok()
                    && response.starts_with("HTTP/1.1 200")
                    && response.contains("\"status\":\"ok\"")
                {
                    return true;
                }
            }
        }
        thread::sleep(Duration::from_millis(250));
    }
    false
}

fn resource_path<R: Runtime>(app: &AppHandle<R>, relative: &str) -> anyhow::Result<PathBuf> {
    Ok(app.path().resource_dir()?.join(relative))
}

/// Forward one line of sidecar output to our stderr, ignoring write errors.
///
/// Must never panic: this runs in the pipe-drain task, and `eprintln!` panics
/// when stderr is a broken pipe (an inherited console/interop pipe whose
/// other end exited). If the drain task dies, the child's pipes fill and the
/// backend's own logging blocks its event loop mid-`flush()` — observed live
/// as an installed app whose /healthz stopped answering entirely.
fn forward_backend_line(bytes: &[u8]) {
    use std::io::Write;

    let mut stderr = std::io::stderr().lock();
    let _ = stderr.write_all(b"[amadeus-backend] ");
    let _ = stderr.write_all(bytes);
    let _ = stderr.write_all(b"\n");
}

// Concrete (Wry) rather than generic over Runtime: the health task below
// spawns the pet window on pet-default startups, and window creation only
// exists for the real runtime.
fn launch_backend(app: &AppHandle, window: WebviewWindow) -> anyhow::Result<()> {
    // Instant-on: a pet-default launch shows NOTHING but a tray icon for the
    // entire sidecar boot otherwise — from the outside indistinguishable
    // from a failed launch. Spawn the pet window with the splash right here,
    // before the port is even picked or the sidecar process spawned, so
    // something appears on screen within a frame or two of the app starting.
    // Window creation must happen on the main/event-loop thread (same
    // constraint `toggle_pet_window` works around) — `launch_backend` itself
    // runs inside `.setup()`, which IS the main thread, so this can call
    // straight through instead of needing the `run_on_main_thread` channel
    // dance the async IPC command needs.
    if pet_default_enabled(app) {
        if let Err(err) = build_pet_splash_window(app) {
            // Splash failed to spawn: fall through exactly as before this
            // feature existed (main window stays visible through the whole
            // boot, pet spawns post-health as it always did) rather than
            // aborting the launch over a cosmetic window.
            eprintln!("[pet] splash window spawn failed: {err}");
        }
    }

    let port = portpicker::pick_unused_port().ok_or_else(|| anyhow::anyhow!("No free local port found"))?;
    let app_data_dir = app.path().app_data_dir()?;
    std::fs::create_dir_all(&app_data_dir)?;

    let config_path = app_data_dir.join("config.yaml");
    let frontend_dir = resource_path(app, "frontend/dist")?;
    let static_dir = resource_path(app, "static")?;
    let resources_dir = app.path().resource_dir()?;

    // Pass only the binary name: the bundler flattens external binaries next
    // to the app executable, so a "binaries/" prefix resolves to a directory
    // that doesn't exist in an installed copy (os error 3 before any window).
    // `mut` is only exercised under the `qwen` feature (which pushes
    // --prefer-qwen below); allow it so the CPU build compiles warning-free.
    #[allow(unused_mut)]
    let mut sidecar_args = vec![
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.to_string(),
        "--config".to_string(),
        config_path.to_string_lossy().to_string(),
        "--frontend-dir".to_string(),
        frontend_dir.to_string_lossy().to_string(),
        "--static-dir".to_string(),
        static_dir.to_string_lossy().to_string(),
        "--resources-dir".to_string(),
        resources_dir.to_string_lossy().to_string(),
        "--app-data-dir".to_string(),
        app_data_dir.to_string_lossy().to_string(),
        "--write-default-config".to_string(),
    ];

    // The GPU installer is built with `--features qwen`; it tells the sidecar
    // to prefer the Qwen3 cloned voice when a CUDA GPU + reference clip are
    // present (the sidecar itself falls back to ONNX otherwise). The default
    // CPU installer omits the flag and always uses the ONNX voice.
    #[cfg(feature = "qwen")]
    sidecar_args.push("--prefer-qwen".to_string());

    // CPU build: the backend is a single --onefile exe shipped via Tauri
    // `externalBin`, launched by name. Full GPU portable: the CUDA sidecar is
    // ~3GB, which NSIS cannot pack as one file (`File: failed creating mmap`),
    // so it ships as a PyInstaller --onedir tree under resources and is
    // launched by absolute path to the inner exe. Slim GPU installer: built
    // with the same `qwen` feature (for --prefer-qwen) but ships the CPU
    // onefile sidecar via `externalBin` and NO onedir tree — so the qwen
    // branch must fall back to the standard sidecar when the onedir exe is
    // absent, or an installed slim app never boots its backend at all (found
    // live cutting v0.15.0: only a stale dev-dir onedir tree made it look
    // like it worked).
    #[cfg(not(feature = "qwen"))]
    let mut command = app
        .shell()
        .sidecar("amadeus-backend")?
        .args(sidecar_args);

    #[cfg(feature = "qwen")]
    let mut command = {
        let backend_exe = resource_path(app, "amadeus-backend/amadeus-backend.exe")?;
        if backend_exe.exists() {
            app.shell()
                .command(backend_exe.to_string_lossy().to_string())
                .args(sidecar_args)
        } else {
            app.shell().sidecar("amadeus-backend")?.args(sidecar_args)
        }
    };

    if let Ok(key) = gemini_entry().and_then(|entry| entry.get_password().map_err(|err| err.to_string())) {
        if !key.is_empty() {
            command = command.env("GEMINI_API_KEY", key);
        }
    }

    let (mut rx, child) = command.spawn()?;
    app.state::<BackendProcess>()
        .child
        .lock()
        .expect("backend process mutex poisoned")
        .replace(child);

    // Watch the sidecar for the rest of the app's life: if it dies after a
    // successful boot (crash, killed externally, OOM), the webview is by then
    // showing the live backend host and would otherwise just sit on dead
    // fetches with no explanation. Stdout/Stderr must also be drained here —
    // tauri-plugin-shell forwards pipe output through this same channel, and
    // an unconsumed receiver risks backpressure on the child's pipes.
    //
    // The drain must NEVER die on a write failure: eprintln! PANICS when
    // stderr is a broken pipe (e.g. the app inherited a console/interop pipe
    // whose other end has exited). Seen live on v0.15.0: the panic killed
    // this task, the child's pipes filled, and the backend's logging then
    // blocked its event loop on flush() — the whole app wedged (healthz 60s+
    // timeouts). forward_backend_line() ignores write errors instead.
    let watcher_window = window.clone();
    async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    forward_backend_line(&bytes);
                }
                CommandEvent::Stderr(bytes) => {
                    forward_backend_line(&bytes);
                }
                CommandEvent::Error(message) => {
                    forward_backend_line(format!("event error: {message}").as_bytes());
                }
                CommandEvent::Terminated(payload) => {
                    if SHUTTING_DOWN.load(Ordering::SeqCst) {
                        // Deliberate: tray "Quit" or main-window close already
                        // killed the child. Nothing to report.
                        break;
                    }
                    forward_backend_line(
                        format!("exited unexpectedly: {payload:?}").as_bytes(),
                    );
                    // The webview is currently showing the live backend host,
                    // so it must navigate back to the bundled UI before the
                    // failed-state flag means anything (DesktopBackendWait
                    // only renders there). `navigate` returns before the load
                    // completes and a single eval fired right after it can
                    // land mid-navigation and get lost, so re-assert the flag
                    // on a short retry loop instead of trusting one shot. Runs
                    // via spawn_blocking since it parks the thread between
                    // retries — `navigate`/`eval` themselves just dispatch to
                    // the UI thread and are fine to call off it.
                    let _ = watcher_window.navigate(bundled_app_url());
                    let retry_window = watcher_window.clone();
                    let _ = async_runtime::spawn_blocking(move || {
                        for _ in 0..40 {
                            thread::sleep(Duration::from_millis(100));
                            let _ = retry_window.eval(
                                "document.documentElement.dataset.amadeusBackendFailed = '1'",
                            );
                        }
                    })
                    .await;
                    break;
                }
                _ => {}
            }
        }
    });

    let app_handle_for_health = app.clone();
    async_runtime::spawn(async move {
        let addr = ("127.0.0.1", port);
        // The GPU (qwen) build loads the Qwen3-TTS model and captures CUDA
        // graphs before uvicorn starts serving — well over the CPU build's
        // 45s. Give it 5 minutes before declaring the backend dead.
        let deadline = if cfg!(feature = "qwen") { 300 } else { 45 };
        if cfg!(feature = "qwen") {
            // Flag the SPA's pre-boot screen (DesktopBackendWait) to show the
            // long-warmup copy instead of replacing the page with raw HTML —
            // the eval used to nuke document.body, so the styled screen never
            // showed on the GPU build, whose wait is the longest.
            let _ = window
                .eval("document.documentElement.dataset.amadeusGpuWarmup = '1'");
        }
        let ready = async_runtime::spawn_blocking(move || wait_for_health(addr, Duration::from_secs(deadline)))
            .await
            .unwrap_or(false);
        if ready {
            app_handle_for_health
                .state::<BackendState>()
                .port
                .lock()
                .expect("backend port mutex poisoned")
                .replace(port);
            // Pet-as-default: hide the main window BEFORE navigating it to
            // the live URL. Navigation boots the SPA, whose LoadingScreen
            // auto-connects within ~1s under Tauri and then fires the
            // greeting/brief requests — the server pops those exactly-once,
            // so a still-visible-but-doomed main window can win the pop and
            // voice the greeting into a window that's about to be hidden,
            // leaving the pet a clean null (perceived as "her voice didn't
            // work"). Hidden-from-birth means document.hidden is true from
            // the SPA's first frame, so its presence guards defer to the pet.
            // Un-hidden below if the pet window fails to spawn.
            let pet_default = pet_default_enabled(&app_handle_for_health);
            if pet_default {
                let _ = window.hide();
            }
            if let Ok(url) = Url::parse(&format!("http://127.0.0.1:{port}/")) {
                let _ = window.navigate(url);
            }
            // Pet-as-default: once the backend is up, move the pet overlay
            // off the splash and onto the live URL, and tuck the main chat
            // window away. Main keeps its webview (parked on the
            // click-to-enter screen, so it runs no presence polls and plays
            // no audio while hidden) — the tray's "Open chat window" and the
            // pet's chat button bring it back.
            //
            // The splash window was already created back at the top of
            // `launch_backend` (instant-on), so this is a `navigate`, not a
            // `build` — build_pet_window here would just error with "window
            // with that label already exists". The one exception is if the
            // splash itself failed to spawn (logged there, non-fatal): in
            // that case there is no pet window yet, so fall back to the
            // pre-splash behavior of building it fresh now.
            if pet_default {
                let handle = app_handle_for_health.clone();
                let main_window = window.clone();
                let _ = app_handle_for_health.run_on_main_thread(move || {
                    let opened = if let Some(pet_window) = handle.get_webview_window(PET_WINDOW_LABEL) {
                        pet_window.navigate(pet_live_url(port)).map(|()| pet_window)
                            .map_err(|err| err.to_string())
                    } else {
                        build_pet_window(&handle, port).map_err(|err| err.to_string())
                    };
                    match opened {
                        Ok(_) => {
                            // Main was already hidden pre-navigate; repeat is
                            // a no-op but keeps this branch self-sufficient.
                            let _ = main_window.hide();
                        }
                        Err(err) => {
                            // Pet failed to spawn/navigate: bring the main
                            // window back (it was hidden pre-navigate) — a
                            // hidden-everything boot would look like a
                            // silent crash.
                            eprintln!("[pet] default-window spawn failed: {err}");
                            let _ = main_window.show();
                        }
                    }
                });
            }
        } else {
            let _ = window
                .eval("document.documentElement.dataset.amadeusBackendFailed = '1'");
            // Mirror the failure onto the pet splash too, if one is up
            // (pet-default launch): otherwise it would sit on "waking up..."
            // forever with no indication anything ever went wrong, since the
            // pet window has no other UI wired to the backend-failed flag the
            // main window's DesktopBackendWait screen reads.
            if pet_default_enabled(&app_handle_for_health) {
                if let Some(pet_window) = app_handle_for_health.get_webview_window(PET_WINDOW_LABEL) {
                    let _ = pet_window.navigate(data_html_url(SPLASH_ERROR_HTML));
                }
            }
        }
    });

    Ok(())
}

fn build_tray(app: &tauri::App) -> tauri::Result<()> {
    use tauri::menu::{MenuBuilder, MenuItemBuilder};
    use tauri::tray::TrayIconBuilder;

    let toggle_pet = MenuItemBuilder::with_id("toggle_pet", "Toggle pet mode").build(app)?;
    let no_click_through =
        MenuItemBuilder::with_id("disable_click_through", "Disable click-through").build(app)?;
    let show_main = MenuItemBuilder::with_id("show_main", "Open chat window").build(app)?;
    let quit = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
    let menu = MenuBuilder::new(app)
        .items(&[&toggle_pet, &no_click_through, &show_main, &quit])
        .build()?;

    let mut tray = TrayIconBuilder::new()
        .menu(&menu)
        .tooltip("Amadeus Kurisu")
        .on_menu_event(|app, event| match event.id().as_ref() {
            "toggle_pet" => {
                if let Err(err) = toggle_pet_window_impl(app) {
                    eprintln!("[tray] pet toggle failed: {err}");
                }
            }
            "disable_click_through" => {
                // The guaranteed escape hatch: a click-through pet window
                // can't be clicked, so the tray (always reachable, no pet
                // interaction needed) has to be able to turn it back off.
                // CmdOrCtrl+Shift+K (`register_click_through_shortcut`) is a
                // faster alternate route to the same fix, but registration
                // can fail (hotkey held by another app), so this menu item
                // stays the one path that's never allowed to stop working.
                let _ = set_pet_click_through(app.clone(), false);
            }
            "show_main" => {
                show_main_window_impl(app);
            }
            "quit" => {
                if let Some(state) = app.try_state::<BackendProcess>() {
                    kill_backend_child(Arc::clone(&state.child));
                }
                app.exit(0);
            }
            _ => {}
        });
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    tray.build(app)?;
    Ok(())
}

/// CmdOrCtrl+Shift+K: recover from an accidentally-enabled click-through pet
/// window without going near the tray. Non-fatal on registration failure
/// (another app can hold the same global hotkey) — the tray's "Disable
/// click-through" item remains the guaranteed recovery path either way, this
/// is purely an extra convenience.
fn register_click_through_shortcut(app: &tauri::App) {
    use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut};

    let shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyK);
    let handle = app.handle().clone();
    let result = app.global_shortcut().on_shortcut(shortcut, move |_app, _shortcut, event| {
        // Fires on both press and release; only act on press or the toggle
        // would flip twice per key-tap and net out to a no-op.
        if event.state() == tauri_plugin_global_shortcut::ShortcutState::Pressed {
            toggle_pet_click_through_impl(&handle);
        }
    });
    if let Err(err) = result {
        eprintln!("[shortcut] click-through hotkey registration failed: {err}");
    }
}

/// How often the lock-state poller checks `OpenInputDesktop`. Bounded
/// detection latency (not instant) is an accepted tradeoff for avoiding
/// window-message-loop surgery — see `lock_detect::is_locked`'s doc comment.
const LOCK_POLL_INTERVAL_SECS: u64 = 12;

/// Emitted to both the pet and main windows on a locked→unlocked transition.
const SESSION_UNLOCKED_EVENT: &str = "session-unlocked";

#[derive(serde::Serialize, Clone)]
struct SessionUnlockedPayload {
    locked_seconds: u64,
}

/// Spawns the background poller that watches for a locked→unlocked
/// transition and emits `session-unlocked` to whichever of the pet/main
/// windows currently exist. No-op on non-Windows targets (the thread never
/// starts — there is nothing to detect and nothing to degrade).
///
/// Failure handling: `lock_detect::poll_locked` has no fallible path left to
/// surface here (see its doc comment) — its own rare `CloseDesktop` failure
/// case already logs itself once, internally. This loop itself stays quiet
/// on every tick, matching the existing `fullscreen_detect` poll's
/// no-logging-on-the-happy-path convention.
#[allow(unused_variables)]
fn spawn_lock_watcher(app: &AppHandle) {
    #[cfg(target_os = "windows")]
    {
        let handle = app.clone();
        thread::spawn(move || {
            let mut was_locked = lock_detect::poll_locked();
            let mut locked_since = if was_locked {
                Some(Instant::now())
            } else {
                None
            };
            loop {
                thread::sleep(Duration::from_secs(LOCK_POLL_INTERVAL_SECS));
                let now_locked = lock_detect::poll_locked();
                if was_locked && !now_locked {
                    // locked -> unlocked transition.
                    let locked_seconds = locked_since
                        .map(|since| since.elapsed().as_secs())
                        .unwrap_or(0);
                    let payload = SessionUnlockedPayload { locked_seconds };
                    for label in ["pet", "main"] {
                        if handle.get_webview_window(label).is_some() {
                            let _ = handle.emit_to(label, SESSION_UNLOCKED_EVENT, payload.clone());
                        }
                    }
                }
                if now_locked && !was_locked {
                    locked_since = Some(Instant::now());
                } else if !now_locked {
                    locked_since = None;
                }
                was_locked = now_locked;
            }
        });
    }
}

fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        // MacosLauncher only matters on macOS (see the plugin's own
        // `#[cfg(target_os = "macos")]` on Builder::macos_launcher); the
        // default LaunchAgent variant is fine since this app has no reason to
        // want the AppleScript variant. No launch args: pet-default is
        // decided by the marker file `pet_default_flag_path` reads at
        // startup, not by anything passed on the command line, so `None`
        // here is correct rather than a placeholder.
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        // The pet overlay stays out of window-state: restoring stale
        // geometry for an always-on-top frameless overlay is wrong — it
        // should always spawn at its default spot, and a saved off-screen
        // position would strand an undecorated window with no way to grab it.
        // (Position/size ARE still persisted for the pet, just via the
        // bespoke, monitor-validated `PetRect` path above instead of this
        // plugin's blind restore — see `pet_rect_is_visible`.)
        .plugin(
            tauri_plugin_window_state::Builder::default()
                .with_denylist(&["pet"])
                .build(),
        )
        .manage(BackendProcess {
            child: Arc::new(Mutex::new(None)),
        })
        .manage(BackendState {
            port: Arc::new(Mutex::new(None)),
        })
        .manage(PetClickThroughState {
            ignoring: AtomicBool::new(false),
        })
        .manage(PetRectDebounce {
            latest: Mutex::new(None),
            generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
        .manage(PerchState(Mutex::new(None)))
        .manage(PetRoamAnim {
            generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
        .invoke_handler(tauri::generate_handler![
            get_gemini_key_status,
            set_gemini_key,
            delete_gemini_key,
            toggle_pet_window,
            set_pet_click_through,
            set_pet_default,
            show_main_window,
            capture_screenshot,
            ambient_signals,
            pet_cursor_context,
            set_pet_window_preset,
            is_foreground_fullscreen,
            pet_perch_state,
            pet_roam_context,
            animate_pet_window_to,
            cancel_pet_window_animation,
            set_autostart,
            get_autostart,
            check_app_update,
            install_app_update
        ])
        .setup(|app| {
            let handle = app.handle().clone();
            let window = app.get_webview_window("main").expect("main window is missing");
            launch_backend(&handle, window)?;
            build_tray(app)?;
            register_click_through_shortcut(app);
            spawn_lock_watcher(&handle);
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                // Only the MAIN window owns the backend: closing the pet
                // overlay must not kill the sidecar under the main window.
                if window.label() != "main" {
                    return;
                }
                // Pet alive: she keeps living on the desktop, so closing
                // the chat window just hides it (menubar-app semantics —
                // the tray and the pet's chat button bring it back, and
                // tray "Quit" remains the real exit). No pet: closing the
                // main window exits the app exactly as before.
                if window
                    .app_handle()
                    .get_webview_window(PET_WINDOW_LABEL)
                    .is_some()
                {
                    api.prevent_close();
                    let _ = window.hide();
                    return;
                }
                if let Some(state) = window.try_state::<BackendProcess>() {
                    kill_backend_child(Arc::clone(&state.child));
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() {
    run();
}
