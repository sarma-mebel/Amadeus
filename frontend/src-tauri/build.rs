fn main() {
    // Application commands invoked from JS must be exported into the ACL
    // before a capability can grant them. Without this, they are implicitly
    // allowed on the bundled origin only — and the packaged app runs on the
    // backend-served (remote) origin, where every invoke fails with
    // "not allowed by ACL" (the dead pet button / vision capture bug).
    // Keychain commands stay unexported: nothing invokes them from JS.
    let attributes = tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "toggle_pet_window",
            "set_pet_click_through",
            "set_pet_default",
            "show_main_window",
            "capture_screenshot",
            "ambient_signals",
            "pet_cursor_context",
            "set_pet_window_preset",
            "is_foreground_fullscreen",
            "pet_perch_state",
            "pet_roam_context",
            "animate_pet_window_to",
            "cancel_pet_window_animation",
            "set_autostart",
            "get_autostart",
            "check_app_update",
            "install_app_update",
        ]),
    );
    tauri_build::try_build(attributes).expect("failed to run tauri-build");
}
