#!/usr/bin/env node
// Run a command with TAURI_SIGNING_PRIVATE_KEY_PASSWORD present but EMPTY.
//
// Windows shells cannot do this themselves: PowerShell's `$env:VAR = ""`,
// cmd's `set VAR=`, and .NET/Win32 SetEnvironmentVariable all DELETE the
// variable when handed an empty value, and WSLENV drops empty vars too.
// With the variable missing, `tauri build` prompts interactively for the
// key password after makensis ("Decrypting updater signing key") and hangs
// every scripted or headless build (observed cutting v0.14.0 and v0.15.0).
//
// Node is the escape hatch: it constructs the child's environment block
// itself and preserves "VAR=" entries, and every descendant (powershell,
// cmd, npm, the tauri CLI) inherits that block verbatim. Verified live:
// the empty value survives node -> powershell -> cmd -> node.
//
// Usage (the signing key path must still be set by the wrapped command or
// the caller's environment):
//   node scripts/win_empty_signing_password.cjs powershell.exe -ExecutionPolicy Bypass -File scripts/package_desktop_windows.ps1
const { spawnSync } = require("child_process");

const [cmd, ...args] = process.argv.slice(2);
if (!cmd) {
  console.error(
    "usage: node scripts/win_empty_signing_password.cjs <command> [args...]"
  );
  process.exit(2);
}
const spawnOpts = {
  env: { ...process.env, TAURI_SIGNING_PRIVATE_KEY_PASSWORD: "" },
  stdio: "inherit",
};
let result = spawnSync(cmd, args, spawnOpts);
// Wrapping `npm ...` (a .cmd shim) fails without shell: true on modern
// node — EINVAL when the .cmd is named explicitly (CVE-2024-27980
// hardening) or ENOENT because extension-less shim resolution needs the
// shell (observed on node 24). Retry through the shell in those cases;
// .exe commands (powershell.exe etc.) never hit this and keep the
// exact-args path with no quoting hazards. A genuinely missing command
// still fails on the retry, just with the shell's error instead.
if (
  result.error &&
  (result.error.code === "EINVAL" || result.error.code === "ENOENT") &&
  process.platform === "win32"
) {
  const quoted = args.map((a) => (/\s/.test(a) ? `"${a}"` : a));
  result = spawnSync(cmd, quoted, { ...spawnOpts, shell: true });
}
if (result.error) {
  console.error(String(result.error));
  process.exit(1);
}
process.exit(result.status === null ? 1 : result.status);
