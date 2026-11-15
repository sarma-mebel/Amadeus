# Setting up a self-hosted Windows runner

`.github/workflows/desktop-checks.yml` and the Windows jobs in
`.github/workflows/release.yml` need a Windows machine with the Rust and
Node toolchains to run `cargo check` on the Tauri shell, the frontend tests
on a native filesystem, and the full signed release builds. GitHub-hosted `windows-latest` runners could
technically do this too, but this project is a private repo on the free
plan and Windows minutes bill at 2x, so a self-hosted runner on your own
Windows machine (the same one you already use to build the desktop app) is
the practical choice. This takes about 20 minutes the first time.

## Register the runner

1. On GitHub, go to the repo's **Settings > Actions > Runners**, then click
   **New self-hosted runner**.
2. Pick **Windows** and the architecture that matches your machine (x64 for
   almost everyone). GitHub shows you a page of commands specific to your
   repo, with a short-lived registration token baked in. The steps below
   summarize that flow; follow the exact commands GitHub shows you, since
   the token and download URL are per-repo and expire quickly.
3. On the Windows machine, in PowerShell, create a folder for the runner
   and download it (GitHub's page gives you the current version and a
   matching command; it looks like this):

   ```powershell
   mkdir actions-runner; cd actions-runner
   Invoke-WebRequest -Uri <url-from-github> -OutFile actions-runner.zip
   Expand-Archive -Path actions-runner.zip -DestinationPath .
   ```

4. Configure it with the token GitHub gave you:

   ```powershell
   .\config.cmd --url https://github.com/<owner>/<repo> --token <token>
   ```

   When it asks for runner group, name, and labels, accept the defaults for
   name unless you want something specific, but make sure the labels
   include `self-hosted` and `windows` (GitHub adds these automatically for
   a Windows runner; `desktop-checks.yml` targets
   `runs-on: [self-hosted, windows]`, so the labels have to match).

5. Run it once interactively to confirm it connects:

   ```powershell
   .\run.cmd
   ```

   You should see it report "Listening for Jobs" and, if you check the
   Runners page on GitHub, the runner should show as Idle.

## Keep it running across logins

Running `run.cmd` in a terminal window only works while that window stays
open. Two ways to make it durable, and the obvious one is a trap:

**Don't use the Windows service install.** The `svc install` script that
GitHub's docs mention is Linux/macOS only; the Windows runner does not ship
it (validated 2026-08-05: `C:\actions-runner` contains only `config.cmd` and
`run.cmd`). The Windows equivalent, re-configuring with
`config.cmd --runasservice`, runs the service as NETWORK SERVICE by
default — an account that does not have your per-user toolchains (rustup's
`%USERPROFILE%\.cargo\bin`, a per-user Node install) on its PATH, so the
`cargo check` and npm steps fail cryptically. Running the service as
yourself instead requires typing your Windows password into the config
prompt. Skip all of that.

**Use a logon scheduled task instead.** It runs `run.cmd` as your own user
(full PATH, no password prompt) and restarts it every time you log in,
which on a single-developer desktop is the same lifetime as "always":

```powershell
schtasks /Create /F /SC ONLOGON /TN "GitHubActionsRunner-Amadeus" /TR "C:\actions-runner\run.cmd"
schtasks /Run /TN "GitHubActionsRunner-Amadeus"
```

The second command starts it immediately without waiting for the next
logon. Check it's alive on the repo's Runners page (Idle = connected), or
with `gh api repos/<owner>/<repo>/actions/runners`.

## What the machine needs installed

`desktop-checks.yml` needs only the toolchains: Rust (`rustup`), Node 20+,
and Python 3.11 (reachable as `py -3.11`).

The `release.yml` Windows jobs additionally expect, on the machine itself
(none of these are installed by the workflow):

- **GitHub CLI** (`gh`, authenticated as you) — used to download the
  `assets-v1` runtime assets and upload release assets. The stock MSI
  install location (`C:\Program Files\GitHub CLI`) is added to the job PATH
  by the workflow when `gh` isn't already resolvable, so a plain
  `winget install GitHub.cli` is enough.
- **7-Zip** — the full-GPU job splits the portable build into <2GiB
  volumes. `7z` on PATH or the stock `C:\Program Files\7-Zip\7z.exe`
  install both work.
- **SoX** — the full-GPU sidecar bundles `sox.exe` + DLLs. Resolved from
  `SOX_DIR` if set, else choco's `sox.portable` lib dir, else PATH.
- **The updater signing key** at
  `local-assets\updater-keys\amadeus_updater.key` inside the repo checkout
  you build from (gitignored). The workflow prefers the
  `TAURI_SIGNING_PRIVATE_KEY` repository secret when it's set; the local
  file is the fallback that makes a single-machine setup work with zero
  secrets configured.

One 5.1-specific trap for anyone editing the workflows: keep `run:` blocks
ASCII-only. The runner writes step scripts UTF-8 without a BOM and Windows
PowerShell 5.1 reads them as ANSI, so a literal em dash (or any smart
punctuation) mangles into cp1252 bytes that can break string parsing with
errors pointing at an unrelated line.

## Security notes

This runner executes arbitrary workflow code from this repo on your actual
Windows machine, with your actual toolchains and file access. That's fine
for a private repo where you're the only one pushing, because the only code
it ever runs is code you wrote and pushed yourself. It would not be fine on
a public repo, or any repo that accepts external contributions:

- **Never enable this runner for fork pull requests.** A `pull_request`
  trigger from a fork can run attacker-controlled workflow YAML with your
  runner's permissions, on your machine. `desktop-checks.yml` is
  deliberately not triggered by `pull_request` at all (only
  `workflow_dispatch` and a schedule), which sidesteps this, but keep that
  in mind if you ever add push/PR triggers to a workflow that targets this
  runner.
- Keep the repo private for as long as this runner is registered against
  it, unless you swap to GitHub-hosted runners first.
- If you ever suspect the token or runner was compromised, remove it from
  **Settings > Actions > Runners** immediately; that revokes its access.

## Triggering desktop-checks manually

The workflow runs on a weekly schedule automatically, but you can also run
it on demand:

1. Go to the repo's **Actions** tab.
2. Select **Desktop checks (self-hosted Windows)** from the workflow list
   on the left.
3. Click **Run workflow**, pick the branch, and click the green **Run
   workflow** button.

You can also trigger it from the command line with the GitHub CLI:

```bash
gh workflow run desktop-checks.yml
```
