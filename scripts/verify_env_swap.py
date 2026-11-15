#!/usr/bin/env python3
"""Live-browser verification for the VRMViewer environment-swap refactor.

The refactor's whole point is a runtime behavior no unit test can observe:
changing the environment must swap ONLY the environment. It must NOT tear down
the WebGL scene, re-fetch the VRM, or reset the AudioContext/lip-sync graph. An
outfit (VRM) swap, by contrast, MUST still fully rebuild.

This drives a real headless Chromium (software GL) against a running backend
that serves the built SPA, triggers both swaps via the app's own window events,
and asserts:

  1. env swap    -> zero new Kurisu.vrm network fetches
  2. env swap    -> window.kurisuLipSync object identity is preserved (a stamp
                    we put on it survives), i.e. the scene effect did not re-run
  3. env swap    -> the VRM loading spinner (.vrm-overlay[role=status]) never
                    reappears
  4. outfit swap -> the VRM IS re-fetched and the stamp is gone (control case:
                    proves the rebuild path still works and our sentinel is real)

Prerequisites (see docs/conversation-eval.md's sibling and the env memory):
  # 1. Build a fresh dist that INCLUDES the refactor (Windows node in WSL):
  #    cd frontend && npm run build:desktop
  # 2. Run a backend that serves that dist (dev-preview avoids the GPU voice):
  #    AMADEUS_CONFIG=config.dev-preview.yaml uvicorn amadeus.web.app:app \
  #      --host 127.0.0.1 --port 8041
  # 3. Point this script at it:
  #    python scripts/verify_env_swap.py --base-url http://127.0.0.1:8041

Uses the Python Playwright (pip install playwright && playwright install
chromium). Exits non-zero on any failed assertion so it can gate a release.
"""

from __future__ import annotations

import argparse

from playwright.sync_api import sync_playwright

VRM_FILE = "Kurisu.vrm"

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))


def dismiss_overlays(page) -> None:
    # 1) click-to-enter loading screen (waits on the real backend /healthz).
    loading = page.locator("div.loading-screen[aria-label='Establish connection']")
    try:
        loading.wait_for(state="visible", timeout=25000)
        loading.click()
        loading.wait_for(state="detached", timeout=25000)
    except Exception:
        pass  # some configs skip past it; not fatal
    # 2) onboarding settings modal (auto-opens when no API key is configured).
    skip = page.locator("button.settings-skip-button").first
    try:
        skip.wait_for(state="visible", timeout=4000)
        skip.click()
    except Exception:
        pass  # no onboarding overlay (key already set)


def wait_for_vrm_ready(page) -> None:
    # Ready = the lip-sync bridge is published and the spinner is gone.
    page.wait_for_function("() => !!window.kurisuLipSync", timeout=60000)
    try:
        page.locator('.vrm-overlay[role="status"]').wait_for(
            state="detached", timeout=60000
        )
    except Exception:
        pass


def run(base_url: str) -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=[
                "--use-gl=swiftshader",
                "--ignore-gpu-blocklist",
                "--enable-webgl",
            ]
        )
        page = browser.new_page()

        vrm_fetches = {"n": 0}
        page.on(
            "request",
            lambda req: vrm_fetches.__setitem__(
                "n", vrm_fetches["n"] + (1 if VRM_FILE in req.url else 0)
            ),
        )

        try:
            page.goto(base_url, wait_until="domcontentloaded")
            dismiss_overlays(page)
            wait_for_vrm_ready(page)

            after_boot = vrm_fetches["n"]
            check("VRM loaded on boot", after_boot >= 1, f"{after_boot} fetch(es)")

            # Stamp the live lip-sync bridge. It's a fresh object each time the
            # scene effect runs (null during teardown), so a surviving stamp
            # proves the scene was NOT rebuilt.
            page.evaluate("""() => {
                    window.__lipSyncStamp = 'boot-' + Math.random().toString(36).slice(2);
                    window.kurisuLipSync.__stamp = window.__lipSyncStamp;
                }""")

            # --- Environment swap: must NOT rebuild ------------------------
            print("\n[env swap: time_machine -> lab]")
            vrm_before_env = vrm_fetches["n"]
            # Poll for a spinner reappearance across the swap window.
            spinner_watch = page.evaluate_handle("""() => new Promise((resolve) => {
                    let seen = false;
                    const started = Date.now();
                    const iv = setInterval(() => {
                        if (document.querySelector('.vrm-overlay[role="status"]')) seen = true;
                        if (Date.now() - started > 2500) { clearInterval(iv); resolve(seen); }
                    }, 50);
                })""")
            # VRMViewer's listener re-reads localStorage (the SettingsPanel
            # contract: pref persisted BEFORE the event fires) and ignores the
            # event detail — dispatching without writing the key swaps
            # nothing, which made every check below pass trivially.
            page.evaluate("""() => {
                    window.localStorage.setItem('kurisu_environment_v1', 'lab');
                    window.dispatchEvent(
                        new CustomEvent('kurisu:environment-change', { detail: 'lab' }));
                }""")
            page.wait_for_timeout(2500)
            spinner_flashed = spinner_watch.json_value()

            vrm_after_env = vrm_fetches["n"]
            stamp_survived = page.evaluate("""() => !!window.kurisuLipSync
                    && window.kurisuLipSync.__stamp === window.__lipSyncStamp""")

            check(
                "env swap: no VRM re-fetch",
                vrm_after_env == vrm_before_env,
                f"{vrm_before_env} -> {vrm_after_env}",
            )
            check(
                "env swap: lip-sync graph preserved (no scene rebuild)",
                stamp_survived is True,
            )
            check(
                "env swap: loading spinner did not reappear",
                spinner_flashed is False,
            )

            # --- Outfit swap (control): MUST rebuild -----------------------
            print("\n[outfit swap control: expect a rebuild]")
            models = page.evaluate("""async () => {
                    try {
                        const r = await fetch('/static/models/models.json', { cache: 'no-store' });
                        if (!r.ok) return [];
                        const d = await r.json();
                        return Array.isArray(d && d.models) ? d.models.map(m => m.id) : [];
                    } catch { return []; }
                }""")
            other = next((m for m in models if m and m != "kurisu_default"), None)
            if not other:
                print(
                    "  (skipped: only the default outfit is registered - "
                    "nothing to swap to)"
                )
            else:
                vrm_before_model = vrm_fetches["n"]
                page.evaluate(
                    "(id) => window.dispatchEvent("
                    "new CustomEvent('kurisu:model-change', { detail: id }))",
                    other,
                )
                wait_for_vrm_ready(page)
                page.wait_for_timeout(500)
                vrm_after_model = vrm_fetches["n"]
                stamp_gone = page.evaluate("""() => !window.kurisuLipSync
                        || window.kurisuLipSync.__stamp !== window.__lipSyncStamp""")
                check(
                    "outfit swap: VRM re-fetched (rebuild happened)",
                    vrm_after_model > vrm_before_model,
                    f"{vrm_before_model} -> {vrm_after_model}",
                )
                check(
                    "outfit swap: lip-sync graph rebuilt (stamp cleared)",
                    stamp_gone is True,
                )
        except Exception as exc:  # pragma: no cover - live-run failure surface
            check("script ran without error", False, str(exc))
        finally:
            browser.close()

    failed = [r for r in _results if not r[1]]
    passed = len(_results) - len(failed)
    print(f"\n{'PASSED' if not failed else 'FAILED'}: {passed}/{len(_results)} checks")
    if failed:
        print("Failed checks:")
        for name, _, detail in failed:
            print(f"  - {name}" + (f" ({detail})" if detail else ""))
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8041",
        help="origin of a backend serving the freshly built SPA",
    )
    args = parser.parse_args()
    return run(args.base_url)


if __name__ == "__main__":
    raise SystemExit(main())
