# Pet release checklist

The pet window's platform layer (roaming, tuck-away, monitor clamping,
ACL-gated app commands, WebView2 quirks) is the fiddliest code in the repo.
Its failure mode is "works until a Windows/WebView2 update," and none of it
is validatable from WSL or CI. Run this on the packaged Windows build once
per release, in order. End at Diagnostics: all counters at zero is the
healthy state, any non-zero counter is a finding.

## Launch
- [ ] Fresh pet-default install: the inline splash appears immediately, then
      navigates to the live app once the backend passes `/healthz`.
- [ ] The main window stays hidden the whole time; only the pet window is
      visible.

## Voice
- [ ] Greeting plays through the pet's own lip sync, with a caption shown.
- [ ] Speak a turn: reply audio plays and lip sync tracks it.

## Interaction
- [ ] Click/poke her: she reacts.
- [ ] Drag the window: it follows the cursor and settles where you drop it.
- [ ] Type a message in the pet chat: it sends and gets a reply.

## Roam
- [ ] Enable roaming (`kurisu_pet_roam` in localStorage) and wait for the
      first stroll. The first tick can take about 90 seconds, so don't
      assume it's broken before then.
- [ ] Interact with her mid-stroll: she yields.
- [ ] Drag her mid-stroll: the stroll aborts instead of fighting the drag.
- [ ] On a multi-monitor setup, let a stroll cross from one screen to
      another: it doesn't bounce back at the boundary.

## Tuck
- [ ] Slide her off a screen edge: she tucks and goes quiet (no unprompted
      lines, no pops burned) while tucked.
- [ ] Untuck: she restores to her prior position.
- [ ] Restart the app while tucked: she comes back still tucked.
- [ ] Untuck before moving on (`pet_roam_context` errors by design while
      tucked, which would false-fail the ACL canary below).

## Tray
- [ ] Toggle the pet from the tray menu. This is the only recovery from
      click-through, so confirm it actually works, not just that the menu
      item exists.
- [ ] Close the MAIN window while the pet is up: the backend keeps running
      and the pet keeps working.
- [ ] Tray Quit: this is the real exit, confirm it kills the backend
      sidecar process too (check Task Manager), not just the windows.

## ACL canary
Exercise at least one JS-invoked app command from each of these families and
confirm none fail with "not allowed by ACL" (the classic packaged-only
failure that never reproduces in dev):
- [ ] Roam context (`pet_roam_context`).
- [ ] Window animate (`animate_pet_window_to` or a tuck/untuck move).
- [ ] Screenshot capture (`capture_screenshot`, e.g. via vision).
- [ ] Ambient signals (foreground title / battery / now-playing).

## Diagnostics
- [ ] Open Settings > Diagnostics. Confirm every client and server counter
      reads zero. Note any non-zero counter as a finding, it means one of
      the above degraded silently instead of loudly.
