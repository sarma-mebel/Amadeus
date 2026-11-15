# Shakedown checklist (2026-07-24 branch)

Eight passes of behavior landed on feat/realism-pass-2026-07-23 with WSL-level
verification only. This is the live-with-it validation list, in order. Check
Settings > Diagnostics afterward: every counter at zero is the healthy state.

## Boot chain
- [ ] Double-click the packaged exe: the pet splash ("waking up...") appears
      immediately, no white flash, becomes her when the backend is healthy.
- [ ] Greeting plays. If the voice is still warming, the one-shot "voice waking
      up" hint shows instead of silent-looking-broken.
- [ ] Enable Settings > Start with Windows. Reboot. She's just there.
- [ ] Kill the backend process while the pet is up: splash error page, not a
      stranded "waking up...".

## Voice and expressiveness (ear-vet)
- [ ] Trigger laughter/sighs in conversation: nonverbal clips splice into her
      speech at the right moments.
- [ ] Listen to every wav in static/nonverbals and static/fillers. Delete any
      that sound wrong; re-run the generate scripts with --force to fill gaps.
- [ ] Slow reply: thinking-filler plays after ~600ms, preempted when real audio
      starts.
- [ ] Hands-free while YOU talk: occasional backchannel ("un", "ee") and nods.

## Physicality
- [ ] Drag her around: one protest line per drag gesture (first live test —
      these 422'd silently before this branch).
- [ ] Enable click-through: one parting jab. Ctrl+Shift+K restores.
- [ ] Perch her on the taskbar and both side edges, on BOTH monitors. Restart:
      she wakes up where you left her.
- [ ] Quick-tap her head (indignant poke) vs press-and-hold (pat). The pat
      should read warmer given the accumulated relationship arc.

## Context respect
- [ ] Launch a real fullscreen game: she hides and goes quiet. Exit: she
      returns. Borderless-windowed counts as a finding either way.
- [ ] DND 30 minutes: zero unprompted lines; a reminder due during DND
      announces after it ends (never lost).
- [ ] Lock Windows over a meal. Unlock: perk-up, words only if the gap earned
      them (server should_greet decides).

## Night (do this one on her birthday if the timing works)
- [ ] Idle past 23:00: drowsy -> visibly dozing (eyes closed, head droop).
- [ ] Wake her by mic press: groggy half-denial line.
- [ ] Tomorrow's first greeting mentions her birthday once, offhandedly.

## Reading together
- [ ] 20 minutes of actual manga: page detection fires once per settled page
      (knobs if not: 0.10 diff threshold, 2.5s probe, 5s floor in
      readTogether.ts).
- [ ] Comment frequency feels companionable, not chatty (90s speak cooldown,
      server-side read_along config).
- [ ] Ask about the current panel mid-read: she sees it.
- [ ] Stop after 5+ pages: sign-off line. Nothing about the session appears in
      her memory afterward.

## Numbers afterward
- [ ] Settings > Diagnostics: turn.first_audio_ms average (target feel:
      under ~1500ms) — note it for the latency-cut pass.
- [ ] All client and server counters zero. Any non-zero counter is a silent
      failure made loud — note which.

## Ops while you're at it
- [ ] Self-hosted runner install (docs/self-hosted-runner.md, ~20 min).
- [ ] GitHub Actions billing/spending cap check, then push the branch.
