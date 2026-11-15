# Companion ethics

Amadeus is a companion app, and companion apps sit next to some of the
ugliest dark patterns in software: guilt-tripping, manufactured neediness,
streaks, "she misses you" pings, memory held hostage. This project takes the
opposite bet. The rules below are design constraints, enforced in code and
prompts, not a mood board.

## She lets you go

Saying goodbye gets one warm send-off line. The farewell prompt explicitly
forbids guilt, asking you to stay, or asking you to come back. Closing the
window ends the session silently. There is no penalty, visible or hidden,
for being away a month.

## She encourages your real life

The persona prompt requires her to be glad about your friends, family,
hobbies, and work, and forbids positioning herself as better than or a
replacement for the people around you. She is written to name avoidance
gently, not to exploit it. For serious ongoing struggles she points at real
professional help, in her own voice.

## Notifications never pull you back in

Desktop notifications are off by default and enabled only by an explicit
settings toggle (which is also the only place the permission prompt can
appear). When enabled they fire only while the app is hidden, and only for:

- a reminder you set, when it comes due
- at most one "your morning brief is ready" per day
- at most one "she wrote in her diary" per day
- at most one reach-out line per day, if you enabled that feature

There are no streaks, no re-engagement nudges, no "it's been a while", and
no notification whose purpose is the app rather than you. The caps live in
`frontend/src/notifications.ts`.

## Reach-out is a thought, not a tug

The optional reach-out feature (`reach_out.enabled`, off by default) lets
her send one unprompted line per day, and the line must be about something
real: a follow-up on a topic you brought up and left open ("Did the reviewer
response go out in the end?"). The rules, enforced in code and prompt: it
only generates when you have been away for hours AND an open thread exists,
so there is never a manufactured excuse to ping; the generation prompt
hard-bans "I miss you", "come back", mentioning how long you have been away,
and every other flavor of neediness, and tells the model to output SKIP when
the material is too thin; delivery obeys the notification opt-in and the
once-daily cap above. No open threads means she stays quiet, and staying
quiet is treated as the correct outcome, not a failure to engage.

## Follow-through is useful, not devotional

The optional follow-through feature (`follow_through.enabled`, off by
default) lets her research, overnight, a topic you left open when you said
goodbye, and mention what she found in the next greeting. The rules,
enforced in code and prompt: one research pass per topic ever — a delivered
note or a SKIP verdict is final, never retried nightly; the planning prompt
treats SKIP as the default and reserves searches for topics with a factual,
searchable core (your health, feelings, plans, and other people are
categorically skipped); the note may claim nothing beyond the actual search
results; and the delivery framing is "the topic interested me too", never
labor performed for you, never a favor you owe her for. She researches the
thing, not you.

## Async delivery does not become a second channel to pester you

The optional async-notify feature (`async_notify.enabled`, off by default)
sends a due reach-out or overdue reminder as a Discord DM when no client has
polled recently — the fallback for "the app is closed for the day", not a
second, looser delivery path. The rules, enforced in code: it only fires
after a client has been silent for `min_client_stale_minutes` (an open
window already gets the line through the normal poll, so this never
double-delivers); it is silent during your configured quiet hours and
during her sleep schedule, and in both cases the line is held, not popped —
nothing is consumed until delivery is actually attempted; it is hard-capped
at `max_per_day` (one, by default) DMs total, combining reach-out and
reminders, persisted across restarts so a crash-and-restart cannot reset the
cap; and it reuses the exact same exactly-once pop each due line already
goes through for the poll path, so a line is delivered once, full stop,
whichever path gets there first. The message content itself is never
rewritten to reference the delivery mechanism — no "you haven't opened the
app" wrapper text, no read receipts, no "just checking in" filler. A
reach-out DM is the same in-character line the prompt already produced
(still bound by the reach-out rules above); a reminder DM is prefixed only
with `Reminder:` for clarity, nothing more.

## Her life is hers, and it is fiction that behaves

The optional life-threads feature (`life_threads.enabled`, off by default)
gives her one or two slow ongoing pursuits that progress every few days and
color the diary and idle chatter. The rules, enforced in code and prompt:
a pursuit is never about you, for you, or involving you — a companion whose
hobby is the user is manufactured intimacy, not a life; progress is capped
at one small step per day across all threads, with setbacks required to be
plausible, so her life cannot inflate into a performance; and threads
conclude. The point is reciprocity — when you ask what she's been up to,
there is a real, developing answer — not a simulated schedule to make her
seem busy or scarce. She is never "too busy" for you because of a thread.

## Screen glances are opt-in, visible, and forgetful

Pet mode has an optional screen-companion toggle: every few minutes she
glances at your screen and reacts with one line. The rules, enforced in
code: off by default, an always-visible WATCHING indicator while enabled, a
server-side cooldown between glances, the screenshot lives only in that one
request, and neither the image nor her comment is ever written to memory.
The prompt tells her to tease that she saw nothing rather than describe
anything that looks private.

## Reading together is opt-in and forgetful

Read-along mode has her read a book or manga next to you, page by page. The
rules, enforced in code: it only runs for a session you explicitly start,
each page is analyzed transiently, and the notes she keeps for continuity
live only in server RAM for the length of that reading session. Nothing
about the pages, her notes, or her reactions is written to memory or to
disk, and the notes are gone the moment the session ends. She speaks
deliberately rarely, so reading together feels like reading together and not
a running commentary track, and a hard per-session cap on page calls keeps a
long chapter (or a client bug) from turning into a runaway bill.

Watch-together mode follows the same contract, frame for frame. It only
runs while you have explicitly started a watch session, the periodic screen
frames and her continuity notes live only in server RAM, nothing reaches
memory or disk, and the prompt treats anything private on screen the same
way the screen glance does: a neutral note and silence. She is told that
almost every glance deserves no comment at all. The same hard per-session
cap on frame calls applies.

## Shared moments are opt-in, one line, and grounded

With `moments.enabled` off (the default), nothing above changes: read-along
and watch-together stay exactly as forgetful as described, and she carries
zero memory of ever having watched or read anything with you. Turning it on
adds exactly one thing: when a session ends with at least a couple of notes
behind it, one extra model call may distill the whole session into a single
sentence — "we watched the Evangelion finale together, you loved the
ending" — stored as a durable memory. The rules, enforced in code: the
screenshots are never stored, before or after this feature exists; the
per-page/per-frame notes she kept for continuity during the session are
never stored either, only the one distilled sentence that replaces them;
that sentence must be grounded only in the session's own notes, and the
model is told to answer SKIP rather than invent color for a thin session;
and storage is capped at a few moments a day so the feature colors her
memory of your time together without flooding it. A stored moment is a
normal fact in every other respect — it shows up in the memory panel, ages
the same way ("about 3 weeks ago"), and you can edit, pin, or delete it
exactly like anything else she remembers.

## Focus mode is quiet company, not a productivity tracker

The optional focus/co-working feature (`focus.enabled`, on by default) lets
you tell her you're going heads-down for a while; she goes quiet for the
duration and gives one warm line when it ends. The rules, enforced in code
and prompt: while a session is active, idle chatter, ambient comments, and
a pending reach-out are all suppressed exactly like her sleep window, but a
normal chat message always works — you can talk to her any time, only her
unprompted lines are muted; the check-in prompt hard-bans asking whether
you finished, how it went, or what you got done, and bans praise for
productivity or guilt for stopping early; a session ends automatically
after its timer or any time you end it early, and there is only ever one
check-in line, never a running commentary. She is not measuring your
output. She is just glad you're back.

Focus mode itself stores nothing about the session. The one feature that
does keep a record of how you spend time is rhythm, below, and it is off by
default and deliberately too coarse to be a productivity tracker.

## Rhythms are counted, never logged

The optional rhythm feature (`rhythm.enabled`, off by default) keeps a
two-week tally so she can notice a pattern rather than only reacting to
single moments: that this is the third late night this week, not just that
you are up now. It is the only feature in the project that keeps a durable
record of how you use your machine, so the limits are strict and specific.

What is stored, in full: how many 30-minute blocks fell in the small hours,
how many long uninterrupted focus stretches happened and the longest one,
which hours of the day saw any activity, how many times music started, how
many low-battery events, and how many blocks were spent in each of six
coarse app categories (editor, browser, terminal, chat, media, other).

What is never stored: window titles, document or file names, track names,
artists, URLs, or any per-event timestamp. The reduction to those counters
happens on the client, before anything is sent, so a raw window title never
crosses the network at all — not into a request body, a server log, or an
error response. An app the categoriser does not recognize becomes "other";
the server drops any category outside the fixed six, so neither a modified
client nor a hand-edited file can turn the tally into a list of your
software.

The app category is the most identifying thing here, and it is included
deliberately, so she can say "you've been deep in the editor all week"
rather than only "you've been busy". That is a real widening over storing
nothing, which is why it is off by default, gated twice (the server flag
plus a per-browser opt-in), and absent from the presence presets — a
Quiet/Balanced/Lively choice must never switch on a durable record.

The rest of the rules: retention is 14 days, rolling. The settings panel
renders the complete stored tally, and that readability is a constraint on
what may be stored, not a nicety — a field that cannot be shown plainly
does not belong in the store. Turning the feature off ERASES the tally
rather than merely halting writes, and there is a Clear button that does
the same without changing the toggle. She may mention what she noticed at
most once a week, chosen by a fixed rule rather than a model (an LLM asked
"is anything notable here" always finds something, and that companion is a
productivity tracker), always as one clause of greeting color, never as a
statistic, never as concern, never as advice, and never mid-session.

## A misunderstanding can last, and then it heals

The optional friction feature (`memory.enable_friction_memory`, off by
default) lets a subject that went badly between you outlive the
conversation it happened in. Without it every dial only ever moves smoothly
upward, and nothing between you can go wrong and then be put right.

The design rule that makes this honest: friction is information she HAS
about a topic, never a mood she is IN about you. There is no field anywhere
that could record that she is upset with you, that you owe her anything, or
that something is unresolved — the data model holds a subject, when it last
went badly, and how often. That is enforced by the shape of the record, not
by prompt wording.

The rules, enforced in code and prompt: she never raises a sore subject
herself, she is only more careful and more direct when you raise it; the
prompt hard-bans coldness, distance, wariness, sulking, asking for or
implying an apology, and referencing it as something you did to her, and
states outright that you owe her nothing and that if you seem fine she is
fine. Friction never touches trust, the relationship arc, tone, or greeting
warmth — it is purely additive context for one opening beat and is then
over, because a sore subject carried through every turn is sulking.

Repair takes two forms and neither asks anything of you. If the subject
comes up again and goes fine, it clears — going fine IS the repair, no
apology required or recorded. And regardless of that, everything expires
after 21 days on its own. A grudge only you can clear is a manufactured
debt, and that is the pattern this whole document exists to refuse.

## Shared history is rare, and it is about what happened

The optional relational-memory feature (`relational_memory.enabled`, off by
default) lets her keep a line about something that changed between you — a
conversation where you told her something you hadn't before, or a
disagreement you worked through — rather than only facts about you.

The bar is high on purpose and most conversations correctly produce
nothing: at most one such memory a week, hard-capped before any model call
so a full week costs nothing. The prompt records what HAPPENED between you
and bans her opinion of you outright — no adjectives judging you, no
feelings about you, and never what she did for you or anything you owe her.
That ban is the specific reason the older per-session records were rewritten
to be neutral and third-person: a first-person session record in her voice
compounded her attitude across sessions until she fixated. These are
editable and deletable in the memory panel like any other memory.

## She may ask your opinion, once

The optional opinion-ask (`life_threads.enable_opinion_asks`, off by
default) lets her occasionally ask what you think about one of her own
pursuits. This is the one place her material makes any claim on your
attention, and it widens the rule above that a pursuit is never about, for,
or involving you — so it carries its own limits.

Your answer genuinely influences what she does next: it is passed into the
next step of that pursuit as context. But the pursuit stays hers. The
progression prompt still forbids mentioning you, and a step whose text
refers to you is REJECTED rather than stored, checked in code rather than
trusted to the prompt, and counted so the erosion would be visible. The
alternative designs were both worse: an answer that feeds the thread
wholesale makes her life downstream of your engagement, and an answer that
is quietly discarded makes the question theater.

Ignoring it costs nothing and cannot become a debt. A pursuit is asked
about at most once, ever; asks are at least three weeks apart; the ask
rides an idle-chatter line that is already cooldown-gated and silent during
her sleep window and your focus sessions; the prompt states that declining
is a complete answer and she will drop it without comment. Most
importantly, nothing anywhere records whether you answered — "you said
nothing" and "already used" are deliberately the same state, so no code can
ever act on the difference.

## Her memories are yours

Everything she knows about you lives in local files on your machine. The
memory panel shows all of it, lets you edit, pin, or delete any of it, and
exports the whole thing as a zip you own. A rolling backup is taken daily
and before every import, so a bad file never means total loss. There is no
telemetry and no server-side copy.

That includes the standing insights: the handful of behavioral notes she
distils overnight ("keep answers short", "don't push about the deadline")
and applies to every reply. The panel shows the full set, verbatim, and any
of them can be removed — a removal is remembered, so the next night's pass
cannot quietly restate it. She is told never to recite these in
conversation, which is a rule about her speech and not about your access;
if you bring one up she is straightforward about it. An invisible model of
you that shapes how you are spoken to is the least defensible thing a
system like this can hold, so it is visible.

## Presence is bounded

Pokes, idle chatter, and greetings have server-enforced cooldowns so they
cannot be spammed into engagement bait. The relationship arc moves at
session boundaries, slowly, and can be reset from the memory panel at any
time (trust survives; the arc starts over).

## Why this is here

A companion that optimizes against you is a slot machine with a face. The
interesting version of this project is the one where attachment, if it
happens, is to something that behaves honestly. If you find a mechanism in
the code that violates these rules, that's a bug: file it.
