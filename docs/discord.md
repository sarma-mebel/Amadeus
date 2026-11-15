# Discord bridge

The Discord integration is really two things, and the first one matters
more than the second. It is the away-channel: when a reach-out line or a
due reminder is generated and nobody's browser or pet window is open to
receive it, `amadeus/async_notify.py` delivers it as a Discord DM instead
of letting it go stale. That is the whole point of a text channel here,
her self-initiated lines still reach you when you're away from the
desktop, and a DM notification is text by nature, so text-only is not a
limitation for this role.

The second thing is a plain text chat fallback: an optional bridge that
lets the companion chat over Discord DMs and @mentions when you'd rather
type than open the app. It shares the same ChatEngine and memory tiers as
the CLI: each Discord user gets their own persistent chat session, keyed
as `discord-<id>`, so recall/facts/relationship arc all work per-person
the same way they do for the CLI or web app.

Neither path carries voice, the avatar, or presence, that's the tradeoff
of Discord as a surface, and it's fine because delivering a line while
you're away is worth more here than replicating the full experience. No
slash commands, no channel reading beyond a DM or an explicit mention.
Both the chat bridge and the async notify path are off by default.

## What it does and does not do

- Replies to direct messages (if `reply_to_dms` is on, the default).
- Replies when @mentioned in a server channel (if `reply_to_mentions` is on,
  the default).
- Ignores everything else: other bots, messages in channels it wasn't
  mentioned in, slash commands.
- Splits long replies across multiple Discord messages at sentence
  boundaries (Discord caps a single message at 2000 characters).
- Detects a real goodbye the same way the CLI's `/quit` does
  (`amadeus.farewell.is_goodbye`) and sends one warm send-off line instead
  of a normal reply, then finalizes that user's session.

## The away-channel: async notify

This is the part that ties Discord back into the project's core idea,
that the relationship keeps going even when you're not sitting at the
computer. It needs only a bot token, not the full chat bridge above, and
it needs no `discord.py` at all: `amadeus/async_notify.py` sends DMs with
bare stdlib `urllib` REST calls, because this path only ever fires
one-off messages and never listens for anything.

What it delivers, checked in this order, whichever is due first wins for
that tick:

- An unconsumed reach-out line for the day, if `reach_out.enabled` has
  generated one and nothing has popped it yet.
- Any due reminders, if `reminders.enabled`, sent as one message
  prefixed `Reminder: `.

Both of those features are off by default, so async notify has nothing to
deliver until at least one of them is turned on. `daily_brief` and
`follow_through` notes are not delivered directly through this path; they
land through the normal greeting instead.

Delivery only fires when a client (browser or pet window) has gone quiet
for a while, so an open app keeps using the instant 30s poll and this
stays a true fallback. Add an `async_notify` section to
`config.local.yaml` (there's no example block for it in
`config/config.example.yaml` yet, so this is written from
`amadeus/config.py`'s `AsyncNotifySettings`):

```yaml
async_notify:
  enabled: true             # off by default
  channel: discord          # only "discord" exists today
  discord_user_id: ""       # the Discord user id to DM; required for delivery to do anything
  min_client_stale_minutes: 30   # no client seen for this long before a DM is even considered
  quiet_hours_start: 22     # local hour, wraps midnight; start == end disables quiet hours
  quiet_hours_end: 8
  max_per_day: 1            # hard cap, reach-out + reminders combined
```

It reuses `discord.token` (or the `DISCORD_BOT_TOKEN` env var) rather than
a second token field, since it's the same bot credential either way. So
the minimum setup for just the away-channel, without the full chat
bridge, is: create a bot (step 1 below), set `discord.token`, and set
`async_notify.enabled: true` plus your `discord_user_id`. You don't need
to invite the bot to a server or turn on `reply_to_dms`/
`reply_to_mentions` for this path; it only ever opens a DM channel and
posts to it.

Delivery is best-effort by contract: a failed send never raises and never
takes down the scheduler tick, it just counts
`notify.delivery_failures`. Skips are counted too
(`notify.skipped_client_active`, `notify.skipped_quiet_hours`,
`notify.skipped_daily_cap`), so a quiet night and a misconfigured
`discord_user_id` look different in `/api/status` instead of both looking
like silence.

## 1. Create a bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
   and create a new application.
2. Under **Bot**, add a bot user and copy its token. Keep this secret; treat
   it like an API key.
3. Under **Bot > Privileged Gateway Intents**, turn on **Message Content
   Intent**. The bridge needs it to read message text (DMs and mentions
   alike).

## 2. Invite it to a server (optional, only needed for @mentions)

DMs work as soon as the bot exists and is running; you don't need to invite
it anywhere for that. To let it respond to @mentions in a server, build an
invite URL under **OAuth2 > URL Generator** with:

- Scopes: `bot`
- Bot permissions: **Send Messages**, **Read Message History**

Open the generated URL and add the bot to your server.

## 3. Configure Amadeus

Install the extra (discord.py isn't part of the base install):

```bash
pip install ".[discord]"
```

Add a `discord` section to `config.local.yaml`:

```yaml
discord:
  enabled: true
  token: null              # or set the DISCORD_BOT_TOKEN env var instead
  allowed_user_ids: []      # empty = anyone may DM/mention it; otherwise only these Discord user ids
  reply_to_mentions: true
  reply_to_dms: true
```

The token can also come from the `DISCORD_BOT_TOKEN` environment variable
instead of the config file, the same pattern as the other provider keys.
Useful if you don't want the token sitting in `config.local.yaml`.

`allowed_user_ids` is a list of Discord user ids (the numeric snowflake, not
a username). Leave it empty for a personal bot only you know about; set it
to lock the bot down to specific people once it's reachable by anyone who
finds it or is in a shared server with it.

## 4. Run it

```bash
python -m amadeus.discord_bridge --config config.local.yaml
```

It builds one LLM provider (same `providers.llm` config as the CLI/web app)
and connects to Discord. Leave it running in a terminal, a `screen`/`tmux`
session, or a small systemd unit/process manager; there's no built-in
daemonization.

## Notes

- Memory is per Discord user id and lives alongside the CLI's own memory
  files, scoped the same way the web app scopes per-user memory
  (`memory_store_users/discord-<id>.json` etc.), so a Discord conversation
  never mixes with the CLI's or a web session's.
- The bridge does not run the tool loop (web search, code interpreter,
  etc.). It's a plain chat turn, kept deliberately simple for a first cut.
- A failed provider call (rate limit, invalid key, any other error) never
  crashes the bridge: it replies with the same in-character canned failure
  line the CLI and web app use, and counts it under `discord.turn_failures`
  in the process telemetry.
