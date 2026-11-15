# How memory works

Amadeus remembers you across sessions. Not by stuffing the whole chat history
into every prompt, but through a layered system that keeps recent conversation
verbatim, compresses older conversation into summaries, and stores durable
facts for semantic recall weeks or months later. This doc explains the layers,
the lifecycle of a memory, and where to look when recall feels off.

Everything here is configured under the `memory:` section of your config
(`config.local.yaml`), unless noted otherwise. Long-term memory is on by
default (`memory.enable_long_term: true`).

## The three tiers

1. **Short-term context.** A rolling window of the most recent messages, sent
   with every request verbatim. This is what makes the current conversation
   coherent.
2. **Rolling summary.** As the short-term window fills, an async background
   job compresses older messages into a running summary of the session.
   `memory.summary_model` can point at a cheaper model than the chat model
   (`null` reuses the chat model), so summarization does not have to cost
   what conversation costs.
3. **Long-term vector store.** Durable records (facts about you, episode
   summaries of past sessions, reflections) are embedded and stored on disk
   (`memory_embeddings.json` by default, or Chroma via
   `memory.vector_backend: chroma`). Each turn, the relevant ones are
   recalled by semantic similarity and injected into the prompt.

Memory is scoped per user and, when `memory.per_character_memory` is on, per
character, so a guest character never sees another character's history.

## What gets recalled each turn, and why facts only

Per-turn recall surfaces durable *facts* only (`memory.recall_facts_only`,
default on). Episodes and reflections are still stored and still power other
features, but they are filtered out of per-turn recall on purpose: an episode
re-narrates a past session, which duplicates what the rolling summary already
says. Early versions recalled everything and the visible symptom was
repetition, she kept retelling past conversations.

Recall is also deduplicated and gated:

- Items near-identical to the current summary are dropped
  (`memory.recall_summary_dedupe`, default 0.82 similarity).
- Facts already discussed in the recent conversation window are dropped
  (`memory.recall_recent_dedupe`, default 0.6 word overlap), so she does not
  re-inject the same memory every turn after you have moved on.
- Very short messages ("hi", "ok") skip recall entirely
  (`memory.recall_min_query_chars`, default 10).
- A relevance floor filters out weak matches. The effective threshold is the
  higher of `memory.recall_min_score` and the embedder's own measured noise
  floor, so unrelated memories do not ride along on every turn.

## Temporal grounding

Recalled memories carry a humanized age from their stored timestamp, so the
model sees "(about 3 weeks ago)" next to each item. That is what lets her say
"you mentioned a while back..." instead of reciting an old fact as if it were
current.

Facts are also bitemporal: when the user actually places an event in time
("my defense is on the 15th"), the extractor stores that event time
separately from the learn time. Recall then renders both, and the event side
can point forward: "(2 days ago; happens in about 3 weeks)". Telling her
about something after the fact reads correctly too: "(earlier today;
happened about 2 months ago)". The learn time keeps driving all ranking and
decay; the event time is presentation only, so existing stores need no
migration. The model-facing memory tool takes the same optional `when` for
explicit "remember this" requests.

## Nothing is destroyed: the archive

Memory cleanup never hard-deletes. When a fact is superseded by a newer
version ("I moved to Berlin" replacing "I live in Munich"), forgotten on
request, edited, or evicted because the store hit its cap, the old record is
appended to an archive file next to the vector store
(`<embeddings path>_archive.jsonl`) with its text, metadata, timestamp, and
the reason it was archived. It leaves the active store, so recall stops
surfacing it, but the history stays on disk and superseding facts keep a
reference to what they replaced.

The caps: `memory.max_facts` (default 2000) bounds stored facts, with the
weakest unpinned facts archived past the cap, and `memory.max_episodes`
(default 200) bounds episodes.

## Telling her to remember (and forget)

Facts are normally extracted automatically in the background. On top of that
there is a model-facing memory tool, enabled by default, that handles
explicit instructions deterministically:

- "Remember that I..." stores the fact directly.
- "No, I moved to Berlin, not Munich" archives the outdated fact and stores
  the correction, with provenance linking the two.
- "Forget what I said about..." archives the matching fact.

The tool is deliberately cautious about forgetting: if no stored fact matches
what you described closely enough (a 0.75 similarity bar), it declines and
tells her the nearest candidates instead of guessing, so she asks you rather
than deleting the wrong thing. An explicit forget does remove a pinned fact;
your direct instruction outranks the pin.

The tool is write-only by design. Reading memories is the recall system's
job, and it happens automatically.

## Pinning

In the web UI's memory panel (and its synapse map view) you can pin
individual memories. Pinned records are protected from automatic cleanup:
supersede, deduplication, and cap eviction all skip them. Only your explicit
actions (a direct forget, or unpin-then-delete in the UI) remove a pinned
memory.

## Embedders

Recall quality depends on the embedder, set by `providers.embedding`:

| Value | What it is | Needs |
|---|---|---|
| `sentence_transformers` | Local semantic embeddings (MiniLM), the config default | the `[memory]` extra (pulls in torch) |
| `gemini` | Semantic embeddings via the Gemini API (`gemini-embedding-2`), the best recall quality | a Gemini API key (same one as the chat provider) |
| `hashing` | Dependency-free keyword hashing, lexical matching only | nothing; this is what the packaged desktop build uses |

If the configured embedder is unavailable (missing extra, missing key), the
app falls back to `hashing` with a startup warning rather than crashing.

**Switching embedders changes the vector dimension.** Recall detects the
mismatch against your existing store and disables itself, on purpose, until
you re-embed:

```bash
python scripts/migrate_memory.py --config config.local.yaml --dry-run
python scripts/migrate_memory.py --config config.local.yaml
```

The script is idempotent, backs every file up as `<name>.bak-migrate` first,
covers per-user and per-character stores, and also retro-prunes duplicate
episodes.

## Storage backend: JSON or SQLite

By default all stores are JSON files. `memory.storage_backend: sqlite`
switches the message store and vector store to SQLite (stdlib, WAL mode, no
new dependency). To convert an existing installation:

```bash
python scripts/convert_memory_sqlite.py --config config.local.yaml --dry-run
python scripts/convert_memory_sqlite.py --config config.local.yaml
```

The converter is one-way and safe: it writes new `.db` files, preserves the
original `.json` files as `.bak-migrate`, and does not flip
`storage_backend` in your config. Confirm the app runs against the converted
store, then change the setting yourself.

## Entity graph (experimental, off by default)

`memory.graph_enabled: true` builds a lightweight entity/relation graph from
the facts already being extracted (no extra LLM calls). When on, memories
mentioning an entity from your current message are merged into recall ahead
of plain vector matches. It stores as a JSON sidecar next to the message
store, or inside the SQLite database when that backend is active.

## Speculative recall

When you speak instead of type, the server pre-starts the memory recall for
your transcribed text while the client is still round-tripping the
transcript back into a chat request (`memory.speculative_recall`, default
on). The pre-computed result is used only when the chat request that arrives
matches the transcript, and it expires after
`memory.speculative_recall_ttl_seconds` (default 10). The effect is shaving
the query-embedding latency off voice turns; it never changes what a turn
recalls.

## Context token budget

`memory.context_token_budget` (default 12000 estimated tokens) caps the
assembled request. Over-budget requests are trimmed conservatively: recalled
items go first (lowest-ranked first), then the oldest short-term messages,
never the system prompt and never your current message. Set it to `0` to
disable trimming.

## When memory feels off

Silent memory failure is a bug class this project takes seriously, so
degraded paths count themselves. Check, in order:

1. **`GET /api/status`** (or `/status` in the CLI). The `counters` section is
   the health surface; all zeros is the healthy state. Ones worth knowing:
   - `memory.recall_embed_timeouts`: query embedding is timing out
     (network embedder), recall is degrading to empty.
   - `memory.recall_floor_filtered`: how much the relevance floor is
     dropping (high numbers with bad recall may mean the floor is fighting
     your embedder).
   - `memory.speculative_recall_hits` / `_misses`: whether voice-turn
     precomputation is landing.
   - `memory.profile_load_corrupt` / `memory.vector_store_load_corrupt`: a
     store file failed to load and was quarantined as `.bak-corrupt`.
   - `memory.orphaned_user_messages_healed`: interrupted turns healed at
     load.
   The `durations` section includes `memory.recall_ms` percentiles.
2. **Settings > Diagnostics** in the web UI renders the same counters plus
   client-side ones, and highlights recall-degradation counters with hints.
3. **Embedder mismatch.** If you recently changed `providers.embedding` (or
   the Gemini embedding dimension), recall silently disables itself until
   `scripts/migrate_memory.py` runs. This is the first thing to rule out
   after a config change.
4. **Dummy fallback.** If the whole assistant feels canned, you may be on
   the dummy provider (missing key or extra); memory extraction needs a real
   LLM. `GET /api/status` shows what is actually loaded.
