# claude-history-search

Use this skill when you need to find, narrow, or quote prior Claude Code conversation context with `claude-history`.

## Workflow

Start with a bounded lexical search so user search configuration cannot accidentally switch the agent into a slower mode:

```sh
claude-history agent search --lexical "auth cache bug"
```

The output is protocol text, not JSON. Global search is grouped by conversation, with readable snippets after `|` and copyable `read ref=... focus=...` lines:

```text
protocol agent-search v=2 mode=lexical groups=1 hits=1
query text=auth%20cache%20bug hits=1
groups count=1
conversation rank=1 ref=ch_1234abcd5678 score=12.500000 hits=1 total=1 | fix auth cache
hit ref=ch_1234abcd5678 source=lexical score=12.500000 focus=m8..m8 | auth cache bug repro
read ref=ch_1234abcd5678:m7..m9 focus=m8..m8
```

Copy the emitted `read ref=... focus=...` line as an instruction for the next command. Do not treat hit order, scores, ranks, or chunks as stable addresses.

If the top hit is probably the right conversation but you need better evidence inside it, narrow first:

```sh
claude-history agent within ch_1234abcd5678 --lexical "auth cache bug"
```

If you need to choose a section before reading, outline the conversation:

```sh
claude-history agent outline ch_1234abcd5678
```

Then read only the emitted range and preserve `focus=` in `--focus`:

```sh
claude-history agent read ch_1234abcd5678:m7..m9 --focus m8..m8
```

Use one `agent read` command per emitted `read` line unless you qualify focus with the conversation ref, for example `--focus ch_1234abcd5678:m8..m8`. A bare `--focus m8..m8` is only unambiguous when reading one conversation.

Do not read a full transcript by default. Prefer `search`, then `within` or `outline`, then a bounded `read` range. Use `--flat` only when you need raw message-hit ordering, `--hits-per-conv` when you need more evidence from each conversation, and `--all-hits` only when duplicate suppression hides relevant tool-heavy evidence. Use `--tools`, `--tool-results`, `--thinking`, or `--subagents` only when that hidden content is relevant.

## Query mode guidance

Use `--lexical` for normal search and identifier-like terms such as `api_key`, `build_id`, or `AgentSearchRequest`. Use `--exact` or quoted text for exact tokens, secrets, IDs, error strings, and case-sensitive identifiers:

```sh
claude-history agent search --exact "DEPLOYMENT_TOKEN"
claude-history agent within ch_1234abcd5678 --lexical "api_key"
```

Use `--semantic` only when conceptual wording matters more than exact text. Use `--hybrid` when semantic recall is useful but exact terms still matter.
