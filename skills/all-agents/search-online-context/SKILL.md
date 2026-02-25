---
name: Search Online And Fetch Context
handler: scripts/search_and_fetch.py
description: Search online with SearXNG and fetch clean text context from returned links.
required_tools: exec
recommended_tools: exec
forbidden_tools:
---

# Purpose

Use this skill when you need current online information and source context.

# What It Does

1. Searches via SearXNG.
2. Extracts top result links and snippets.
3. Fetches each page and converts HTML to clean text.
4. Returns one JSON payload for one search round.

# Iterative Exploration Policy

This skill is designed for multi-step exploration, not just one search.

Use this loop until answer quality is sufficient:

1. Run an initial broad query.
2. Read `fetched_context`.
3. Identify promising sub-links/domains/entities from fetched pages.
4. Run follow-up queries (for example `site:domain.com key phrase`).
5. Repeat search/fetch rounds and cross-check facts across sources.
6. Stop when evidence is enough to answer confidently.

Recommended max rounds: 2-5 (unless user explicitly asks deeper research).

# Runtime Script

- Script path: `skills/all-agents/search-online-context/scripts/search_and_fetch.py`
- Executor: `python` via `script_path` + `script_args`

# Runtime Contract

1. `stdout` must contain one final JSON object.
2. Use `stderr` only for unexpected runtime failures.
3. Keep `stdout` concise but informative so `workflow_hist` remains readable.

# Action Input Template

Use `exec` with `code_type=python`, `script_path`, and `script_args` array:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/search-online-context/scripts/search_and_fetch.py",
  "script_args": [
    "--query", "site:forecast.weather.gov chicago tomorrow weather",
    "--searxng-base-url", "http://127.0.0.1:8888",
    "--limit", "8",
    "--fetch", "4",
    "--context-chars", "2500",
    "--max-total-context-chars", "15000",
    "--language", "en-US",
    "--categories", "general",
    "--safesearch", "1"
  ]
}
```

# Output Contract

The script prints one JSON object:

- `executed_skill`: `search-online-context`
- `status`: `ok|error`
- `query`: search query
- `fetched_context`: one concatenated string with fetched contexts

# Notes

- Ensure SearXNG is running (for example `http://127.0.0.1:8888`).
- If search/fetch fails, script returns `status=error` with error text in `fetched_context`.
- Defaults: `--limit 8`, `--fetch 4`, `--context-chars 2500`, `--max-total-context-chars 15000`.
