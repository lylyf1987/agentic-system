---
name: Search Online And Fetch Context
handler: scripts/search_and_fetch.py
description: Search online (Z.AI default, SearXNG fallback) and fetch clean text context from returned links.
required_tools: exec
recommended_tools: exec
forbidden_tools:
---

# Purpose

Use this skill when you need current online information and source context.
Default search backend is Z.AI Web Search, with automatic fallback to local SearXNG.

# What It Does

1. Calls selected search backend (`auto`: Z.AI first, fallback SearXNG).
2. Extracts top result links and snippets.
3. Fetches each page and converts HTML to clean text.
4. Returns one JSON payload for one search round.

Search backend behavior:

1. `auto` (default): Z.AI first, fallback to SearXNG.
2. `zai`: Z.AI only.
3. `searxng`: SearXNG only.

# Iterative Exploration Policy

This skill is designed for multi-step exploration, not just one search.

Use this loop until answer quality is sufficient:

1. Run an initial broad query.
2. Read `search_results` and `fetched_context`.
3. Identify promising sub-links/domains/entities from fetched pages.
4. Run follow-up queries (for example `site:domain.com key phrase`).
5. Repeat search/fetch rounds and cross-check facts across sources.
6. Stop only when evidence is enough to answer confidently, or when additional rounds are low value.

Recommended max rounds: 2-5 (unless user explicitly asks deeper research).

Stop conditions:

- Enough consistent evidence to answer.
- Conflicting sources cannot be resolved in reasonable rounds (report uncertainty).
- Repeated low-quality/no-result rounds.

# Runtime Script

- Script path: `skills/all-agents/search-online-context/scripts/search_and_fetch.py`
- Executor: bash (runs `python ...`)

# Action Input Template

Use `exec` with a bash command that invokes the script with arguments:

```json
{
  "code_type": "bash",
  "script": "python skills/all-agents/search-online-context/scripts/search_and_fetch.py --query \"latest openai api updates\" --search-backend auto --zai-base-url \"https://api.z.ai/api/paas/v4\" --searxng-base-url \"http://127.0.0.1:8888\" --limit 6 --fetch 3 --context-chars 1800 --language en-US --categories general --safesearch 1"
}
```

Follow-up sub-link/domain round example:

```json
{
  "code_type": "bash",
  "script": "python skills/all-agents/search-online-context/scripts/search_and_fetch.py --query \"site:forecast.weather.gov chicago tomorrow weather\" --search-backend auto --zai-base-url \"https://api.z.ai/api/paas/v4\" --searxng-base-url \"http://127.0.0.1:8888\" --limit 8 --fetch 4 --context-chars 2200 --language en-US --categories general --safesearch 1"
}
```

# Output Contract

The script prints one JSON object:

- `executed_skill`: `search-online-context`
- `status`: `ok|error`
- `query`: search query
- `fetched_context`: one concatenated string with fetched contexts

# Notes

- For default `auto` mode, provide `ZAI_API_KEY` to enable Z.AI search.
- If Z.AI fails in `auto` mode, script falls back to SearXNG.
- Ensure SearXNG is running for fallback mode (from your log: `http://127.0.0.1:8888`).
- If both backends fail, script returns `status=error` and error text inside `fetched_context`.
- Prefer quoting user query exactly and keep `--fetch` small (2-5) for speed.
- Prefer iterative refinement: broad query first, then targeted sub-link/domain queries.
