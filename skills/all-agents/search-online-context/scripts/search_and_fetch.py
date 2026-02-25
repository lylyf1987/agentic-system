#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from html import unescape
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_EXECUTED_SKILL = "search-online-context"


def _ok(query: str, fetched_context: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "ok",
        "query": str(query),
        "fetched_context": str(fetched_context),
    }


def _err(query: str, fetched_context: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "error",
        "query": str(query),
        "fetched_context": str(fetched_context),
    }


def _http_get_text(url: str, timeout: int) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (AgenticSystemSkill/1.0)",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        data = resp.read(1_500_000)
    return data.decode(charset, errors="replace")


def _http_get_json(url: str, timeout: int) -> dict[str, Any]:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (AgenticSystemSkill/1.0)",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        body = resp.read(1_500_000).decode(charset, errors="replace")
    parsed = json.loads(body)
    return parsed if isinstance(parsed, dict) else {}


def _clean_text(text: str) -> str:
    out = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", text)
    out = re.sub(r"(?s)<!--.*?-->", " ", out)
    out = re.sub(r"(?is)<[^>]+>", " ", out)
    out = unescape(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _clean_inline_html(text: str) -> str:
    out = re.sub(r"(?is)<[^>]+>", " ", text)
    out = unescape(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def search_searxng(
    *,
    base_url: str,
    query: str,
    limit: int,
    timeout: int,
    language: str,
    categories: str,
    safesearch: int,
) -> list[dict[str, Any]]:
    base = base_url.rstrip("/")
    params = {
        "q": query,
        "format": "json",
        "language": language,
        "categories": categories,
        "safesearch": str(safesearch),
    }
    url = f"{base}/search?{urlencode(params)}"
    payload = _http_get_json(url, timeout=timeout)
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        href = str(item.get("url", "")).strip()
        if not href.lower().startswith(("http://", "https://")):
            continue
        if href in seen:
            continue
        seen.add(href)
        title = str(item.get("title", "")).strip() or href
        snippet = _clean_inline_html(str(item.get("content", "")))
        engines = item.get("engines", [])
        if not isinstance(engines, list):
            engines = []
        out.append(
            {
                "rank": len(out) + 1,
                "title": title,
                "url": href,
                "snippet": snippet,
                "engines": [str(v) for v in engines if str(v).strip()],
            }
        )
        if len(out) >= limit:
            break
    return out


def fetch_context(url: str, max_chars: int, timeout: int) -> tuple[str, str]:
    html = _http_get_text(url, timeout=timeout)
    text = _clean_text(html)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "...", ""
    return text, ""


def _format_fetched_context(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    blocks: list[str] = []
    for row in rows:
        title = str(row.get("title", "")).strip()
        url = str(row.get("url", "")).strip()
        status = str(row.get("status", "")).strip()
        context = str(row.get("context", "")).strip()
        error = str(row.get("error", "")).strip()
        blocks.append(
            "\n".join(
                [
                    f"# {title}".strip(),
                    f"url: {url}",
                    f"status: {status}",
                    "context:",
                    context if context else "(empty)",
                    f"error: {error}" if error else "error: (none)",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def run(
    *,
    query: str,
    limit: int,
    fetch_count: int,
    context_chars: int,
    max_total_context_chars: int,
    timeout: int,
    searxng_base_url: str,
    language: str,
    categories: str,
    safesearch: int,
) -> dict[str, Any]:
    try:
        results = search_searxng(
            base_url=searxng_base_url,
            query=query,
            limit=limit,
            timeout=timeout,
            language=language,
            categories=categories,
            safesearch=safesearch,
        )
    except Exception as exc:
        return _err(query=query, fetched_context=f"search_error: searxng: {exc}")

    if not results:
        return _err(query=query, fetched_context="search_error: searxng: no results found")

    fetched_rows: list[dict[str, Any]] = []
    remaining_context_chars = max(0, int(max_total_context_chars))
    for item in results[: max(0, fetch_count)]:
        if remaining_context_chars <= 0:
            break
        rank = int(item.get("rank", 0))
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        row: dict[str, Any] = {
            "rank": rank,
            "title": title,
            "url": url,
            "status": "ok",
            "context": "",
            "error": "",
        }
        try:
            context, error = fetch_context(url=url, max_chars=context_chars, timeout=timeout)
            if context and len(context) > remaining_context_chars:
                context = context[:remaining_context_chars].rstrip() + "..."
                error = (f"{error} | " if error else "") + "truncated by max_total_context_chars"
            if context:
                remaining_context_chars = max(0, remaining_context_chars - len(context))
            row["context"] = context
            row["error"] = error
            if error:
                row["status"] = "partial"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        fetched_rows.append(row)

    summary = (
        "search_ok: "
        f"query={query!r}; search_results={len(results)}; fetched={len(fetched_rows)}; "
        f"backend=searxng; max_total_context_chars={max_total_context_chars}"
    )
    details = _format_fetched_context(fetched_rows)
    return _ok(query=query, fetched_context=summary if not details else f"{summary}\n\n{details}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search via SearXNG and fetch context from top results."
    )
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--limit", type=int, default=8, help="Max search results to keep")
    parser.add_argument("--fetch", type=int, default=4, help="How many top links to fetch for context")
    parser.add_argument("--context-chars", type=int, default=2500, help="Max chars kept per fetched page")
    parser.add_argument(
        "--max-total-context-chars",
        type=int,
        default=15000,
        help="Global max chars across all fetched contexts",
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    parser.add_argument(
        "--searxng-base-url",
        default=os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8888"),
        help="SearXNG base URL",
    )
    parser.add_argument("--language", default="en-US", help="SearXNG language code")
    parser.add_argument("--categories", default="general", help="SearXNG categories")
    parser.add_argument("--safesearch", type=int, default=1, help="SearXNG safesearch level (0-2)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query = str(args.query).strip()
    try:
        result = run(
            query=query,
            limit=max(1, int(args.limit)),
            fetch_count=max(0, int(args.fetch)),
            context_chars=max(200, int(args.context_chars)),
            max_total_context_chars=max(1000, int(args.max_total_context_chars)),
            timeout=max(5, int(args.timeout)),
            searxng_base_url=str(args.searxng_base_url).strip(),
            language=str(args.language).strip() or "en-US",
            categories=str(args.categories).strip() or "general",
            safesearch=max(0, min(2, int(args.safesearch))),
        )
        print(json.dumps(result, ensure_ascii=True))
        return 0 if str(result.get("status", "")).strip().lower() == "ok" else 1
    except Exception as exc:
        out = _err(query=query, fetched_context=f"search_error: unexpected exception: {exc}")
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
