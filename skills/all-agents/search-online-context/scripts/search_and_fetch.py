#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from html import unescape
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_EXECUTED_SKILL = "search-online-context"


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


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    timeout: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    req_headers = {
        "User-Agent": "Mozilla/5.0 (AgenticSystemSkill/1.0)",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if isinstance(headers, dict):
        req_headers.update(headers)
    req = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=req_headers,
        method="POST",
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

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        href = str(item.get("url", "")).strip()
        if not href or not href.lower().startswith(("http://", "https://")):
            continue
        if href in seen:
            continue
        seen.add(href)
        title = str(item.get("title", "")).strip() or href
        snippet = _clean_inline_html(str(item.get("content", "")))
        engines = item.get("engines", [])
        if not isinstance(engines, list):
            engines = []
        results.append(
            {
                "rank": len(results) + 1,
                "title": title,
                "url": href,
                "snippet": snippet,
                "engines": [str(v) for v in engines if str(v).strip()],
            }
        )
        if len(results) >= limit:
            break
    return results


def search_zai(
    *,
    base_url: str,
    api_key: str,
    query: str,
    limit: int,
    timeout: int,
    search_engine: str,
    search_domain_filter: str,
    search_recency_filter: str,
) -> list[dict[str, Any]]:
    token = str(api_key).strip()
    if not token:
        raise ValueError("missing ZAI_API_KEY for Z.AI web search")
    normalized_base = str(base_url).strip().rstrip("/")
    if not normalized_base:
        raise ValueError("empty Z.AI base URL")
    if normalized_base.endswith("/web_search"):
        endpoint = normalized_base
    else:
        endpoint = f"{normalized_base}/web_search"

    count = max(1, min(50, int(limit)))
    payload: dict[str, Any] = {
        "search_engine": str(search_engine).strip() or "search-prime",
        "search_query": query,
        "count": count,
    }
    domain_filter = str(search_domain_filter).strip()
    if domain_filter:
        payload["search_domain_filter"] = domain_filter
    recency_filter = str(search_recency_filter).strip()
    if recency_filter:
        payload["search_recency_filter"] = recency_filter

    raw = _http_post_json(
        endpoint,
        payload=payload,
        timeout=timeout,
        headers={"Authorization": f"Bearer {token}"},
    )
    rows = raw.get("search_result", [])
    if not isinstance(rows, list):
        rows = []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        href = str(item.get("link", "")).strip()
        if not href or not href.lower().startswith(("http://", "https://")):
            continue
        if href in seen:
            continue
        seen.add(href)
        title = str(item.get("title", "")).strip() or href
        snippet = _clean_inline_html(str(item.get("content", "")).strip())
        out.append(
            {
                "rank": len(out) + 1,
                "title": title,
                "url": href,
                "snippet": snippet,
                "engines": [f"zai:{payload['search_engine']}"],
            }
        )
        if len(out) >= count:
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
    timeout: int,
    searxng_base_url: str,
    language: str,
    categories: str,
    safesearch: int,
    search_backend: str,
    zai_base_url: str,
    zai_api_key: str,
    zai_search_engine: str,
    zai_search_domain_filter: str,
    zai_search_recency_filter: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "executed_skill": _EXECUTED_SKILL,
        "status": "ok",
        "query": query,
        "fetched_context": "",
    }

    backend = str(search_backend).strip().lower() or "auto"
    errors: list[str] = []
    backend_used = "none"
    results: list[dict[str, Any]] = []

    should_try_zai = backend in {"auto", "zai"}
    should_try_searxng = backend in {"auto", "searxng"}

    if should_try_zai:
        try:
            results = search_zai(
                base_url=zai_base_url,
                api_key=zai_api_key,
                query=query,
                limit=limit,
                timeout=timeout,
                search_engine=zai_search_engine,
                search_domain_filter=zai_search_domain_filter,
                search_recency_filter=zai_search_recency_filter,
            )
            if results:
                backend_used = "zai"
            else:
                errors.append("zai: no results found")
        except Exception as exc:
            errors.append(f"zai: {exc}")

    if not results and should_try_searxng:
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
            if results:
                backend_used = "searxng"
            else:
                errors.append("searxng: no results found")
        except Exception as exc:
            errors.append(f"searxng: {exc}")

    if not results:
        payload["status"] = "error"
        payload["fetched_context"] = "search_error: " + " | ".join(errors if errors else ["no results found"])
        return payload

    fetched_rows: list[dict[str, Any]] = []
    for item in results[: max(0, fetch_count)]:
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
        f"backend={backend_used}"
    )
    if errors:
        summary += f"; fallback_notes={' | '.join(errors)}"
    details = _format_fetched_context(fetched_rows)
    payload["fetched_context"] = summary if not details else f"{summary}\n\n{details}"

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search via Z.AI (default) with SearXNG fallback and fetch context from top results."
    )
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--limit", type=int, default=6, help="Max search results to keep")
    parser.add_argument("--fetch", type=int, default=3, help="How many top links to fetch for context")
    parser.add_argument("--context-chars", type=int, default=1800, help="Max chars kept per fetched page")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    parser.add_argument(
        "--searxng-base-url",
        default=os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8888"),
        help="SearXNG base URL",
    )
    parser.add_argument("--language", default="en-US", help="SearXNG language code")
    parser.add_argument("--categories", default="general", help="SearXNG categories")
    parser.add_argument("--safesearch", type=int, default=1, help="SearXNG safesearch level (0-2)")
    parser.add_argument(
        "--search-backend",
        choices=["auto", "zai", "searxng"],
        default=os.getenv("SEARCH_BACKEND", "auto"),
        help="Search backend selection. auto: Z.AI first then SearXNG fallback.",
    )
    parser.add_argument(
        "--zai-base-url",
        default=os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4"),
        help="Z.AI API base URL for web_search endpoint",
    )
    parser.add_argument(
        "--zai-api-key",
        default=os.getenv("ZAI_API_KEY", ""),
        help="Z.AI API key (can also come from ZAI_API_KEY env)",
    )
    parser.add_argument(
        "--zai-search-engine",
        default=os.getenv("ZAI_SEARCH_ENGINE", "search-prime"),
        help="Z.AI search engine name",
    )
    parser.add_argument(
        "--zai-search-domain-filter",
        default=os.getenv("ZAI_SEARCH_DOMAIN_FILTER", ""),
        help="Optional Z.AI domain filter",
    )
    parser.add_argument(
        "--zai-search-recency-filter",
        default=os.getenv("ZAI_SEARCH_RECENCY_FILTER", "noLimit"),
        help="Optional Z.AI recency filter (for example: noLimit, oneDay, oneWeek)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        query=str(args.query).strip(),
        limit=max(1, int(args.limit)),
        fetch_count=max(0, int(args.fetch)),
        context_chars=max(200, int(args.context_chars)),
        timeout=max(5, int(args.timeout)),
        searxng_base_url=str(args.searxng_base_url).strip(),
        language=str(args.language).strip() or "en-US",
        categories=str(args.categories).strip() or "general",
        safesearch=max(0, min(2, int(args.safesearch))),
        search_backend=str(args.search_backend).strip().lower() or "auto",
        zai_base_url=str(args.zai_base_url).strip(),
        zai_api_key=str(args.zai_api_key),
        zai_search_engine=str(args.zai_search_engine).strip() or "search-prime",
        zai_search_domain_filter=str(args.zai_search_domain_filter).strip(),
        zai_search_recency_filter=str(args.zai_search_recency_filter).strip() or "noLimit",
    )
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
