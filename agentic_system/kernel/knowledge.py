from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4


class KnowledgeEngine:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.knowledge_root = self.workspace / "knowledge"
        self.docs_root = self.knowledge_root / "docs"
        self.index_root = self.knowledge_root / "index"
        self.docs_root.mkdir(parents=True, exist_ok=True)
        self.index_root.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.index_root / "catalog.json"

    def load_knowledge_index(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.catalog_path.exists():
            return []
        try:
            raw = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        rows = [item for item in raw if isinstance(item, dict)]
        return rows[:limit]

    def load_knowledge(self, doc_ids: str | list[str]) -> list[dict[str, Any]]:
        if isinstance(doc_ids, str):
            requested = [doc_ids.strip()] if doc_ids.strip() else []
        else:
            requested = [str(item).strip() for item in doc_ids if str(item).strip()]
        if not requested:
            return []

        index_rows = self.load_knowledge_index(limit=100000)
        by_id: dict[str, dict[str, Any]] = {}
        for row in index_rows:
            doc_id = str(row.get("doc_id", "")).strip()
            if doc_id:
                by_id[doc_id] = row

        docs: list[dict[str, Any]] = []
        for doc_id in requested:
            path = self.docs_root / f"{doc_id}.md"
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            row = by_id.get(doc_id, {})
            title = str(row.get("title", "")).strip()
            if not title:
                title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), doc_id)
            docs.append(
                {
                    "title": title,
                    "text": text,
                    "quality_score": float(row.get("quality_score", 0.0) or 0.0),
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                }
            )
        return docs

    def create_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        payload = dict(doc) if isinstance(doc, dict) else {}
        doc_id = str(payload.get("doc_id", f"doc_{uuid4().hex[:12]}")).strip() or f"doc_{uuid4().hex[:12]}"
        title = str(payload.get("title", "Untitled")).strip() or "Untitled"
        body = str(payload.get("body") or payload.get("text") or payload.get("content") or "").strip()

        path = self.docs_root / f"{doc_id}.md"
        out = f"# {title}\n\n{body}\n"
        path.write_text(out, encoding="utf-8")

        index_rows = self.load_knowledge_index(limit=100000)
        entry = {
            "doc_id": doc_id,
            "title": title,
            "quality_score": float(payload.get("quality_score", 0.0) or 0.0),
            "confidence": float(payload.get("confidence", 0.0) or 0.0),
        }

        replaced = False
        for idx, row in enumerate(index_rows):
            if str(row.get("doc_id", "")).strip() == doc_id:
                index_rows[idx] = entry
                replaced = True
                break
        if not replaced:
            index_rows.append(entry)

        self.catalog_path.write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
        return {
            "doc_id": doc_id,
            "path": str(path),
            "created": True,
        }
