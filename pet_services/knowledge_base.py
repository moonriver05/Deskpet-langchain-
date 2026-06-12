"""RAG knowledge-base chunking and retrieval."""

import re
import uuid
from collections import defaultdict

from pet_services.chroma_mcp import (
    CHROMA_COLLECTION_KB,
    chroma_add_documents_sync,
    chroma_get_documents_sync,
    chroma_query_documents_sync,
)


def split_markdown_into_chunks(text, chunk_size=800, overlap=120):
    """Split long text into Markdown-aware chunks with light overlap."""
    if not text or not text.strip():
        return []

    lines = text.replace("\r\n", "\n").split("\n")
    heading_stack = []
    sections = []
    buf = []

    def flush():
        if buf:
            block = "\n".join(buf).strip()
            if block:
                sections.append(([h for _, h in heading_stack], block))
            buf.clear()

    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            buf.append(line)
    flush()

    if not sections:
        sections = [([], text)]

    chunks = []
    overlap = max(0, int(overlap))
    chunk_size = max(200, int(chunk_size))
    hard_limit = int(chunk_size * 1.5)

    for heading_path, block in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", block) if p.strip()]
        if not paragraphs:
            continue
        cur = ""
        for para in paragraphs:
            while len(para) > hard_limit:
                if cur:
                    chunks.append({"text": cur, "heading_path": list(heading_path)})
                    tail = cur[-overlap:] if overlap else ""
                    cur = ""
                    head_piece = para[: chunk_size - len(tail)]
                    chunks.append({"text": tail + head_piece, "heading_path": list(heading_path)})
                    para = para[chunk_size - len(tail):]
                else:
                    chunks.append({"text": para[:chunk_size], "heading_path": list(heading_path)})
                    para = para[chunk_size - overlap:] if overlap else para[chunk_size:]
            if not cur:
                cur = para
            elif len(cur) + 2 + len(para) <= chunk_size:
                cur = cur + "\n\n" + para
            else:
                chunks.append({"text": cur, "heading_path": list(heading_path)})
                tail = cur[-overlap:] if overlap else ""
                cur = (tail + "\n\n" + para) if tail else para
        if cur:
            chunks.append({"text": cur, "heading_path": list(heading_path)})

    return chunks


class KnowledgeBase:
    """Chroma-backed knowledge base with Markdown chunk windows."""

    DEFAULT_CHUNK_SIZE = 800
    DEFAULT_OVERLAP = 120
    DEFAULT_WINDOW = 1

    def add_document(self, text, source="unknown", chunk_size=None, overlap=None):
        if not text or not str(text).strip():
            return None
        chunk_size = int(chunk_size) if chunk_size else self.DEFAULT_CHUNK_SIZE
        overlap = int(overlap) if overlap is not None else self.DEFAULT_OVERLAP

        pieces = split_markdown_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
        if not pieces:
            return None

        doc_id = uuid.uuid4().hex
        total = len(pieces)
        ids = [f"{doc_id}_chunk_{i:04d}" for i in range(total)]
        docs = [p["text"] for p in pieces]
        metadatas = []
        for i, piece in enumerate(pieces):
            meta = {
                "source": str(source),
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": total,
            }
            heading = piece.get("heading_path") or []
            if heading:
                meta["heading_path"] = " > ".join(str(h) for h in heading)
            metadatas.append(meta)

        try:
            chroma_add_documents_sync(CHROMA_COLLECTION_KB, docs, ids, metadatas=metadatas)
            print(f"[KB] 已写入 source={source!r} doc_id={doc_id} chunks={total}")
            return doc_id
        except Exception as e:
            print("写入知识库失败 (Chroma MCP):", e)
            return None

    def _expand_window(self, hits, window):
        wanted = defaultdict(set)
        legacy = []
        for hit in hits:
            if hit["doc_id"] and hit["chunk_index"] is not None:
                for offset in range(-window, window + 1):
                    idx = hit["chunk_index"] + offset
                    if idx >= 0:
                        wanted[hit["doc_id"]].add(idx)
            else:
                legacy.append(hit)

        expanded = {}
        for doc_id, idx_set in wanted.items():
            try:
                got = chroma_get_documents_sync(
                    CHROMA_COLLECTION_KB,
                    where={"$and": [
                        {"doc_id": {"$eq": doc_id}},
                        {"chunk_index": {"$in": sorted(idx_set)}},
                    ]},
                )
            except Exception as e:
                print(f"[KB] 窗口扩展失败 doc_id={doc_id}: {e}")
                continue
            if not got:
                continue
            g_docs = got.get("documents") or []
            g_metas = got.get("metadatas") or []
            for doc, meta in zip(g_docs, g_metas):
                if not doc or not isinstance(meta, dict):
                    continue
                chunk_index = meta.get("chunk_index")
                try:
                    chunk_index = int(chunk_index)
                except (TypeError, ValueError):
                    continue
                expanded[(doc_id, chunk_index)] = {
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "heading": meta.get("heading_path", ""),
                }
        return expanded, legacy

    def search(self, query, keywords=None, top_k=3, window=None):
        if not query or not str(query).strip():
            return []
        if window is None:
            window = self.DEFAULT_WINDOW

        q = str(query).strip()
        if keywords:
            q = f"{q} {' '.join(str(k) for k in keywords if k)}"

        n_results = max(int(top_k), min(40, int(top_k) * 6))
        try:
            data = chroma_query_documents_sync(CHROMA_COLLECTION_KB, [q], n_results=n_results)
        except Exception as e:
            print("知识库检索失败 (Chroma MCP):", e)
            return []
        if not data:
            return []

        docs = (data.get("documents") or [[]])[0]
        metas = (data.get("metadatas") or [[]])[0]
        ids = (data.get("ids") or [[]])[0]

        hits = []
        for i, doc in enumerate(docs):
            if doc is None:
                continue
            meta = metas[i] if i < len(metas) and metas[i] else {}
            if not isinstance(meta, dict):
                meta = {}
            chunk_idx = meta.get("chunk_index")
            try:
                chunk_idx = int(chunk_idx) if chunk_idx is not None else None
            except (TypeError, ValueError):
                chunk_idx = None
            hits.append({
                "id": ids[i] if i < len(ids) else None,
                "doc_id": meta.get("doc_id"),
                "chunk_index": chunk_idx,
                "source": meta.get("source", "unknown"),
                "heading": meta.get("heading_path", ""),
                "text": doc,
            })

        if not hits:
            return []

        anchors = hits[:max(1, int(top_k))]
        expanded, legacy = self._expand_window(anchors, window=window)

        out = []
        seen_doc = set()
        for hit in anchors:
            doc_id = hit["doc_id"]
            if doc_id is None or doc_id in seen_doc:
                continue
            seen_doc.add(doc_id)
            indices = sorted(idx for (d, idx) in expanded.keys() if d == doc_id)
            if not indices:
                continue
            label = hit["source"] or "unknown"
            if hit["heading"]:
                label = f"{label}§{hit['heading']}"
            parts = [expanded[(doc_id, chunk_index)]["text"] for chunk_index in indices]
            joined = "\n…\n".join(parts) if len(parts) > 1 else parts[0]
            out.append(f"[{label}] {joined}")
            if len(out) >= top_k:
                break

        if len(out) < top_k:
            for hit in legacy:
                out.append(f"[{hit['source']}] {hit['text']}")
                if len(out) >= top_k:
                    break

        return out
