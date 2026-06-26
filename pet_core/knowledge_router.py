"""Routing helpers for optional knowledge-base retrieval.

The knowledge base is a tool, not a default memory channel. It should only run
when the user explicitly asks to search imported documents or local materials.
"""


EXPLICIT_KB_TRIGGERS = (
    "知识库",
    "资料库",
    "文档库",
    "本地资料",
    "本地文档",
    "导入的资料",
    "导入的文档",
    "导入过的",
    "查资料",
    "查一下资料",
    "检索资料",
    "检索文档",
    "从资料里",
    "从文档里",
    "根据资料",
    "根据文档",
    "引用资料",
    "引用文档",
)


def should_search_knowledge_base(user_text, has_attachment=False):
    """Return a strict tool-routing decision for knowledge-base search.

    This intentionally avoids fuzzy intent guessing. Food, mood, study progress
    and ordinary chat should not trigger document retrieval just because vector
    search can return something weakly related.
    """
    text = str(user_text or "").strip()
    normalized = "".join(text.split()).lower()
    matched = ""
    for trigger in EXPLICIT_KB_TRIGGERS:
        if "".join(trigger.split()).lower() in normalized:
            matched = trigger
            break

    if matched:
        return {
            "used": True,
            "mode": "explicit_tool_call",
            "reason": f"用户显式提到资料/知识库触发词：{matched}",
            "query": text,
        }

    if has_attachment:
        return {
            "used": False,
            "mode": "direct_attachment",
            "reason": "本轮有附件时直接阅读附件内容，不额外检索知识库。",
            "query": text,
        }

    return {
        "used": False,
        "mode": "not_requested",
        "reason": "用户没有明确要求查知识库或导入文档。",
        "query": text,
    }
