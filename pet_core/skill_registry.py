"""Standardized local skill registry for Yuzu's desktop agent.

The internal format is intentionally close to common tool schemas:
- MCP tools use ``name / description / inputSchema``.
- OpenAI function tools use ``type=function / function.parameters``.
- Claude/OpenClaw-style skills can be generated from the same description.

Concrete executors still live in the app.  This module only declares what the
agent is allowed to do and how arguments should be shaped.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class SkillSpec:
    name: str
    title: str
    description: str
    input_schema: JsonDict
    risk: str = "low"
    requires_confirmation: bool = False
    autonomous_allowed: bool = False
    executor: str = ""
    legacy_names: List[str] = field(default_factory=list)
    boundary: str = ""
    examples: List[str] = field(default_factory=list)

    def to_mcp_tool(self, *, legacy_name: Optional[str] = None) -> JsonDict:
        return {
            "name": legacy_name or self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": {
                "title": self.title,
                "risk": self.risk,
                "requires_confirmation": self.requires_confirmation,
                "autonomous_allowed": self.autonomous_allowed,
                "executor": self.executor,
                "canonical_name": self.name,
                "boundary": self.boundary,
            },
        }

    def to_openai_tool(self) -> JsonDict:
        return {
            "type": "function",
            "function": {
                "name": self.name.replace(".", "_"),
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def as_capability_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "label": self.title,
            "risk": self.risk,
            "requires_confirmation": self.requires_confirmation,
            "autonomous_allowed": self.autonomous_allowed,
            "inputs": list((self.input_schema.get("properties") or {}).keys()),
            "can": self.description,
            "cannot": self.boundary,
            "boundary": self.boundary,
            "executor": self.executor,
            "legacy_names": list(self.legacy_names),
        }


TODO_ADD_SCHEMA: JsonDict = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "待办核心内容，30 字以内"},
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "紧急程度；默认 medium",
        },
        "category": {
            "type": "string",
            "enum": ["work", "study", "life", "homework", "other"],
            "description": "待办分类；默认 other",
        },
        "due_date": {
            "type": "string",
            "description": "截止时间，YYYY-MM-DD 或 YYYY-MM-DD HH:MM；没有就留空",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "可选标签",
        },
    },
    "required": ["text"],
    "additionalProperties": False,
}


TIMER_START_SCHEMA: JsonDict = {
    "type": "object",
    "properties": {
        "seconds": {
            "type": "integer",
            "minimum": 5,
            "maximum": 86400,
            "description": "倒计时秒数",
        },
        "label": {
            "type": "string",
            "description": "计时器名称，例如 专注、背单词、短休息",
        },
    },
    "required": ["seconds"],
    "additionalProperties": False,
}


AGENT_SKILLS: List[SkillSpec] = [
    SkillSpec(
        name="chat.reply",
        title="聊天回复",
        description="在聊天窗口、气泡和语音中回复用户。",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        autonomous_allowed=True,
        executor="chat.reply",
        boundary="只能表达程序回复，不能把未发生的现实动作说成已经完成。",
    ),
    SkillSpec(
        name="todo.add",
        title="添加待办",
        description=(
            "在用户明确要求记录、安排、提醒，或用户确认上一轮待办建议时，"
            "向本地待办清单写入一条待办。"
        ),
        input_schema=TODO_ADD_SCHEMA,
        autonomous_allowed=False,
        executor="todo_store.add",
        legacy_names=["add_todo"],
        boundary="不能把愿望、闲聊、感叹自动当成待办；只管理本地待办，不代表现实执行任务。",
        examples=["用户：明天提醒我复习数据库", "用户：好，帮我记一下"],
    ),
    SkillSpec(
        name="timer.start",
        title="启动专注计时",
        description=(
            "在用户明确要求启动/设置/帮忙开倒计时，或确认上一轮计时建议时，"
            "启动本地专注计时器。"
        ),
        input_schema=TIMER_START_SCHEMA,
        autonomous_allowed=False,
        executor="Pet.start_focus_timer",
        legacy_names=["start_focus_timer"],
        boundary="不能在程序未运行时计时，也不能替代系统闹钟；用户只是提到学习/专注不等于立即启动。",
        examples=["用户：帮我开 25 分钟专注", "用户：开吧"],
    ),
    SkillSpec(
        name="rss.recommend",
        title="RSS 外部内容推荐",
        description="从用户配置的 RSS 内容池里低频推荐一个网页、视频或图片原链接。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "trigger": {"type": "string", "enum": ["startup", "idle", "user_request"]},
            },
            "required": ["trigger"],
            "additionalProperties": False,
        },
        autonomous_allowed=True,
        executor="rss_content_runtime.suggest",
        boundary="只能基于已抓取的标题、简介、链接和缓存内容推荐；不能声称完整观看或全网搜索。",
    ),
    SkillSpec(
        name="browser.open_url",
        title="打开链接",
        description="在用户点击或明确要求时，用系统浏览器打开链接，并记录点击作为隐式反馈。",
        input_schema={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        },
        autonomous_allowed=False,
        executor="QDesktopServices.openUrl",
        boundary="不能私自打开未知链接；推荐内容应优先让用户点击确认。",
    ),
    SkillSpec(
        name="system.sense_state",
        title="粗粒度桌面状态",
        description="读取前台应用类别、空闲时间和时间段，用于判断是否打扰或推荐。",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        autonomous_allowed=True,
        executor="collect_system_state",
        boundary="不读取窗口正文、网页内容、聊天内容、键盘输入或截图。",
    ),
]


def list_skills(names: Optional[Iterable[str]] = None) -> List[SkillSpec]:
    if names is None:
        return list(AGENT_SKILLS)
    wanted = {str(name) for name in names}
    return [skill for skill in AGENT_SKILLS if skill.name in wanted]


def find_skill(name: str) -> Optional[SkillSpec]:
    raw = str(name or "")
    for skill in AGENT_SKILLS:
        if raw == skill.name or raw in skill.legacy_names:
            return skill
    return None


def export_mcp_tools(names: Optional[Iterable[str]] = None, *, legacy: bool = False) -> List[JsonDict]:
    tools = []
    for skill in list_skills(names):
        if legacy and skill.legacy_names:
            tools.append(skill.to_mcp_tool(legacy_name=skill.legacy_names[0]))
        else:
            tools.append(skill.to_mcp_tool())
    return tools


def export_openai_tools(names: Optional[Iterable[str]] = None) -> List[JsonDict]:
    return [skill.to_openai_tool() for skill in list_skills(names)]


def format_skill_catalog_for_prompt(names: Optional[Iterable[str]] = None) -> str:
    lines = [
        "【本地 Skill 注册表】",
        "Skill 决定你实际能做什么；角色设定只决定说话方式。只能承诺下列程序能力。",
        "",
    ]
    for skill in list_skills(names):
        props = ", ".join((skill.input_schema.get("properties") or {}).keys()) or "无"
        lines.append(
            f"- {skill.name}｜{skill.title}｜risk={skill.risk}｜"
            f"confirm={skill.requires_confirmation}｜auto={skill.autonomous_allowed}｜"
            f"inputs={props}｜{skill.description} 边界：{skill.boundary}"
        )
    return "\n".join(lines)


def export_router_tools(names: Optional[Iterable[str]] = None) -> List[JsonDict]:
    """Legacy-compatible schemas for the current in-process tool router."""
    return export_mcp_tools(names or ["todo.add", "timer.start"], legacy=True)


def export_skill_manifest(names: Optional[Iterable[str]] = None) -> JsonDict:
    return {
        "schema_version": "yuzu.skill_registry.v1",
        "formats": ["mcp_tool", "openai_function_tool", "markdown_skill_doc"],
        "skills": [skill.as_capability_dict() | {"input_schema": skill.input_schema} for skill in list_skills(names)],
    }


def export_skill_markdown(names: Optional[Iterable[str]] = None) -> str:
    chunks = [
        "# Yuzu Local Skills",
        "",
        "These are local desktop capabilities. A role prompt may change tone, but it must not expand these capabilities.",
        "",
    ]
    for skill in list_skills(names):
        props = skill.input_schema.get("properties") or {}
        chunks.extend(
            [
                f"## {skill.name}",
                "",
                f"- Title: {skill.title}",
                f"- Risk: {skill.risk}",
                f"- Requires confirmation: {skill.requires_confirmation}",
                f"- Autonomous allowed: {skill.autonomous_allowed}",
                f"- Executor: `{skill.executor}`",
                f"- Description: {skill.description}",
                f"- Boundary: {skill.boundary}",
                f"- Inputs: {', '.join(props.keys()) or 'none'}",
                "",
            ]
        )
    return "\n".join(chunks).rstrip() + "\n"


def _main() -> None:
    parser = argparse.ArgumentParser(description="Export Yuzu local skill/tool definitions.")
    parser.add_argument(
        "--format",
        choices=["manifest", "mcp", "openai", "router", "markdown"],
        default="manifest",
    )
    parser.add_argument("--output", default="", help="Optional output path. Stdout is used when omitted.")
    args = parser.parse_args()

    if args.format == "manifest":
        text = json.dumps(export_skill_manifest(), ensure_ascii=False, indent=2)
    elif args.format == "mcp":
        text = json.dumps(export_mcp_tools(), ensure_ascii=False, indent=2)
    elif args.format == "openai":
        text = json.dumps(export_openai_tools(), ensure_ascii=False, indent=2)
    elif args.format == "router":
        text = json.dumps(export_router_tools(), ensure_ascii=False, indent=2)
    else:
        text = export_skill_markdown()

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    _main()
