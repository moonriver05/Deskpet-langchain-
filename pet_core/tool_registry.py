"""Compatibility layer over the standardized local skill registry."""

from pet_core.skill_registry import (
    AGENT_SKILLS,
    export_mcp_tools,
    export_openai_tools,
    export_router_tools,
    find_skill,
    format_skill_catalog_for_prompt,
    list_skills,
)


AGENT_TOOL_REGISTRY = tuple(skill.as_capability_dict() for skill in AGENT_SKILLS)


def format_capability_registry_for_prompt():
    return format_skill_catalog_for_prompt()
