# Copyright © 2026 Apple Inc.

"""Helpers for Qwen2.5-Coder raw JSON tool-call fallback parsing."""

import json
import re
import uuid
from typing import Any, Optional


# Qwen2.5-Coder can leak chat-template tokens into generated text.
_SPECIAL_CHAT_TOKEN_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>")


def strip_special_chat_tokens(text: str) -> str:
    """Remove chat-template tokens from generated text."""
    return _SPECIAL_CHAT_TOKEN_RE.sub("", text or "").strip()


def _clean_raw_json_text(text: str) -> str:
    """Normalize fenced or tagged JSON-like assistant text."""
    text = strip_special_chat_tokens(text)

    # Tool tags can appear around JSON in some generations.
    text = re.sub(r"</?tools>", "", text).strip()

    # Markdown fences can appear around JSON in some generations.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text).strip()

    # Extra text can leak before or after the JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    # Qwen2.5 can double-wrap tool JSON as {{ ... }}.
    if text.startswith("{{") and text.endswith("}}"):
        text = text[1:-1].strip()

    return text


def try_parse_raw_tool_call(text: str, tools: Optional[Any] = None):
    """Parse a Qwen2.5 raw JSON tool call if the response contains one."""
    del tools  # Qwen2.5 raw JSON already contains argument values.

    text = _clean_raw_json_text(text)
    if not text:
        return None

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None

    # Common shape: {"name": "glob", "arguments": {"pattern": "**/*"}}.
    if isinstance(obj.get("name"), str):
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            args = {"input": args}
        return {
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "name": obj["name"],
            "arguments": args,
        }

    # Alternate shape: {"function": "read", "filePath": "README.md"}.
    if isinstance(obj.get("function"), str):
        return {
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "name": obj["function"],
            "arguments": {k: v for k, v in obj.items() if k != "function"},
        }

    return None


def to_openai_tool_call(parsed_tool):
    """Convert a parsed Qwen2.5 tool call to OpenAI tool_call shape."""
    return {
        "id": parsed_tool["id"],
        "type": "function",
        "function": {
            "name": parsed_tool["name"],
            "arguments": json.dumps(
                parsed_tool["arguments"],
                separators=(",", ":"),
            ),
        },
    }


def add_stream_tool_call_indexes(tool_calls):
    """Add OpenAI's required stream index to formatted tool calls."""
    if not tool_calls:
        return []

    formatted = []
    for index, tool_call in enumerate(tool_calls):
        if isinstance(tool_call, dict):
            item = dict(tool_call)
            item.setdefault("index", index)
            formatted.append(item)
        else:
            formatted.append(tool_call)

    return formatted
