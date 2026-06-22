"""NoteTool — file-based note persistence, compatible with hello_agents Tool protocol."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List

from hello_agents.tools.base import Tool, ToolParameter


class NoteTool(Tool):
    """Create and update markdown notes in a workspace directory.

    Inherits from hello_agents Tool base to provide name / description /
    expandable for ToolRegistry compatibility, while run() returns plain
    strings matching the calling convention in agent.py.
    """

    def __init__(self, workspace: str = "./notes") -> None:
        super().__init__(
            name="note",
            description="创建和更新研究笔记，每个笔记以 Markdown 文件存储",
        )
        self._workspace = Path(workspace).resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)

    # -- Tool ABC ---------------------------------------------------------
    def run(self, parameters: Dict[str, Any]) -> str:
        action = parameters.get("action", "create")
        if action == "create":
            return self._create(parameters)
        elif action == "update":
            return self._update(parameters)
        return f"❌ Unknown action: {action}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="action", type="string", description="操作类型: create 或 update"),
            ToolParameter(name="title", type="string", description="笔记标题"),
            ToolParameter(name="content", type="string", description="笔记内容 (Markdown)"),
            ToolParameter(name="note_id", type="string", description="笔记ID (update 时必需)", required=False),
            ToolParameter(name="note_type", type="string", description="笔记类型", required=False),
            ToolParameter(name="tags", type="array", description="标签列表", required=False),
        ]

    # -- actions ----------------------------------------------------------
    def _create(self, params: Dict[str, Any]) -> str:
        title = str(params.get("title", "untitled"))
        content = str(params.get("content", ""))
        note_id = str(uuid.uuid4())[:8]

        file_path = self._workspace / f"{note_id}.md"
        file_path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")

        return f"✅ 笔记已创建\nID: {note_id}\n路径: {file_path}"

    def _update(self, params: Dict[str, Any]) -> str:
        note_id = str(params.get("note_id", ""))
        if not note_id:
            return "❌ update 操作需要提供 note_id"

        file_path = self._workspace / f"{note_id}.md"
        if not file_path.exists():
            return f"❌ 笔记不存在: {note_id}"

        title = str(params.get("title", "untitled"))
        content = str(params.get("content", ""))
        file_path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")

        return f"✅ 笔记已更新\nID: {note_id}\n路径: {file_path}"
