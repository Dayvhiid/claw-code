from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from .commands import PORTED_COMMANDS, execute_command
from .tools import PORTED_TOOLS, execute_tool
from .real_tools import RealTools

@dataclass(frozen=True)
class FunctionalTool:
    name: str
    source_hint: str
    tool_impl: RealTools

    def execute(self, payload: str) -> str:
        # Simple string-based routing for now
        if self.name == "BashTool":
            return self.tool_impl.bash(payload).message
        elif self.name == "FileReadTool":
            return self.tool_impl.file_read(payload).message
        elif self.name == "FileWriteTool":
            # For FileWrite, we need to parse content. For now, we assume payload is pure content or a simple format.
            return self.tool_impl.file_write(payload, content="").message
        return execute_tool(self.name, payload).message

@dataclass(frozen=True)
class MirroredCommand:
    name: str
    source_hint: str

    def execute(self, prompt: str) -> str:
        return execute_command(self.name, prompt).message

@dataclass(frozen=True)
class MirroredTool:
    name: str
    source_hint: str

    def execute(self, payload: str) -> str:
        return execute_tool(self.name, payload).message

@dataclass(frozen=True)
class ExecutionRegistry:
    commands: tuple[MirroredCommand, ...]
    tools: tuple[MirroredTool | FunctionalTool, ...]

    def command(self, name: str) -> MirroredCommand | None:
        lowered = name.lower()
        for command in self.commands:
            if command.name.lower() == lowered:
                return command
        return None

    def tool(self, name: str) -> MirroredTool | FunctionalTool | None:
        lowered = name.lower()
        for tool in self.tools:
            if tool.name.lower() == lowered:
                return tool
        return None

def build_execution_registry() -> ExecutionRegistry:
    real_tools = RealTools()
    tools = []
    for module in PORTED_TOOLS:
        if module.name in {"BashTool", "FileReadTool", "FileWriteTool"}:
            tools.append(FunctionalTool(module.name, module.source_hint, real_tools))
        else:
            tools.append(MirroredTool(module.name, module.source_hint))
            
    return ExecutionRegistry(
        commands=tuple(MirroredCommand(module.name, module.source_hint) for module in PORTED_COMMANDS),
        tools=tuple(tools),
    )
