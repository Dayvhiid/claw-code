from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agent_tools import ToolOutcome, WorkspaceTools
from .ollama_client import OllamaClient


@dataclass(frozen=True)
class AgentTurn:
    session_id: str
    reply: str
    stop_reason: str
    tool_events: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            'session_id': self.session_id,
            'reply': self.reply,
            'stop_reason': self.stop_reason,
            'tool_events': list(self.tool_events),
        }


@dataclass
class AgentSession:
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)


class LocalAgentService:
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        model: str = 'qwen2.5-coder:latest',
        ollama_url: str = 'http://localhost:11434',
        trust_commands: bool = True,
        max_turns: int = 6,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.client = OllamaClient(base_url=ollama_url, model=model)
        self.tools = WorkspaceTools(self.workspace_root, trust_commands=trust_commands)
        self.max_turns = max_turns
        self.sessions: dict[str, AgentSession] = {}
        self.system_prompt = (
            'You are Ore, a local coding agent inside a VS Code fork. '
            'Be concise, use the available tools when needed, and prefer workspace-relative paths. '
            'Use read_file before editing, search_code to locate symbols, apply_patch for diff-based edits, '
            'write_file for generated files, and run_command when verification is useful.'
        )

    def _tool_schema(self) -> list[dict[str, Any]]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'read_file',
                    'description': 'Read a file from the workspace.',
                    'parameters': {
                        'type': 'object',
                        'properties': {'path': {'type': 'string'}},
                        'required': ['path'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'write_file',
                    'description': 'Write a file in the workspace.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'path': {'type': 'string'},
                            'content': {'type': 'string'},
                        },
                        'required': ['path', 'content'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'search_code',
                    'description': 'Search the workspace for a string query.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string'},
                            'glob_pattern': {'type': 'string'},
                        },
                        'required': ['query'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'list_files',
                    'description': 'List files under a workspace path.',
                    'parameters': {
                        'type': 'object',
                        'properties': {'path': {'type': 'string'}},
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'apply_patch',
                    'description': 'Apply a unified diff patch to the workspace using git apply.',
                    'parameters': {
                        'type': 'object',
                        'properties': {'patch': {'type': 'string'}},
                        'required': ['patch'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'run_command',
                    'description': 'Run a shell command in the workspace.',
                    'parameters': {
                        'type': 'object',
                        'properties': {'command': {'type': 'string'}},
                        'required': ['command'],
                    },
                },
            },
        ]

    def _get_session(self, session_id: str | None) -> AgentSession:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        new_session = AgentSession(session_id=session_id or uuid4().hex)
        new_session.messages.append({'role': 'system', 'content': self.system_prompt})
        self.sessions[new_session.session_id] = new_session
        return new_session

    def _tool_result_message(self, outcome: ToolOutcome) -> str:
        return json.dumps(outcome.as_dict(), ensure_ascii=False)

    @staticmethod
    def _normalize_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {'value': arguments}
        return {}

    def _augment_prompt(self, prompt: str, active_file: str | None, selection: str | None) -> str:
        sections = [prompt]
        if active_file:
            sections.append(f'Active file: {active_file}')
            try:
                file_result = self.tools.read_file(active_file)
                if file_result.ok and file_result.output:
                    sections.append(f'Active file contents:\n{file_result.output[:12000]}')
            except Exception:
                pass
        if selection:
            sections.append(f'Selection:\n{selection}')
        return '\n\n'.join(section for section in sections if section)

    async def handle_prompt(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        active_file: str | None = None,
        selection: str | None = None,
    ) -> AgentTurn:
        session = self._get_session(session_id)
        session.messages.append({'role': 'user', 'content': self._augment_prompt(prompt, active_file, selection)})
        tool_events: list[dict[str, Any]] = []

        for _ in range(self.max_turns):
            response = await asyncio.to_thread(self.client.chat, session.messages, self._tool_schema())
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    function = tool_call.get('function', {})
                    name = function.get('name', '')
                    arguments = self._normalize_arguments(function.get('arguments', {}))
                    outcome = await asyncio.to_thread(self.tools.execute, name, arguments)
                    tool_events.append({
                        'tool_call_id': tool_call.get('id'),
                        'name': name,
                        'arguments': arguments,
                        'result': outcome.as_dict(),
                    })
                    session.messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.get('id'),
                        'content': self._tool_result_message(outcome),
                    })
                continue

            reply = response.text.strip()
            session.messages.append({'role': 'assistant', 'content': reply})
            return AgentTurn(session_id=session.session_id, reply=reply, stop_reason='completed', tool_events=tuple(tool_events))

        return AgentTurn(
            session_id=session.session_id,
            reply='Agent stopped after reaching the maximum tool loop depth.',
            stop_reason='max_turns_reached',
            tool_events=tuple(tool_events),
        )
