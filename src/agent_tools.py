from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolOutcome:
    name: str
    ok: bool
    message: str
    output: str = ''
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'ok': self.ok,
            'message': self.message,
            'output': self.output,
            'details': self.details,
        }


class WorkspaceTools:
    def __init__(self, workspace_root: str | Path, trust_commands: bool = True, command_timeout: int = 60):
        self.workspace_root = Path(workspace_root).resolve()
        self.trust_commands = trust_commands
        self.command_timeout = command_timeout

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve()
        if resolved != self.workspace_root and self.workspace_root not in resolved.parents:
            raise ValueError(f'Path escapes workspace root: {path}')
        return resolved

    def _relative_path(self, path: Path) -> str:
        return str(path.relative_to(self.workspace_root)).replace('\\', '/')

    def read_file(self, path: str) -> ToolOutcome:
        try:
            resolved = self._resolve_path(path)
            content = resolved.read_text(encoding='utf-8')
            return ToolOutcome(
                name='read_file',
                ok=True,
                message=f'Read {len(content)} characters from {self._relative_path(resolved)}',
                output=content,
                details={'path': self._relative_path(resolved)},
            )
        except Exception as exc:
            return ToolOutcome(name='read_file', ok=False, message=str(exc))

    def write_file(self, path: str, content: str) -> ToolOutcome:
        try:
            resolved = self._resolve_path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding='utf-8')
            return ToolOutcome(
                name='write_file',
                ok=True,
                message=f'Wrote {len(content)} characters to {self._relative_path(resolved)}',
                details={'path': self._relative_path(resolved)},
            )
        except Exception as exc:
            return ToolOutcome(name='write_file', ok=False, message=str(exc))

    def list_files(self, path: str = '.') -> ToolOutcome:
        try:
            resolved = self._resolve_path(path)
            if not resolved.exists():
                return ToolOutcome(name='list_files', ok=False, message=f'Path not found: {path}')
            if resolved.is_file():
                return ToolOutcome(name='list_files', ok=True, message='Listed 1 item', output=self._relative_path(resolved), details={'count': 1})
            entries = sorted(self._relative_path(item) for item in resolved.rglob('*') if item.is_file())
            output = '\n'.join(entries)
            return ToolOutcome(name='list_files', ok=True, message=f'Listed {len(entries)} files under {self._relative_path(resolved)}', output=output, details={'count': len(entries)})
        except Exception as exc:
            return ToolOutcome(name='list_files', ok=False, message=str(exc))

    def search_code(self, query: str, glob_pattern: str | None = None) -> ToolOutcome:
        rg = shutil.which('rg')
        if rg:
            command = [rg, '--line-number', '--hidden', '--glob', glob_pattern or '!node_modules', query, str(self.workspace_root)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, cwd=self.workspace_root, timeout=self.command_timeout)
                output = result.stdout.strip() or result.stderr.strip()
                return ToolOutcome(
                    name='search_code',
                    ok=result.returncode in (0, 1),
                    message=f'rg exited with code {result.returncode}',
                    output=output,
                    details={'returncode': result.returncode},
                )
            except Exception as exc:
                return ToolOutcome(name='search_code', ok=False, message=str(exc))

        matches: list[str] = []
        try:
            for file_path in self.workspace_root.rglob('*'):
                if not file_path.is_file():
                    continue
                try:
                    text = file_path.read_text(encoding='utf-8')
                except Exception:
                    continue
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if query.lower() in line.lower():
                        matches.append(f'{self._relative_path(file_path)}:{line_number}:{line.strip()}')
                        if len(matches) >= 200:
                            break
                if len(matches) >= 200:
                    break
            return ToolOutcome(name='search_code', ok=True, message=f'Found {len(matches)} matches', output='\n'.join(matches), details={'count': len(matches)})
        except Exception as exc:
            return ToolOutcome(name='search_code', ok=False, message=str(exc))

    def run_command(self, command: str) -> ToolOutcome:
        if not self.trust_commands:
            return ToolOutcome(name='run_command', ok=False, message='Command execution is disabled for this server session.')
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
            )
            output = '\n'.join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
            return ToolOutcome(
                name='run_command',
                ok=True,
                message=f'Command exited with code {result.returncode}',
                output=output,
                details={'returncode': result.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolOutcome(name='run_command', ok=False, message=f'Command timed out after {self.command_timeout} seconds')
        except Exception as exc:
            return ToolOutcome(name='run_command', ok=False, message=str(exc))

    def apply_patch(self, patch_text: str) -> ToolOutcome:
        if not self.trust_commands:
            return ToolOutcome(name='apply_patch', ok=False, message='Patch application is disabled for this server session.')
        patch_path = None
        try:
            with tempfile.NamedTemporaryFile('w', suffix='.patch', delete=False, encoding='utf-8', newline='\n') as handle:
                handle.write(patch_text)
                patch_path = Path(handle.name)
            result = subprocess.run(
                ['git', 'apply', '--whitespace=nowarn', str(patch_path)],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
            )
            output = '\n'.join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
            if result.returncode != 0:
                return ToolOutcome(name='apply_patch', ok=False, message='git apply failed', output=output, details={'returncode': result.returncode})
            diff = subprocess.run(
                ['git', 'diff', '--name-only'],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
            )
            changed_files = [line.strip() for line in diff.stdout.splitlines() if line.strip()]
            return ToolOutcome(
                name='apply_patch',
                ok=True,
                message='Patch applied successfully',
                output='\n'.join(changed_files),
                details={'changed_files': changed_files},
            )
        except subprocess.TimeoutExpired:
            return ToolOutcome(name='apply_patch', ok=False, message=f'Patch application timed out after {self.command_timeout} seconds')
        except Exception as exc:
            return ToolOutcome(name='apply_patch', ok=False, message=str(exc))
        finally:
            if patch_path is not None:
                patch_path.unlink(missing_ok=True)

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        lowered = name.lower()
        if lowered == 'read_file':
            return self.read_file(str(arguments.get('path', '')))
        if lowered == 'write_file':
            return self.write_file(str(arguments.get('path', '')), str(arguments.get('content', '')))
        if lowered == 'list_files':
            return self.list_files(str(arguments.get('path', '.')))
        if lowered == 'search_code':
            return self.search_code(str(arguments.get('query', '')), arguments.get('glob_pattern'))
        if lowered == 'run_command':
            return self.run_command(str(arguments.get('command', '')))
        if lowered == 'apply_patch':
            return self.apply_patch(str(arguments.get('patch', '')))
        return ToolOutcome(name=name, ok=False, message=f'Unknown tool: {name}')
