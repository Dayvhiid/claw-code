import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ToolResult:
    message: str
    handled: bool
    output: Optional[str] = None

class RealTools:
    def __init__(self, trust_mode: bool = False):
        self.trust_mode = trust_mode

    def _prompt_permission(self, action: str) -> bool:
        if self.trust_mode:
            return True
        print(f"\n[PERMISSION] The assistant wants to: {action}")
        choice = input("Allow this action? [y/N]: ").strip().lower()
        return choice == 'y'

    def bash(self, command: str) -> ToolResult:
        if not self._prompt_permission(f"Run shell command: `{command}`"):
            return ToolResult(message="Action denied by user.", handled=False)
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            return ToolResult(message=f"Command executed (exit code {result.returncode})", handled=True, output=output)
        except Exception as e:
            return ToolResult(message=f"Error executing command: {e}", handled=False)

    def file_read(self, path: str) -> ToolResult:
        try:
            full_path = Path(path).resolve()
            content = full_path.read_text(encoding="utf-8")
            return ToolResult(message=f"Read {len(content)} characters from {path}", handled=True, output=content)
        except Exception as e:
            return ToolResult(message=f"Error reading file: {e}", handled=False)

    def file_write(self, path: str, content: str) -> ToolResult:
        if not self._prompt_permission(f"Write to file: {path}"):
            return ToolResult(message="Action denied by user.", handled=False)
            
        try:
            full_path = Path(path).resolve()
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            return ToolResult(message=f"Successfully wrote to {path}", handled=True)
        except Exception as e:
            return ToolResult(message=f"Error writing file: {e}", handled=False)

if __name__ == "__main__":
    # Test
    tools = RealTools()
    print(tools.file_read("README.md").message)
