from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..agent_service import LocalAgentService


class AgentRequest(BaseModel):
    prompt: str = Field(min_length=1)
    session_id: str | None = None
    active_file: str | None = None
    selection: str | None = None


class FilePathRequest(BaseModel):
    path: str


class WriteFileRequest(BaseModel):
    path: str
    content: str


class SearchRequest(BaseModel):
    query: str
    glob_pattern: str | None = None


class CommandRequest(BaseModel):
    command: str


class PatchRequest(BaseModel):
    patch: str


@dataclass(frozen=True)
class ServerConfig:
    workspace_root: Path
    model: str = 'qwen2.5-coder:latest'
    ollama_url: str = 'http://localhost:11434'
    trust_commands: bool = True


def create_app(config: ServerConfig) -> FastAPI:
    service = LocalAgentService(
        config.workspace_root,
        model=config.model,
        ollama_url=config.ollama_url,
        trust_commands=config.trust_commands,
    )
    app = FastAPI(title='Claw IDE Harness', version='0.1.0')

    @app.get('/health')
    async def health() -> dict[str, Any]:
        return {
            'status': 'ok',
            'workspace_root': str(config.workspace_root),
            'model': config.model,
            'trust_commands': config.trust_commands,
        }

    @app.post('/agent')
    async def agent(request: AgentRequest) -> dict[str, Any]:
        turn = await service.handle_prompt(
            request.prompt,
            session_id=request.session_id,
            active_file=request.active_file,
            selection=request.selection,
        )
        return turn.as_dict()

    @app.post('/read')
    async def read_file(request: FilePathRequest) -> dict[str, Any]:
        outcome = service.tools.read_file(request.path)
        if not outcome.ok:
            raise HTTPException(status_code=400, detail=outcome.message)
        return outcome.as_dict()

    @app.post('/edit')
    async def edit_file(request: WriteFileRequest) -> dict[str, Any]:
        outcome = service.tools.write_file(request.path, request.content)
        if not outcome.ok:
            raise HTTPException(status_code=400, detail=outcome.message)
        return outcome.as_dict()

    @app.post('/search')
    async def search(request: SearchRequest) -> dict[str, Any]:
        outcome = service.tools.search_code(request.query, request.glob_pattern)
        if not outcome.ok:
            raise HTTPException(status_code=400, detail=outcome.message)
        return outcome.as_dict()

    @app.post('/run')
    async def run_command(request: CommandRequest) -> dict[str, Any]:
        outcome = service.tools.run_command(request.command)
        if not outcome.ok:
            raise HTTPException(status_code=400, detail=outcome.message)
        return outcome.as_dict()

    @app.post('/apply-patch')
    async def apply_patch(request: PatchRequest) -> dict[str, Any]:
        outcome = service.tools.apply_patch(request.patch)
        if not outcome.ok:
            raise HTTPException(status_code=400, detail=outcome.message)
        return outcome.as_dict()

    return app


def run_server(
    *,
    workspace_root: str | Path,
    host: str = '127.0.0.1',
    port: int = 8000,
    model: str = 'qwen2.5-coder:latest',
    ollama_url: str = 'http://localhost:11434',
    trust_commands: bool = True,
) -> None:
    config = ServerConfig(workspace_root=Path(workspace_root).resolve(), model=model, ollama_url=ollama_url, trust_commands=trust_commands)
    app = create_app(config)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError('uvicorn is required to run the harness server') from exc

    uvicorn.run(app, host=host, port=port, log_level='info')
