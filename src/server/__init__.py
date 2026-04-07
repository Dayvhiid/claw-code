"""Python package placeholder for the archived `server` subsystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / 'reference_data' / 'subsystems' / 'server.json'
_SNAPSHOT = json.loads(SNAPSHOT_PATH.read_text())

ARCHIVE_NAME = _SNAPSHOT['archive_name']
MODULE_COUNT = _SNAPSHOT['module_count']
SAMPLE_FILES = tuple(_SNAPSHOT['sample_files'])
PORTING_NOTE = f"Python placeholder package for '{ARCHIVE_NAME}' with {MODULE_COUNT} archived module references."

if TYPE_CHECKING:
	from .app import ServerConfig, create_app, run_server


def __getattr__(name: str) -> Any:
	if name in {'ServerConfig', 'create_app', 'run_server'}:
		from .app import ServerConfig, create_app, run_server

		return {
			'ServerConfig': ServerConfig,
			'create_app': create_app,
			'run_server': run_server,
		}[name]
	raise AttributeError(name)

__all__ = ['ARCHIVE_NAME', 'MODULE_COUNT', 'PORTING_NOTE', 'SAMPLE_FILES', 'ServerConfig', 'create_app', 'run_server']
