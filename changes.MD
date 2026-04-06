# Changes & Progress: Transformation to "Ore" Assistant

This file tracks the evolution of the `claw-code` Python port from a mirrored architecture study into a functional local coding assistant named **Ore**.

## Initial State
The project started as a "clean-room" Python rewrite of Claude Code's harness, containing only metadata and empty shims (mirrored tools/commands) without any actual LLM logic or functional tool execution.

## Major Milestones

### 1. Ollama LLM Integration
- **Ollama Client**: Created `src/ollama_client.py` to communicate with a local Ollama server using only Python standard libraries.
- **Model Support**: Configured to use `qwen3.5:latest` by default (matching local registry paths).
- **Streaming Support**: Implemented real-time token streaming to make the assistant feel faster and more responsive.

### 2. Functional Tool System
- **Real Execution**: Created `src/real_tools.py` which transitions the assistant from "pretending" to "doing."
    - **BashTool**: Runs actual shell commands on the local system.
    - **FileReadTool**: Reads local files into the assistant's context.
    - **FileWriteTool**: Writes or overwrites local files.

### 3. Safety & Permission Gates
- **Human-in-the-Loop**: As requested, a mandatory permission prompt was implemented for all destructive or external actions.
- **[Y/n] Prompt**: Before running any bash command or modifying any file, Ore will stop and ask for your permission in the terminal.

### 4. Persona: "Ore"
- **Identity**: The assistant was renamed from "Claw Code" to **Ore**.
- **Interface**: Created `src/chat.py`, a dedicated interactive CLI for chatting with Ore.
- **Hybrid Streaming**: Implemented a logic that uses non-streaming calls for reliable tool usage and switches to real-time streaming for the final conversational response.

---

## How to use Ore
Run the following command in your terminal to start a session:

```bash
python -m src.chat
```
