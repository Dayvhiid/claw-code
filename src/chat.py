import sys
from .ollama_client import OllamaClient
from .real_tools import RealTools
from .execution_registry import build_execution_registry

class AssistantChat:
    def __init__(self, model: str = "qwen3.5:latest"):
        self.client = OllamaClient(model=model)
        self.registry = build_execution_registry()
        self.messages = [
            {"role": "system", "content": "You are Ore, a helpful local coding assistant. "
                                         "You are efficient, friendly, and always ask for permission before running commands. "
                                         "Maintain your personality as Ore at all times."}
        ]
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "BashTool",
                    "description": "Execute a shell command in the local terminal",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The command to run"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "FileReadTool",
                    "description": "Read the contents of a local file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "The path to the file"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "FileWriteTool",
                    "description": "Write or overwrite a local file with new content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "The path to the file"},
                            "content": {"type": "string", "description": "The content to write"}
                        },
                        "required": ["path", "content"]
                    }
                }
            }
        ]

    def run(self):
        print("--- Ore Code Assistant (Local Ollama) ---")
        print("Type 'exit' or 'quit' to stop.")
        
        while True:
            try:
                user_input = input("\nUser: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break

                self.messages.append({"role": "user", "content": user_input})
                self._process_turn()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

    def _process_turn(self):
        max_turns = 5
        for _ in range(max_turns):
            # We use non-streaming to check for tool calls first
            # (as parsing tools from stream is more complex with basic urllib)
            response = self.client.chat(self.messages, tools=self.tools_schema)
            
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    func = tool_call.get("function", {})
                    name = func.get("name")
                    args = func.get("arguments", {})

                    print(f"[TOOL CALL] {name}({args})")
                    
                    # Execute via registry
                    tool = self.registry.tool(name)
                    if not tool:
                        result_msg = f"Error: Tool {name} not found."
                    else:
                        # Specialized handling for arguments
                        if name == "FileWriteTool":
                            result_msg = tool.tool_impl.file_write(args.get("path"), args.get("content")).message
                        elif name == "BashTool":
                            result_msg = tool.tool_impl.bash(args.get("command")).message
                        elif name == "FileReadTool":
                            res = tool.tool_impl.file_read(args.get("path"))
                            result_msg = res.output if res.handled else res.message
                        else:
                            result_msg = "Unknown tool."

                    print(f"[TOOL RESULT] {result_msg}")
                    self.messages.append({
                        "role": "tool",
                        "content": result_msg,
                        "tool_call_id": tool_call.get("id")
                    })
                continue # Run another turn after tool executions

            # If no tool calls, perform a streaming chat for the final response
            if response.text:
                print("\nAssistant: ", end="")
                full_text = ""
                for token in self.client.chat_stream(self.messages):
                    print(token, end="")
                    sys.stdout.flush()
                    full_text += token
                print() # New line after stream
                self.messages.append({"role": "assistant", "content": full_text})
            
            break

if __name__ == "__main__":
    chat = AssistantChat()
    chat.run()
