import json
import sys
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from rag.query import retrieve, build_prompt, ask_llm
from mcp.client import MCPClient
from mcp.decision_heuristic import decide_tool_usage
from telemetry import context, session_memory

class CompanyKBAssistant:
    """Company Knowledge Base Assistant combining RAG and MCP."""

    def __init__(self):
        self.mcp = None
        self.task_id = context.start_task()
        self._init_mcp()

    def _init_mcp(self):
        """Initialize MCP client."""
        try:
            import sys
            from pathlib import Path
            python_cmd = sys.executable
            # Get absolute path to MCP server
            mcp_path = Path(__file__).parent / "mcp" / "server.py"
            self.mcp = MCPClient([python_cmd, str(mcp_path)])
        except Exception as e:
            print(f"Warning: Could not initialize MCP client: {e}")
            self.mcp = None

    def _extract_mcp_text(self, mcp_result):
        """MCP tool results are usually plain strings, but read_document
        returns a JSON-encoded {"content", "truncated", ...} envelope on
        success (its error strings, and both other tools' results, stay
        plain). Returns the text that should actually be embedded in the
        prompt — never raw JSON envelope syntax."""
        if not mcp_result:
            return mcp_result
        try:
            parsed = json.loads(mcp_result)
        except (json.JSONDecodeError, TypeError):
            return mcp_result
        if isinstance(parsed, dict) and "content" in parsed:
            text = parsed["content"]
            if parsed.get("truncated"):
                text += "\n\n[Note: this document was truncated to fit the output budget.]"
            return text
        return mcp_result

    def _call_mcp_tool(self, tool_name: str, tool_args: dict):
        """Call an MCP tool with given name and arguments."""
        if not self.mcp:
            return None

        try:
            result = self.mcp.call_tool(tool_name, tool_args)
            return result.get("result", "")
        except Exception as e:
            session_memory.record_error(
                self.task_id, f"MCP tool call '{tool_name}' raised an exception",
                resolution=str(e),
            )
            return f"Error calling MCP tool {tool_name}: {str(e)}"

    def query(self, user_query: str, verbose=False):
        """Answer a question using RAG and optionally MCP tools."""
        context.next_turn()
        # Step 1: Retrieve from RAG
        contexts = retrieve(user_query)

        if verbose:
            print(f"📚 Retrieved {len(contexts)} relevant chunks from knowledge base")

        # Step 2: Decide (heuristically — no LLM call) if an MCP tool is needed
        mcp_result = None
        mcp_tool_used = None
        tool_name, tool_args = decide_tool_usage(user_query, contexts)
        session_memory.record_decision(
            self.task_id,
            f"Heuristic selected tool '{tool_name}'" if tool_name else "Heuristic selected no tool",
            reason=f"query={user_query!r}",
        )

        if tool_name:
            if verbose:
                print(f"🔧 Heuristic decided to use MCP tool: {tool_name} with args: {tool_args}")
            mcp_result = self._call_mcp_tool(tool_name, tool_args)
            mcp_tool_used = tool_name
            if verbose and mcp_result:
                print(f"✅ MCP tool returned result (length: {len(mcp_result)} chars)")

        # Step 3: Build prompt with RAG context
        prompt = build_prompt(user_query, contexts)

        # Step 4: Add MCP result if available
        if mcp_result:
            mcp_text = self._extract_mcp_text(mcp_result)
            prompt += f"\n\n<additional_info_from_mcp_tool>\n{mcp_text}\n</additional_info_from_mcp_tool>\n"

            try:
                parsed = json.loads(mcp_result)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict) and parsed.get("truncated"):
                session_memory.record_change(
                    self.task_id, f"Tool '{mcp_tool_used}' output was truncated",
                )

        # Step 5: Generate answer
        answer = ask_llm(prompt)

        # Step 6: Prepare response with sources
        sources = [c["source"] for c in contexts] if contexts else []

        return {
            "answer": answer,
            "sources": sources,
            "mcp_used": mcp_result is not None,
            "mcp_tool": mcp_tool_used
        }

    def close(self):
        """Clean up resources."""
        if self.mcp:
            self.mcp.close()


if __name__ == "__main__":
    assistant = CompanyKBAssistant()

    print("🤖 Company Knowledge Base Assistant")
    print("Type 'exit' or 'quit' to stop\n")

    try:
        while True:
            query = input("❓ Question: ")
            if query.lower() in {"exit", "quit"}:
                break

            print("\n" + "─" * 60)
            result = assistant.query(query, verbose=True)

            print("\n🤖 Answer:\n")

            console = Console(force_terminal=True)
            console.print(Markdown(result["answer"]))

            if result["sources"]:
                print("\n📚 Sources:")
                seen_sources = set()
                for src in result["sources"]:
                    if src not in seen_sources:
                        print(f"  • {src}")
                        seen_sources.add(src)

            if result["mcp_used"]:
                print(f"\n🔧 Used MCP tool: {result['mcp_tool']}")

            print("─" * 60 + "\n")

    finally:
        assistant.close()
