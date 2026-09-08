import json
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from telemetry.tool_middleware import record_tool_call


def _normalize_result(response):
    """Unwrap FastMCP's real envelope shape into the flat {"result": value}
    shape every downstream caller (assistant.py, tool_middleware.py) and
    existing test fake already assumes.

    A real FastMCP subprocess response looks like:
        {"result": {"_meta": {...}, "content": [{"text": "...", "type": "text"}],
                     "isError": false, "structuredContent": {"result": "..."}}}

    We want response["result"] to become the plain value the tool function
    actually returned, i.e. structuredContent["result"].

    If response["result"] is already a plain (non-dict) value, it's the
    legacy/test-fake flat shape - leave it untouched.

    If response["result"] is a dict but has no structuredContent (e.g. a
    minimal FastMCP response), fall back to content[0]["text"] when present,
    since that's the same plain value FastMCP mirrors into structuredContent.
    Only if neither is available do we leave the dict as-is rather than
    raising - callers already treat unexpected shapes as opaque data.
    """
    result = response.get("result")
    if not isinstance(result, dict):
        return response

    structured = result.get("structuredContent")
    if isinstance(structured, dict) and "result" in structured:
        response["result"] = structured["result"]
        return response

    content = result.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict) and "text" in content[0]:
        response["result"] = content[0]["text"]
        return response

    return response


class MCPClient:
    """Client for communicating with MCP server."""
    
    def __init__(self, cmd):
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        self.lock = threading.Lock()
        self.next_id = 1
        self.initialize()

    def _send(self, payload):
        """Send JSON-RPC message and receive response."""
        with self.lock:
            data = json.dumps(payload)
            self.proc.stdin.write(data + "\n")
            self.proc.stdin.flush()

            line = self.proc.stdout.readline()
            if not line:
                raise ConnectionError("MCP server disconnected")
            return json.loads(line)

    def initialize(self):
        """Initialize MCP connection."""
        payload = {
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "company-kb-assistant", "version": "1.0"}
            }
        }
        self.next_id += 1

        return self._send(payload)

    def call_tool(self, name, arguments):
        """Call an MCP tool, recording telemetry for the call."""
        def _do_call():
            payload = {
                "jsonrpc": "2.0",
                "id": self.next_id,
                "method": "tools/call",
                "params": {
                    "name": name,
                    "arguments": arguments
                }
            }
            self.next_id += 1
            response = self._send(payload)
            return _normalize_result(response)

        return record_tool_call(name, arguments, _do_call)
    
    def close(self):
        """Close the MCP connection."""
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
