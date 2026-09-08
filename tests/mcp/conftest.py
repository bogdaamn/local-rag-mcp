"""Test-collection workaround for a name collision, not related to Task 4's
logic: this project's local `mcp` package (`src/mcp/`) has the same name as
the `mcp` PyPI package that `fastmcp` depends on internally. Under this
repo's `pythonpath = src` (pytest.ini), `src` is on sys.path for the whole
test session, so a plain `import mcp` resolves to our *local* package
instead of the real SDK.

That's harmless for `tests/mcp/test_client.py` (it only needs
`mcp.client.MCPClient`, which is ours). But `tests/mcp/test_server.py`
imports `mcp.server`, whose module body runs `from fastmcp import FastMCP`
and decorates functions with `@mcp.tool` (the FastMCP instance, unrelated
name clash with the package). fastmcp's internals lazily do things like
`from mcp.server.lowlevel.server import LifespanResultT` and
`from mcp import LoggingLevel, ServerSession` — those need the *real* `mcp`
SDK, not our local package, or they raise ImportError/AttributeError.

This is pre-existing: it reproduces identically with the original,
unmodified `src/mcp/server.py` (verified before writing this file) — it's
just that no test imported `mcp.server` in-process before Task 4. The
actual running server is unaffected (`client.py` launches it as a
subprocess, where `src` is never prepended to `sys.path`).

Fix, contained to this test package only: before any test module below
this directory imports anything, fully import `fastmcp` (and decorate a
throwaway tool) with `src` removed from `sys.path`, so fastmcp's lazy
internals resolve and cache against the *real* `mcp` SDK once. Then evict
the real `mcp`/`mcp.*` modules from `sys.modules` and restore `sys.path`,
so `import mcp` (and `mcp.server`) resolve to our local package for the
rest of the session, as intended. fastmcp itself stays cached, so it never
needs to re-resolve `mcp.*` again — decorating the real
read_document/list_documents/search_documents functions afterwards works
against the local package.

No early-return guard on "is fastmcp already imported": that only tells us
`fastmcp`'s top-level module object exists, not that its lazy `mcp.*`
imports were actually resolved against the real SDK (fastmcp resolves
`FastMCP` lazily via `__getattr__`, so `sys.modules["fastmcp"]` can exist
without the `mcp.server.lowlevel`/`mcp.LoggingLevel` chain ever having run).
If some future test elsewhere in the suite does a bare `import fastmcp`
before this conftest runs, such a guard would skip priming and let the
shadowing ImportError resurface with nothing pointing back here. The
warmup below is cheap and fully idempotent (it always resets `sys.modules`
and `sys.path` to a known-good state), so it's simplest and safest to just
always run it, unconditionally.
"""
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[2] / "src")


def _prime_fastmcp_against_real_mcp_sdk():
    saved_path = sys.path[:]
    sys.path = [p for p in sys.path if p != _SRC]
    for name in list(sys.modules):
        if name == "mcp" or name.startswith("mcp."):
            del sys.modules[name]

    import fastmcp

    warmup = fastmcp.FastMCP("warmup")

    @warmup.tool
    def _dummy(x: str = None) -> str:
        return x or ""

    for name in list(sys.modules):
        if name == "mcp" or name.startswith("mcp."):
            del sys.modules[name]
    sys.path = saved_path


_prime_fastmcp_against_real_mcp_sdk()
