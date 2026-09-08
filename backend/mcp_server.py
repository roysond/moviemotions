"""The same two tools, exposed over MCP so a client that is not this repository can use them.

WHY THIS FILE IS SHORT, AND WHY THAT IS THE POINT
    It defines no tool. `search_films` and `lookup_film` are plain Python functions in
    `backend/tools.py` — they import no LangChain, know nothing about MCP, and return a
    string. This file and the agent are two TRANSPORTS over one definition.

    A tool defined twice is a tool that will eventually behave two ways, and the
    divergence appears in whichever transport nobody exercised that week. Here there is
    nothing to diverge: the docstring an MCP client reads is the same object the agent
    reads, because it is the same function.

WHY THE IMPORT IS INSIDE A FUNCTION
    Same bargain as the Google packages in `backend/models.py`: a deployment that serves
    HTTP has no use for the MCP libraries, and importing at module level would drag them
    into requirements-runtime.txt and into every container image. The cost is that a
    missing package must explain itself, which is what the error below does.

WHAT A CLIENT GETS
    Two tools and nothing else. No database credentials, no model keys, no way to write
    anything — every path here is a read. Least privilege is not a setting to add later;
    it is the fact that there is no write tool to expose.

RUN
    pip install "mcp[cli]"
    python -m backend.mcp_server          speaks MCP over stdin/stdout

    Point any MCP client at that command. There is no port and no network listener:
    stdio means the client starts the process and talks to it directly.
"""

from backend.tools import lookup_film, search_films

SERVER_NAME = "moviemotions"

EXPOSED = (search_films, lookup_film)


def build_server():
    """Create the MCP server and register the tools. Importing here, not at the top.

    MCPServer reads each function's SIGNATURE for the argument schema and its DOCSTRING
    for the description — the same two things LangChain sends to a model. So the prose
    written for the agent is, without any further work, the prose an MCP client shows
    its user. That is the reward for the docstring being the interface.
    """
    # mcp 2.x. In 1.x this class was `FastMCP` at `mcp.server.fastmcp`; it was renamed
    # and moved. Pinned at 2.2.0 in requirements.txt so a future rename cannot arrive
    # silently with a `pip install`.
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as problem:
        # "MISSING" AND "CHANGED" ARE DIFFERENT FAILURES AND MUST NOT SHARE A MESSAGE.
        # The first version of this told the reader to install a package that was
        # already installed — the import failed because mcp 2.x renamed the class, and
        # the error confidently sent them down the wrong path. So: look first, then
        # speak.
        import importlib.util

        if importlib.util.find_spec("mcp") is None:
            raise RuntimeError(
                "The MCP server needs a package this project does not ship by "
                "default:\n\n    pip install \"mcp[cli]\"\n\n"
                "It is deliberately absent from requirements-runtime.txt — a container "
                "that serves HTTP has no use for it."
            ) from problem

        raise RuntimeError(
            "mcp IS installed, but this file cannot find MCPServer in it. That is a "
            "VERSION mismatch, not a missing package — the class lived at "
            "mcp.server.fastmcp as FastMCP in 1.x and moved in 2.x.\n\n"
            "    pip install 'mcp==2.2.0'\n\n"
            f"Underlying error: {problem}"
        ) from problem

    server = MCPServer(SERVER_NAME)
    for function in EXPOSED:
        server.tool()(function)          # the documented registration, applied by hand
    return server


def describe():
    """Build the server, register the tools, and print what a client would receive.

    WHY THIS EXISTS
        Run without arguments, this program sits silent and produces nothing — which is
        CORRECT for stdio, because the client owns the conversation and speaks first.
        But correct-and-silent is indistinguishable from hung, and "trust me, it works"
        is not a verification. This proves the import resolved, the server constructed,
        and every tool registered without raising — then exits.
    """
    import inspect

    server = build_server()
    print(f"MCP server '{SERVER_NAME}' built · {type(server).__name__}")
    print(f"{len(EXPOSED)} tools registered\n")

    for function in EXPOSED:
        signature = inspect.signature(function)
        summary = (function.__doc__ or "").strip().splitlines()[0]
        print(f"  {function.__name__}{signature}")
        print(f"      {summary}\n")

    print("Registration succeeded. Run without --list to serve on stdin/stdout;")
    print("it will then print nothing and wait, which is what a stdio server does.")


def main():
    import sys

    if "--list" in sys.argv:
        describe()
        return

    # stdio: the CLIENT starts this process and talks over its stdin and stdout. There
    # is no port and nothing to connect to, which is why running it by hand looks like a
    # hang. Silence here is correct — use --list to prove the wiring without serving.
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
