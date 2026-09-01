"""main.py — Main CLI entry point for Apollo MCP Server."""

import argparse
import sys
import logging
from apollo.config import get_settings
from apollo.server.mcp_server import create_mcp_server
from apollo.router.tool_selector import cli_main as router_cli_main


def setup_logging(debug: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)]  # Use stderr to keep stdio stdout clean for MCP protocol
    )


def main():
    parser = argparse.ArgumentParser(description="Apollo: Anti-Poisoned Research MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default=None, help="Transport protocol (stdio or sse)")
    parser.add_argument("--host", default=None, help="Host to bind for SSE transport (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind for SSE transport (default: 8080)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # If first argument is 'route' or 'selector', invoke the standalone router CLI
    if len(sys.argv) > 1 and sys.argv[1] in ("route", "selector", "router"):
        sys.argv.pop(1)
        router_cli_main()
        return

    args = parser.parse_args()
    settings = get_settings()

    debug_mode = args.debug or settings.DEBUG
    setup_logging(debug=debug_mode)

    transport = args.transport or settings.APOLLO_TRANSPORT
    host = args.host or settings.APOLLO_HOST
    port = args.port or settings.APOLLO_PORT

    server = create_mcp_server()

    if transport == "sse":
        print(f"🚀 Starting Apollo FastMCP Server (SSE mode) on {host}:{port}...", file=sys.stderr)
        server.run(transport="sse", host=host, port=port)
    else:
        print("⚡ Starting Apollo FastMCP Server (stdio mode)...", file=sys.stderr)
        server.run(transport="stdio")


if __name__ == "__main__":
    main()

