"""test_mcp_tools.py — End-to-end tests for Apollo MCP Server tools."""

import unittest
import asyncio

try:
    from apollo.server.mcp_server import create_mcp_server
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


class TestMCPTools(unittest.TestCase):

    @unittest.skipUnless(HAS_MCP, "fastmcp or mcp not installed in local environment")
    def test_mcp_server_initialization(self):
        server = create_mcp_server()
        self.assertIsNotNone(server)
        self.assertEqual(server.name, "apollo-research-server")

    @unittest.skipUnless(HAS_MCP, "fastmcp or mcp not installed in local environment")
    def test_tool_registration(self):
        server = create_mcp_server()

        async def run_check():
            tools = await server.list_tools()
            tool_names = [t.name for t in tools]
            self.assertIn("search_academic_papers", tool_names)
            self.assertIn("fetch_paper_deep_context", tool_names)
            self.assertIn("search_repo_implementations", tool_names)
            self.assertIn("search_wikipedia", tool_names)
            self.assertIn("fallback_web_search", tool_names)
            self.assertIn("match_tools_for_query", tool_names)
            self.assertIn("unified_research_context", tool_names)

        asyncio.run(run_check())


if __name__ == "__main__":
    unittest.main()
