"""test_tool_rag.py — Tests for Tool Capability RAG and Context Overload Prevention."""

import unittest
from apollo.router.tool_rag import ToolRAGSelector, rank_tools_for_query


class TestToolRAG(unittest.TestCase):

    def test_academic_tool_matching(self):
        selector = ToolRAGSelector()
        matched = selector.select_tools("Survey on diffusion models with loss convergence theorems")
        tool_names = [t["tool_name"] for t in matched]
        self.assertIn("search_academic_papers", tool_names)
        self.assertNotIn("fetch_paper_deep_context", tool_names)

    def test_code_tool_matching(self):
        matched = rank_tools_for_query("PyTorch implementation of LoRA linear layer CUDA kernel")
        tool_names = [t["tool_name"] for t in matched]
        self.assertIn("search_repo_implementations", tool_names)

    def test_arxiv_id_short_circuit(self):
        matched = rank_tools_for_query("Explain arXiv paper 2205.14135")
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["tool_name"], "fetch_paper_deep_context")
        self.assertIn("Short-circuiting", matched[0]["reason"])

    def test_web_fallback_matching(self):
        matched = rank_tools_for_query("What is the latest OpenAI GPT-4o pricing and release date?")
        tool_names = [t["tool_name"] for t in matched]
        self.assertIn("fallback_web_search", tool_names)


if __name__ == "__main__":
    unittest.main()
