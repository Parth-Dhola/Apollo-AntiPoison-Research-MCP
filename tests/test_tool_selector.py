"""test_tool_selector.py — Tests for standalone Bag-of-Words tool selector."""

import unittest
from apollo.router.tool_selector import ToolSelector, select_tools_for_query
from apollo.models.schemas import QueryIntent


class TestToolSelector(unittest.TestCase):

    def test_academic_intent(self):
        selector = ToolSelector()
        res = selector.analyze_intent("Attention mechanism theorem and loss function convergence in transformers")
        self.assertIn(res.intent, (QueryIntent.ACADEMIC_PAPER, QueryIntent.DEEP_THEORY))
        self.assertIn("search_academic_papers", res.recommended_tools)
        self.assertGreaterEqual(res.confidence, 0.6)

    def test_code_intent(self):
        res = select_tools_for_query("PyTorch implementation of LoRA linear layer with cuda forward script")
        self.assertEqual(res.intent, QueryIntent.CODE_IMPLEMENTATION)
        self.assertIn("search_repo_implementations", res.recommended_tools)
        self.assertIn("pytorch", res.keywords)

    def test_arxiv_id_intent(self):
        res = select_tools_for_query("Can you explain 2307.08691v2 in detail?")
        self.assertEqual(res.intent, QueryIntent.ACADEMIC_PAPER)
        self.assertIn("fetch_paper_deep_context", res.recommended_tools)
        self.assertGreater(res.confidence, 0.9)

    def test_hybrid_intent(self):
        res = select_tools_for_query("Survey of diffusion architectures with github repository code implementations")
        self.assertEqual(res.intent, QueryIntent.HYBRID)
        self.assertIn("search_academic_papers", res.recommended_tools)
        self.assertIn("search_repo_implementations", res.recommended_tools)

    def test_web_intent(self):
        res = select_tools_for_query("What is the latest OpenAI product announcement and pricing today?")
        self.assertEqual(res.intent, QueryIntent.GENERAL_WEB)
        self.assertIn("fallback_web_search", res.recommended_tools)


if __name__ == "__main__":
    unittest.main()

