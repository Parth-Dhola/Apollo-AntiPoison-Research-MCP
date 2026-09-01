"""test_wikipedia.py — Tests for Wikipedia API client and encyclopedia knowledge extraction."""

import unittest
import asyncio
from apollo.ingestion.wikipedia_client import search_wikipedia, fetch_wikipedia_article


class TestWikipedia(unittest.TestCase):

    def test_search_wikipedia_live(self):
        async def run_search():
            results = await search_wikipedia("Transformer machine learning", max_results=2)
            self.assertIsInstance(results, list)
            if results:  # If internet is available
                first = results[0]
                self.assertIn("title", first)
                self.assertIn("content", first)
                self.assertIn("url", first)
                self.assertEqual(first.get("source"), "wikipedia")

        asyncio.run(run_search())

    def test_fetch_wikipedia_article(self):
        async def run_fetch():
            res = await fetch_wikipedia_article("Transformer_(deep_learning_architecture)")
            if res:
                self.assertIn("title", res)
                self.assertIn("content", res)

        asyncio.run(run_fetch())


if __name__ == "__main__":
    unittest.main()
