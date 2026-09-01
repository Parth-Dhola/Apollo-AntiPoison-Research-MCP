"""test_ingestion.py — Tests for ingestion clients with XML parsing and mocks."""

import unittest
import xml.etree.ElementTree as ET
import asyncio
from unittest.mock import patch, MagicMock
from apollo.ingestion.arxiv_client import _parse_arxiv_entry, search_arxiv
from apollo.ingestion.semantic_scholar import search_semantic_scholar
from apollo.ingestion.github_client import search_github_repos
from apollo.ingestion.web_search import search_duckduckgo
from apollo.utils.cache import SimpleCache, get_cache

_MOCK_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2205.14135v2</id>
    <published>2022-05-27T17:54:19Z</published>
    <title>FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</title>
    <summary>Transformers are slow and memory-hungry on long sequences.</summary>
    <author><name>Tri Dao</name></author>
    <author><name>Daniel Y. Fu</name></author>
    <arxiv:comment>Published at NeurIPS 2022</arxiv:comment>
    <link href="http://arxiv.org/pdf/2205.14135v2" rel="related" type="application/pdf" title="pdf"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""


class TestIngestion(unittest.TestCase):

    def test_arxiv_xml_parsing(self):
        root = ET.fromstring(_MOCK_ARXIV_XML)
        entry = root.find("{http://www.w3.org/2005/Atom}entry")
        paper = _parse_arxiv_entry(entry)

        self.assertEqual(paper.arxiv_id, "2205.14135v2")
        self.assertIn("FlashAttention", paper.title)
        self.assertEqual(paper.year, 2022)
        self.assertIn("Tri Dao", paper.authors)
        self.assertEqual(paper.pdf_url, "http://arxiv.org/pdf/2205.14135v2")
        self.assertIn("cs.LG", paper.categories)

    def test_cache_functionality(self):
        cache = SimpleCache(cache_dir=".test_cache", default_ttl=10)
        cache.set("test_ns", "test_key", {"data": "apollo_test"})
        res = cache.get("test_ns", "test_key")
        self.assertEqual(res, {"data": "apollo_test"})
        cache.clear()


if __name__ == "__main__":
    unittest.main()
