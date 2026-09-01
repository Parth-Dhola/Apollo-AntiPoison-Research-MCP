"""test_guardrail_rag.py — Tests for BM25 ranking, reranking, and grounded snippet packing."""

import unittest
from apollo.guardrail_rag.bow_bm25 import BM25Ranker, tokenize_query
from apollo.guardrail_rag.reranker import rank_snippets
from apollo.guardrail_rag.snippet_packer import pack_grounded_snippets
from apollo.models.schemas import GroundedContextSnippet


class TestGuardrailRAG(unittest.TestCase):

    def test_bm25_tokenize(self):
        tokens = tokenize_query("What is the asymptotic complexity of Quicksort?")
        self.assertIn("asymptotic", tokens)
        self.assertIn("complexity", tokens)
        self.assertIn("quicksort", tokens)
        self.assertNotIn("is", tokens)  # stopword removed
        self.assertNotIn("the", tokens)

    def test_bm25_ranking(self):
        docs = [
            "Quicksort is an efficient sorting algorithm with average O(n log n) runtime complexity.",
            "Convolutional neural networks are widely used for image recognition and computer vision tasks.",
            "FlashAttention optimizes GPU memory IO for transformer attention."
        ]
        ranker = BM25Ranker()
        ranker.fit(docs)
        scores = ranker.score("sorting algorithm runtime complexity")

        self.assertEqual(len(scores), 3)
        # Top document should be index 0
        top_doc_idx, top_score = scores[0]
        self.assertEqual(top_doc_idx, 0)
        self.assertGreater(top_score, 0.0)

    def test_rank_snippets_and_pack(self):
        snippets = [
            GroundedContextSnippet(
                source="arxiv",
                title="FlashAttention: Fast and Memory-Efficient Exact Attention",
                url="https://arxiv.org/abs/2205.14135",
                content="We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling.",
                citation_meta={"arxiv_id": "2205.14135", "year": 2022, "authors": ["Tri Dao"]}
            ),
            GroundedContextSnippet(
                source="github",
                title="Dao-AILab/flash-attention",
                url="https://github.com/Dao-AILab/flash-attention",
                content="Fast and memory-efficient exact attention with IO-awareness in PyTorch and CUDA.",
                citation_meta={"stars": 15000, "language": "C++"}
            )
        ]

        ranked = rank_snippets("FlashAttention algorithm tiling", snippets, top_k=2)
        self.assertEqual(len(ranked), 2)
        self.assertGreaterEqual(ranked[0].relevance_score, 0.0)

        packed_output = pack_grounded_snippets(ranked)
        self.assertIn("### Snippet 1:", packed_output)
        self.assertIn("FlashAttention", packed_output)
        self.assertIn("<untrusted_academic_context", packed_output)
        self.assertIn("Verified Safe", packed_output)
        self.assertIn("Tier 1: Peer-Reviewed", packed_output)

    def test_source_authority_weighting(self):
        # Even with identical text content, arXiv (Tier 1, weight 1.0) must outrank Wikipedia (Tier 3, weight 0.65)
        snippets = [
            GroundedContextSnippet(
                source="wikipedia",
                title="Transformer Architecture - Wikipedia",
                content="The Transformer is a deep learning architecture using self-attention mechanisms.",
                citation_meta={}
            ),
            GroundedContextSnippet(
                source="arxiv",
                title="Attention Is All You Need - arXiv",
                content="The Transformer is a deep learning architecture using self-attention mechanisms.",
                citation_meta={"arxiv_id": "1706.03762"}
            )
        ]
        ranked = rank_snippets("Transformer self-attention architecture", snippets, top_k=2)
        self.assertEqual(ranked[0].source, "arxiv")
        self.assertEqual(ranked[1].source, "wikipedia")
        self.assertGreater(ranked[0].relevance_score, ranked[1].relevance_score)


if __name__ == "__main__":
    unittest.main()

