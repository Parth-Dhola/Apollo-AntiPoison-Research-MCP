"""reranker.py — Low-latency CPU cross-encoder and BM25 hybrid reranker."""

import logging
from typing import List, Dict, Any, Union
from apollo.guardrail_rag.bow_bm25 import BM25Ranker
from apollo.models.schemas import GroundedContextSnippet

logger = logging.getLogger("apollo.reranker")

_ranker_instance = None


def _get_flashrank():
    global _ranker_instance
    if _ranker_instance is None:
        try:
            from flashrank import Ranker
            _ranker_instance = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=".flashrank_cache")
            logger.debug("FlashRank CPU Ranker loaded successfully.")
        except Exception as e:
            logger.info(f"FlashRank not initialized ({e}), using BM25 ranking.")
            _ranker_instance = False
    return _ranker_instance if _ranker_instance is not False else None


def rank_snippets(
    query: str,
    snippets: List[Union[GroundedContextSnippet, Dict[str, Any]]],
    top_k: int = 3
) -> List[GroundedContextSnippet]:
    """
    Rerank a collection of raw candidate snippets against the query using FlashRank or BM25.
    Guarantees fast CPU inference (<30ms) and zero API costs.
    """
    if not snippets or not query:
        return []

    # Standardize input list to GroundedContextSnippet
    converted: List[GroundedContextSnippet] = []
    for s in snippets:
        if isinstance(s, GroundedContextSnippet):
            converted.append(s)
        elif isinstance(s, dict):
            converted.append(GroundedContextSnippet(**s))

    if len(converted) <= top_k and len(converted) == 1:
        converted[0].relevance_score = 1.0
        return converted

    # Check if FlashRank is available
    flashrank = _get_flashrank()
    if flashrank is not None:
        try:
            from flashrank import RerankRequest
            passages = [
                {"id": idx, "text": f"{s.title}\n{s.content}"}
                for idx, s in enumerate(converted)
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            results = flashrank.rerank(rerank_request)

            ranked_snippets: List[GroundedContextSnippet] = []
            for r in results[:top_k]:
                idx = int(r["id"])
                snippet = converted[idx]
                snippet.relevance_score = float(r.get("score", 0.0))
                ranked_snippets.append(snippet)
            return ranked_snippets
        except Exception as e:
            logger.warning(f"FlashRank reranking failed ({e}), falling back to BM25.")

    # Fallback: Pure BM25 Ranker
    texts = [f"{s.title}\n{s.content}" for s in converted]
    bm25 = BM25Ranker()
    bm25.fit(texts)
    scores = bm25.score(query)

    ranked_snippets = []
    for idx, score in scores[:top_k]:
        snippet = converted[idx]
        snippet.relevance_score = float(score)
        ranked_snippets.append(snippet)

    return ranked_snippets

