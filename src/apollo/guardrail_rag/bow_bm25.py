"""bow_bm25.py — Pure Python zero-dependency Bag-of-Words and Okapi BM25 scoring."""

import math
import re
from typing import List, Dict, Tuple, Any

# Standard English stopwords
_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def tokenize_query(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize query string into normalized lowercase terms, extracting both whole and subword terms."""
    if not text:
        return []
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    sub_tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    all_tokens = list(dict.fromkeys(raw_tokens + sub_tokens))
    if remove_stopwords:
        return [w for w in all_tokens if w not in _STOPWORDS and len(w) > 1]
    return all_tokens


class BM25Ranker:
    """Okapi BM25 implementation for zero-cost CPU retrieval and reranking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.documents: List[Any] = []

    def fit(self, documents: List[str]) -> "BM25Ranker":
        """Index a collection of text documents."""
        self.documents = documents
        self.corpus_size = len(documents)
        self.doc_freqs = {}
        self.doc_lengths = []
        self.doc_term_freqs = []

        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            return self

        total_len = 0
        for doc in documents:
            tokens = tokenize_query(doc, remove_stopwords=False)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_len += doc_len

            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_term_freqs.append(tf)

            for token in tf.keys():
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0.0
        return self

    def _calc_idf(self, term: str) -> float:
        n_q = self.doc_freqs.get(term, 0)
        # Standard Lucene/Okapi smoothed IDF
        return math.log(1.0 + (self.corpus_size - n_q + 0.5) / (n_q + 0.5))

    def score(self, query: str) -> List[Tuple[int, float]]:
        """
        Score all indexed documents against the query string.
        Returns list of (doc_index, score) sorted descending by score.
        """
        if self.corpus_size == 0 or not query:
            return []

        query_tokens = tokenize_query(query, remove_stopwords=True)
        if not query_tokens:
            query_tokens = tokenize_query(query, remove_stopwords=False)

        scores: List[Tuple[int, float]] = []
        for idx in range(self.corpus_size):
            doc_score = 0.0
            doc_len = self.doc_lengths[idx]
            tf_dict = self.doc_term_freqs[idx]

            for term in query_tokens:
                if term not in tf_dict:
                    continue

                f = tf_dict[term]
                idf = self._calc_idf(term)
                denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                term_score = idf * (f * (self.k1 + 1.0)) / denom
                doc_score += term_score

            scores.append((idx, doc_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

