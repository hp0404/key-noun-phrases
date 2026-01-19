"""Scoring module for key noun phrase extraction.

This module provides scoring mechanisms to rank extracted key phrases
by importance within a document:

1. TF-IDF scoring: Term frequency with IDF proxy weighting
2. Co-occurrence centrality: Graph-based importance via betweenness centrality
3. Composite scoring: Weighted combination of multiple signals
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import networkx as nx

# Common function words and low-value tokens to downweight
DEFAULT_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "to",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "their",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "so",
        "such",
        "no",
        "not",
        "only",
        "own",
        "same",
        "too",
        "very",
        "just",
        "also",
        "now",
        "here",
    }
)

# POS tags typically associated with content words (higher IDF)
CONTENT_POS = frozenset({"NOUN", "PROPN", "ADJ", "VERB"})


@runtime_checkable
class Scorer(Protocol):
    """Protocol for scoring implementations."""

    def score(
        self, phrases: list[dict], document_tokens: list[str] | None = None
    ) -> list[dict]:
        """Score phrases and return them with added 'score' field.

        Parameters
        ----------
        phrases : list[dict]
            List of phrase dicts from extraction (must have 'key_noun_phrase_processed')
        document_tokens : list[str] | None
            Optional list of all tokens in the document for context

        Returns
        -------
        list[dict]
            Same phrases with 'score' field added
        """
        ...


@dataclass
class TFIDFScorer:
    """Score phrases using TF-IDF weighting.

    Within-document term frequency is combined with an IDF proxy that
    downweights common/stopword tokens.

    Attributes
    ----------
    stopwords : frozenset[str]
        Words to assign low IDF weight
    content_pos_bonus : float
        Bonus multiplier for content-word-heavy phrases
    """

    stopwords: frozenset[str] = field(default_factory=lambda: DEFAULT_STOPWORDS)
    content_pos_bonus: float = 1.2

    def _compute_tf(self, phrases: list[dict]) -> dict[str, int]:
        """Compute term frequency for each unique phrase."""
        return Counter(p["key_noun_phrase_processed"] for p in phrases)

    def _compute_idf_proxy(self, phrase_processed: str) -> float:
        """Compute IDF proxy score for a phrase.

        Phrases with more stopwords get lower scores.
        """
        tokens = phrase_processed.split()
        if not tokens:
            return 0.0

        content_count = sum(1 for t in tokens if t.lower() not in self.stopwords)
        # Score: ratio of content tokens to total tokens
        content_ratio = content_count / len(tokens)

        # Longer phrases with content get bonus
        length_factor = min(len(tokens) / 2, 2.0)  # Cap at 2x for 4+ token phrases

        return content_ratio * length_factor

    def score(
        self, phrases: list[dict], document_tokens: list[str] | None = None
    ) -> list[dict]:
        """Score phrases using TF-IDF proxy.

        Parameters
        ----------
        phrases : list[dict]
            Extracted phrases with 'key_noun_phrase_processed' field
        document_tokens : list[str] | None
            Not used in this scorer

        Returns
        -------
        list[dict]
            Phrases with 'score' field added
        """
        if not phrases:
            return phrases

        # Compute term frequencies
        tf = self._compute_tf(phrases)
        max_tf = max(tf.values()) if tf else 1

        # Score each phrase
        for p in phrases:
            processed = p["key_noun_phrase_processed"]
            term_freq = tf[processed] / max_tf  # Normalize to [0, 1]
            idf_proxy = self._compute_idf_proxy(processed)
            p["score"] = term_freq * idf_proxy

        return phrases


@dataclass
class CentralityScorer:
    """Score phrases using co-occurrence graph centrality.

    Builds a co-occurrence graph where nodes are phrases and edges connect
    phrases that appear within a sliding window. Importance is measured
    via betweenness centrality.

    Attributes
    ----------
    window_size : int
        Number of phrases to consider as co-occurring (before and after)
    """

    window_size: int = 3

    def _build_cooccurrence_graph(self, phrases: list[dict]) -> nx.Graph:
        """Build co-occurrence graph from phrase sequence.

        Phrases within window_size positions of each other are connected.
        Edge weights increase with frequency of co-occurrence.
        """
        G = nx.Graph()

        # Add all unique phrases as nodes
        unique_phrases = set(p["key_noun_phrase_processed"] for p in phrases)
        G.add_nodes_from(unique_phrases)

        # Connect phrases within window
        for i, p1 in enumerate(phrases):
            text1 = p1["key_noun_phrase_processed"]
            # Look at phrases within window
            window_end = min(i + self.window_size + 1, len(phrases))
            for j in range(i + 1, window_end):
                text2 = phrases[j]["key_noun_phrase_processed"]
                if text1 != text2:
                    if G.has_edge(text1, text2):
                        G[text1][text2]["weight"] += 1
                    else:
                        G.add_edge(text1, text2, weight=1)

        return G

    def score(
        self, phrases: list[dict], document_tokens: list[str] | None = None
    ) -> list[dict]:
        """Score phrases using betweenness centrality.

        Parameters
        ----------
        phrases : list[dict]
            Extracted phrases with 'key_noun_phrase_processed' field
        document_tokens : list[str] | None
            Not used in this scorer

        Returns
        -------
        list[dict]
            Phrases with 'score' field added
        """
        if not phrases:
            return phrases

        G = self._build_cooccurrence_graph(phrases)

        # Compute betweenness centrality
        if len(G.nodes) > 1:
            centrality = nx.betweenness_centrality(G, weight="weight")
        else:
            centrality = {node: 1.0 for node in G.nodes}

        # Normalize centrality scores
        max_centrality = max(centrality.values()) if centrality else 1.0
        if max_centrality == 0:
            max_centrality = 1.0

        # Assign scores
        for p in phrases:
            processed = p["key_noun_phrase_processed"]
            p["score"] = centrality.get(processed, 0.0) / max_centrality

        return phrases


@dataclass
class CompositeScorer:
    """Combine multiple scoring signals with configurable weights.

    Attributes
    ----------
    tfidf_weight : float
        Weight for TF-IDF score component
    centrality_weight : float
        Weight for centrality score component
    tfidf_scorer : TFIDFScorer
        TF-IDF scorer instance
    centrality_scorer : CentralityScorer
        Centrality scorer instance
    maximal_bonus : float
        Bonus multiplier for maximal spans (if is_maximal field present)
    """

    tfidf_weight: float = 0.6
    centrality_weight: float = 0.4
    tfidf_scorer: TFIDFScorer = field(default_factory=TFIDFScorer)
    centrality_scorer: CentralityScorer = field(default_factory=CentralityScorer)
    maximal_bonus: float = 1.2

    def score(
        self, phrases: list[dict], document_tokens: list[str] | None = None
    ) -> list[dict]:
        """Score phrases using weighted combination of TF-IDF and centrality.

        Parameters
        ----------
        phrases : list[dict]
            Extracted phrases
        document_tokens : list[str] | None
            Optional document tokens for context

        Returns
        -------
        list[dict]
            Phrases with 'score' field added (composite score)
        """
        if not phrases:
            return phrases

        # Make copies to avoid modifying during intermediate scoring
        phrases_copy = [dict(p) for p in phrases]

        # Get TF-IDF scores
        self.tfidf_scorer.score(phrases_copy, document_tokens)
        tfidf_scores = {
            p["key_noun_phrase_processed"]: p["score"] for p in phrases_copy
        }

        # Reset scores for centrality calculation
        for p in phrases_copy:
            p["score"] = 0.0

        # Get centrality scores
        self.centrality_scorer.score(phrases_copy, document_tokens)
        centrality_scores = {
            p["key_noun_phrase_processed"]: p["score"] for p in phrases_copy
        }

        # Combine scores on original phrases
        for p in phrases:
            processed = p["key_noun_phrase_processed"]
            tfidf = tfidf_scores.get(processed, 0.0)
            centrality = centrality_scores.get(processed, 0.0)

            combined = self.tfidf_weight * tfidf + self.centrality_weight * centrality

            # Bonus for maximal spans
            if p.get("is_maximal"):
                combined *= self.maximal_bonus

            p["score"] = combined

        return phrases


def score_phrases(
    phrases: list[dict],
    scorer: Scorer | None = None,
    document_tokens: list[str] | None = None,
) -> list[dict]:
    """Score extracted phrases using the specified scorer.

    Convenience function that defaults to CompositeScorer if none specified.

    Parameters
    ----------
    phrases : list[dict]
        Extracted phrases from TermsMatcher
    scorer : Scorer | None
        Scorer instance to use. Defaults to CompositeScorer.
    document_tokens : list[str] | None
        Optional list of document tokens for context

    Returns
    -------
    list[dict]
        Phrases with 'score' field added
    """
    if scorer is None:
        scorer = CompositeScorer()

    return scorer.score(phrases, document_tokens)
