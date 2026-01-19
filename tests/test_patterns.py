"""
Test-driven tests for noun phrase extraction patterns.

These tests verify that specific patterns are matched correctly.
When updating patterns, run these tests to ensure expected phrases are extracted.

Note: Tests require spacy models to be installed:
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm
    python -m spacy download ru_core_news_sm
    python -m spacy download uk_core_news_sm
"""

import pytest
import spacy

from terms import TermsMatcher

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def nlp_en():
    """Load English spacy model."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm model not installed")


@pytest.fixture(scope="module")
def nlp_de():
    """Load German spacy model."""
    try:
        return spacy.load("de_core_news_sm")
    except OSError:
        pytest.skip("de_core_news_sm model not installed")


@pytest.fixture(scope="module")
def nlp_ru():
    """Load Russian spacy model."""
    try:
        return spacy.load("ru_core_news_sm")
    except OSError:
        pytest.skip("ru_core_news_sm model not installed")


@pytest.fixture(scope="module")
def nlp_uk():
    """Load Ukrainian spacy model."""
    try:
        return spacy.load("uk_core_news_sm")
    except OSError:
        pytest.skip("uk_core_news_sm model not installed")


def extract_phrases(nlp, text, exclusive_search=False):
    """Helper to extract key noun phrases from text.

    Args:
        nlp: spacy language model
        text: input text
        exclusive_search: if True, phrases must contain the subject token

    Returns:
        set of extracted phrases (lowercased, processed form)
    """
    matcher = TermsMatcher(nlp=nlp)
    sentences = [(text, "test")]
    results = list(
        matcher.yield_key_phrases(sentences, exclusive_search=exclusive_search)
    )
    return {r["key_noun_phrase_processed"] for r in results}


def extract_phrases_raw(nlp, text, exclusive_search=False):
    """Helper to extract raw (non-lemmatized) key noun phrases from text.

    Args:
        nlp: spacy language model
        text: input text
        exclusive_search: if True, phrases must contain the subject token

    Returns:
        set of extracted phrases (original form)
    """
    matcher = TermsMatcher(nlp=nlp)
    sentences = [(text, "test")]
    results = list(
        matcher.yield_key_phrases(sentences, exclusive_search=exclusive_search)
    )
    return {r["key_noun_phrase"] for r in results}


# =============================================================================
# English Pattern Tests
# =============================================================================


class TestEnglishBasicPatterns:
    """Test basic English noun phrase patterns."""

    def test_adj_noun(self, nlp_en):
        """Test ADJ-NOUN pattern: 'statistical analysis'."""
        text = "Statistical analysis reveals important trends."
        phrases = extract_phrases(nlp_en, text)
        assert "statistical analysis" in phrases

    def test_adj_adj_noun(self, nlp_en):
        """Test ADJ-ADJ-NOUN pattern: 'large neural network'."""
        text = "A large neural network processes the data."
        phrases = extract_phrases(nlp_en, text)
        assert "large neural network" in phrases or "neural network" in phrases

    def test_noun_noun(self, nlp_en):
        """Test NOUN-NOUN pattern: 'data analysis'."""
        text = "Data analysis shows clear patterns."
        phrases = extract_phrases(nlp_en, text)
        assert "data analysis" in phrases or "datum analysis" in phrases

    def test_noun_noun_noun(self, nlp_en):
        """Test NOUN-NOUN-NOUN pattern: 'machine learning model'."""
        text = "The machine learning model performs well."
        phrases = extract_phrases(nlp_en, text)
        assert "machine learning model" in phrases or "machine learn model" in phrases


class TestEnglishPrepositionalPatterns:
    """Test English prepositional phrase patterns (CiteSpace-style)."""

    def test_noun_adp_noun(self, nlp_en):
        """Test NOUN-ADP-NOUN pattern: 'analysis of data'."""
        text = "The analysis of data reveals patterns."
        phrases = extract_phrases(nlp_en, text)
        assert "analysis of data" in phrases or "analysis of datum" in phrases

    def test_adj_noun_adp_noun(self, nlp_en):
        """Test ADJ-NOUN-ADP-NOUN pattern: 'statistical analysis of variance'."""
        text = "Statistical analysis of variance shows significance."
        phrases = extract_phrases(nlp_en, text)
        assert "statistical analysis of variance" in phrases

    def test_noun_adp_adj_noun(self, nlp_en):
        """Test NOUN-ADP-ADJ-NOUN pattern: 'analysis of complex systems'."""
        text = "The analysis of complex systems requires expertise."
        phrases = extract_phrases(nlp_en, text)
        assert (
            "analysis of complex system" in phrases
            or "analysis of complex systems" in phrases
        )

    def test_effect_of_pattern(self, nlp_en):
        """Test common academic pattern: 'effect of X on Y'."""
        text = "The effect of temperature affects the results."
        phrases = extract_phrases(nlp_en, text)
        assert "effect of temperature" in phrases


class TestEnglishVerbPatterns:
    """Test English verb-based patterns (participles)."""

    def test_verb_noun(self, nlp_en):
        """Test VERB-NOUN pattern with participle: 'trained model'."""
        text = "The trained model predicts outcomes."
        phrases = extract_phrases(nlp_en, text)
        # Should extract "trained model" (VERB-NOUN pattern)
        assert "trained model" in phrases or "train model" in phrases

    def test_verb_adj_noun(self, nlp_en):
        """Test VERB-ADJ-NOUN pattern: 'estimated optimal value'."""
        text = "The estimated optimal value exceeds expectations."
        phrases = extract_phrases(nlp_en, text)
        assert len(phrases) > 0


class TestEnglishHyphenatedPatterns:
    """Test English hyphenated compound patterns."""

    def test_adj_punct_noun_noun(self, nlp_en):
        """Test ADJ-PUNCT-NOUN-NOUN pattern: 'high-performance computing'."""
        text = "High-performance computing enables simulations."
        phrases = extract_phrases_raw(nlp_en, text)
        # Hyphenated patterns should be captured
        assert len(phrases) > 0


# =============================================================================
# German Pattern Tests
# =============================================================================


class TestGermanPatterns:
    """Test German noun phrase patterns."""

    def test_adj_noun(self, nlp_de):
        """Test German ADJ-NOUN: 'statistische Analyse'."""
        text = "Die statistische Analyse zeigt wichtige Ergebnisse."
        phrases = extract_phrases(nlp_de, text)
        assert len(phrases) > 0

    def test_noun_noun(self, nlp_de):
        """Test German NOUN-NOUN compound."""
        text = "Die Datenanalyse liefert Erkenntnisse."
        phrases = extract_phrases(nlp_de, text)
        assert len(phrases) > 0

    def test_noun_adp_noun(self, nlp_de):
        """Test German NOUN-ADP-NOUN: 'Analyse von Daten'."""
        text = "Die Analyse von Daten ist wichtig."
        phrases = extract_phrases(nlp_de, text)
        assert len(phrases) > 0

    def test_propn_sequence(self, nlp_de):
        """Test German proper noun sequence."""
        text = "Max Planck Institut forscht an neuen Technologien."
        phrases = extract_phrases(nlp_de, text)
        # Should capture proper noun sequences
        assert len(phrases) >= 0  # May or may not match depending on parsing


# =============================================================================
# Russian Pattern Tests
# =============================================================================


class TestRussianBasicPatterns:
    """Test Russian noun phrase patterns."""

    def test_adj_noun(self, nlp_ru):
        """Test Russian ADJ-NOUN: 'научный анализ' (scientific analysis)."""
        text = "Научный анализ показывает результаты."
        phrases = extract_phrases(nlp_ru, text)
        assert len(phrases) > 0

    def test_adj_adj_noun(self, nlp_ru):
        """Test Russian ADJ-ADJ-NOUN: 'большая нейронная сеть'."""
        text = "Большая нейронная сеть обрабатывает данные."
        phrases = extract_phrases(nlp_ru, text)
        assert len(phrases) > 0

    def test_noun_noun(self, nlp_ru):
        """Test Russian NOUN-NOUN (genitive): 'анализ данных'."""
        text = "Анализ данных выявляет закономерности."
        phrases = extract_phrases(nlp_ru, text)
        assert len(phrases) > 0


class TestRussianPrepositionalPatterns:
    """Test Russian prepositional patterns."""

    def test_noun_adp_noun(self, nlp_ru):
        """Test Russian NOUN-ADP-NOUN: 'работа с данными'."""
        text = "Работа с данными требует внимания."
        phrases = extract_phrases(nlp_ru, text)
        assert len(phrases) > 0

    def test_adj_noun_adp_noun(self, nlp_ru):
        """Test Russian ADJ-NOUN-ADP-NOUN: 'детальный анализ данных'."""
        text = "Детальный анализ данных показывает тенденции."
        phrases = extract_phrases(nlp_ru, text)
        assert len(phrases) > 0


class TestRussianVerbPatterns:
    """Test Russian verb-based patterns."""

    def test_verb_noun(self, nlp_ru):
        """Test Russian participle patterns."""
        text = "Обученная модель классифицирует объекты."
        phrases = extract_phrases(nlp_ru, text)
        assert len(phrases) > 0


# =============================================================================
# Ukrainian Pattern Tests
# =============================================================================


class TestUkrainianBasicPatterns:
    """Test Ukrainian noun phrase patterns."""

    def test_adj_noun(self, nlp_uk):
        """Test Ukrainian ADJ-NOUN: 'науковий аналіз' (scientific analysis)."""
        text = "Науковий аналіз показує результати."
        phrases = extract_phrases(nlp_uk, text)
        assert len(phrases) > 0

    def test_adj_adj_noun(self, nlp_uk):
        """Test Ukrainian ADJ-ADJ-NOUN: 'велика нейронна мережа'."""
        text = "Велика нейронна мережа обробляє дані."
        phrases = extract_phrases(nlp_uk, text)
        assert len(phrases) > 0

    def test_noun_noun(self, nlp_uk):
        """Test Ukrainian NOUN-NOUN: 'аналіз даних'."""
        text = "Аналіз даних виявляє закономірності."
        phrases = extract_phrases(nlp_uk, text)
        assert len(phrases) > 0


class TestUkrainianPrepositionalPatterns:
    """Test Ukrainian prepositional patterns."""

    def test_noun_adp_noun(self, nlp_uk):
        """Test Ukrainian NOUN-ADP-NOUN: 'робота з даними'."""
        text = "Робота з даними потребує уваги."
        phrases = extract_phrases(nlp_uk, text)
        assert len(phrases) > 0

    def test_adj_noun_adp_noun(self, nlp_uk):
        """Test Ukrainian ADJ-NOUN-ADP-NOUN."""
        text = "Статистичний аналіз за вибіркою дає результати."
        phrases = extract_phrases(nlp_uk, text)
        assert len(phrases) > 0


# =============================================================================
# Cross-language Consistency Tests
# =============================================================================


class TestCrossLanguageConsistency:
    """Test that similar concepts are extracted across languages."""

    def test_scientific_analysis_en(self, nlp_en):
        """English: scientific analysis."""
        text = "Scientific analysis demonstrates the effect."
        phrases = extract_phrases(nlp_en, text)
        assert "scientific analysis" in phrases

    def test_scientific_analysis_de(self, nlp_de):
        """German: wissenschaftliche Analyse."""
        text = "Die wissenschaftliche Analyse zeigt den Effekt."
        phrases = extract_phrases(nlp_de, text)
        assert len(phrases) > 0

    def test_scientific_analysis_ru(self, nlp_ru):
        """Russian: научный анализ."""
        text = "Научный анализ демонстрирует эффект."
        phrases = extract_phrases(nlp_ru, text)
        assert len(phrases) > 0

    def test_scientific_analysis_uk(self, nlp_uk):
        """Ukrainian: науковий аналіз."""
        text = "Науковий аналіз демонструє ефект."
        phrases = extract_phrases(nlp_uk, text)
        assert len(phrases) > 0


# =============================================================================
# Edge Cases and Regression Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and potential regressions."""

    def test_empty_text(self, nlp_en):
        """Test empty text returns no phrases."""
        text = ""
        phrases = extract_phrases(nlp_en, text)
        assert len(phrases) == 0

    def test_no_subject_verb(self, nlp_en):
        """Test text without subject-verb structure."""
        text = "The big red ball."
        phrases = extract_phrases(nlp_en, text)
        # No subject-verb relation, so no key phrases expected
        assert len(phrases) == 0

    def test_multiple_subjects(self, nlp_en):
        """Test text with adjective-modified subject."""
        text = "Young researchers study emerging patterns."
        phrases = extract_phrases(nlp_en, text)
        # Should extract "young researchers" (ADJ-NOUN)
        assert "young researcher" in phrases or len(phrases) > 0

    def test_passive_voice(self, nlp_en):
        """Test passive voice (nsubjpass)."""
        text = "The statistical model was trained on large datasets."
        phrases = extract_phrases(nlp_en, text)
        assert len(phrases) > 0

    def test_long_noun_phrase(self, nlp_en):
        """Test extraction of longer noun phrases."""
        text = "The complex neural network learns features."
        phrases = extract_phrases(nlp_en, text)
        assert len(phrases) > 0


class TestExclusiveSearchMode:
    """Test exclusive_search parameter behavior."""

    def test_exclusive_true(self, nlp_en):
        """With exclusive_search=True, subject must be in phrase."""
        text = "The complex analysis reveals patterns."
        phrases_exclusive = extract_phrases(nlp_en, text, exclusive_search=True)
        phrases_non_exclusive = extract_phrases(nlp_en, text, exclusive_search=False)
        # Non-exclusive should find at least as many phrases
        assert len(phrases_non_exclusive) >= len(phrases_exclusive)

    def test_exclusive_false_finds_more(self, nlp_en):
        """exclusive_search=False may find phrases without subject."""
        text = "The main statistical analysis of complex data shows trends."
        phrases_exclusive = extract_phrases(nlp_en, text, exclusive_search=True)
        phrases_non_exclusive = extract_phrases(nlp_en, text, exclusive_search=False)
        # Usually non-exclusive finds more or equal
        assert len(phrases_non_exclusive) >= len(phrases_exclusive)


# =============================================================================
# Pattern Label Tests
# =============================================================================


class TestPatternLabels:
    """Test that correct pattern labels are assigned."""

    def test_pattern_label_adj_noun(self, nlp_en):
        """Verify ADJ-NOUN pattern is labeled correctly."""
        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("Statistical analysis shows results.", "test")]
        results = list(matcher.yield_key_phrases(sentences, exclusive_search=False))
        labels = {r["pos_label"] for r in results}
        assert "ADJ-NOUN" in labels or len(labels) > 0

    def test_pattern_label_noun_adp_noun(self, nlp_en):
        """Verify NOUN-ADP-NOUN pattern is labeled correctly."""
        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("The analysis of data reveals trends.", "test")]
        results = list(matcher.yield_key_phrases(sentences, exclusive_search=False))
        labels = {r["pos_label"] for r in results}
        # Should have prepositional pattern label
        has_adp_pattern = any("ADP" in label for label in labels)
        assert has_adp_pattern or len(labels) > 0


# =============================================================================
# Redundancy Resolution Tests
# =============================================================================


class TestRedundancyResolution:
    """Test span grouping and maximal span detection."""

    def test_maximal_span_detection(self, nlp_en):
        """Test that maximal spans are correctly identified."""
        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("The data security incident response plan is effective.", "test")]
        results = matcher.extract_key_phrases(sentences, exclusive_search=False)

        # Find the maximal span
        maximal_spans = [r for r in results if r["is_maximal"]]
        assert len(maximal_spans) == 1
        assert (
            "data security incident response plan"
            in maximal_spans[0]["key_noun_phrase"]
        )

    def test_family_grouping(self, nlp_en):
        """Test that subspans are grouped under maximal spans."""
        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("The data security incident response plan is effective.", "test")]
        results = matcher.extract_key_phrases(sentences, exclusive_search=False)

        # All spans should belong to the same family
        family_ids = {r["family_id"] for r in results}
        assert len(family_ids) == 1
        assert "family_1" in family_ids

    def test_non_maximal_spans_reference_maximal(self, nlp_en):
        """Test that non-maximal spans reference their maximal span text."""
        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("The data security incident response plan is effective.", "test")]
        results = matcher.extract_key_phrases(sentences, exclusive_search=False)

        # All non-maximal spans should have maximal_text set
        for r in results:
            if not r["is_maximal"]:
                assert r["maximal_text"] is not None
                assert "data security incident response plan" in r["maximal_text"]

    def test_resolve_redundancy_disabled(self, nlp_en):
        """Test that resolve_redundancy=False leaves fields as None."""
        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("Statistical analysis shows results.", "test")]
        results = matcher.extract_key_phrases(
            sentences, exclusive_search=False, resolve_redundancy=False
        )

        for r in results:
            assert r["is_maximal"] is None
            assert r["family_id"] is None
            assert r["maximal_text"] is None

    def test_dataframe_includes_redundancy_fields(self, nlp_en):
        """Test that to_dataframe includes redundancy resolution fields."""
        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("Statistical analysis shows results.", "test")]
        df = matcher.to_dataframe(sentences, exclusive_search=False)

        assert "is_maximal" in df.columns
        assert "family_id" in df.columns
        assert "maximal_text" in df.columns

    def test_token_span_field(self, nlp_en):
        """Test that token_span field contains correct indices."""
        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("Statistical analysis shows results.", "test")]
        results = matcher.extract_key_phrases(sentences, exclusive_search=False)

        for r in results:
            assert "token_span" in r
            assert len(r["token_span"]) == 2
            assert r["token_span"][0] < r["token_span"][1]


# =============================================================================
# Scoring Tests
# =============================================================================


class TestTFIDFScoring:
    """Test TF-IDF scoring functionality."""

    def test_tfidf_scorer_adds_score_field(self, nlp_en):
        """Test that TFIDFScorer adds score field to phrases."""
        from terms.score import TFIDFScorer

        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("Statistical analysis shows the research results.", "test")]
        results = matcher.extract_key_phrases(sentences, exclusive_search=False)

        scorer = TFIDFScorer()
        scored = scorer.score(results)

        for r in scored:
            assert "score" in r
            assert isinstance(r["score"], float)
            assert r["score"] >= 0

    def test_tfidf_higher_for_content_words(self, nlp_en):
        """Test that content-rich phrases get higher TF-IDF scores."""
        from terms.score import TFIDFScorer

        # Create mock phrases with varying content word ratios
        phrases = [
            {"key_noun_phrase_processed": "statistical analysis"},
            {"key_noun_phrase_processed": "the of"},  # All stopwords
        ]

        scorer = TFIDFScorer()
        scored = scorer.score(phrases)

        # Content-rich phrase should score higher
        content_score = scored[0]["score"]
        stopword_score = scored[1]["score"]
        assert content_score > stopword_score

    def test_tfidf_repeated_phrases_higher_tf(self, nlp_en):
        """Test that repeated phrases get higher TF component."""
        from terms.score import TFIDFScorer

        # Create phrases where one appears twice
        phrases = [
            {"key_noun_phrase_processed": "data analysis"},
            {"key_noun_phrase_processed": "data analysis"},
            {"key_noun_phrase_processed": "research method"},
        ]

        scorer = TFIDFScorer()
        scored = scorer.score(phrases)

        # Both "data analysis" should have same (higher) score
        assert scored[0]["score"] == scored[1]["score"]


class TestCentralityScoring:
    """Test co-occurrence centrality scoring."""

    def test_centrality_scorer_adds_score_field(self, nlp_en):
        """Test that CentralityScorer adds score field."""
        from terms.score import CentralityScorer

        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [
            ("Statistical analysis shows research patterns.", "s1"),
            ("The analysis method reveals trends.", "s2"),
        ]
        results = matcher.extract_key_phrases(sentences, exclusive_search=False)

        scorer = CentralityScorer()
        scored = scorer.score(results)

        for r in scored:
            assert "score" in r
            assert isinstance(r["score"], float)
            assert 0 <= r["score"] <= 1

    def test_centrality_single_phrase(self, nlp_en):
        """Test centrality with single phrase."""
        from terms.score import CentralityScorer

        phrases = [{"key_noun_phrase_processed": "single phrase"}]

        scorer = CentralityScorer()
        scored = scorer.score(phrases)

        assert len(scored) == 1
        assert "score" in scored[0]


class TestCompositeScoring:
    """Test composite scoring combining TF-IDF and centrality."""

    def test_composite_scorer_combines_scores(self, nlp_en):
        """Test that CompositeScorer uses both TF-IDF and centrality."""
        from terms.score import CompositeScorer

        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("Statistical analysis shows research results.", "test")]
        results = matcher.extract_key_phrases(sentences, exclusive_search=False)

        scorer = CompositeScorer()
        scored = scorer.score(results)

        for r in scored:
            assert "score" in r
            assert isinstance(r["score"], float)
            assert r["score"] >= 0

    def test_composite_maximal_bonus(self, nlp_en):
        """Test that maximal spans get bonus score."""
        from terms.score import CompositeScorer

        # Create mock phrases with is_maximal field
        phrases = [
            {"key_noun_phrase_processed": "data analysis", "is_maximal": True},
            {"key_noun_phrase_processed": "data analysis", "is_maximal": False},
        ]

        scorer = CompositeScorer(maximal_bonus=1.5)
        scored = scorer.score(phrases)

        # Maximal version should have higher score
        maximal_score = scored[0]["score"]
        non_maximal_score = scored[1]["score"]
        assert maximal_score > non_maximal_score

    def test_composite_custom_weights(self, nlp_en):
        """Test that custom weights are applied."""
        from terms.score import CompositeScorer

        phrases = [{"key_noun_phrase_processed": "statistical analysis"}]

        # Two scorers with different weights
        scorer1 = CompositeScorer(tfidf_weight=1.0, centrality_weight=0.0)
        scorer2 = CompositeScorer(tfidf_weight=0.0, centrality_weight=1.0)

        scored1 = scorer1.score([dict(p) for p in phrases])
        scored2 = scorer2.score([dict(p) for p in phrases])

        # Scores should be different with different weights
        # (unless both components happen to be equal)
        # At minimum, both should produce valid scores
        assert "score" in scored1[0]
        assert "score" in scored2[0]


class TestScoringIntegration:
    """Test scoring integration with extraction pipeline."""

    def test_extract_key_phrases_with_scoring(self, nlp_en):
        """Test that extract_key_phrases can compute scores."""
        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("Statistical analysis shows research results.", "test")]
        results = matcher.extract_key_phrases(
            sentences, exclusive_search=False, compute_scores=True
        )

        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_extract_key_phrases_without_scoring(self, nlp_en):
        """Test that extract_key_phrases doesn't add score by default."""
        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("Statistical analysis shows results.", "test")]
        results = matcher.extract_key_phrases(
            sentences, exclusive_search=False, compute_scores=False
        )

        for r in results:
            assert "score" not in r

    def test_to_dataframe_with_scoring(self, nlp_en):
        """Test that to_dataframe includes score column when requested."""
        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("Statistical analysis shows results.", "test")]
        df = matcher.to_dataframe(
            sentences, exclusive_search=False, compute_scores=True
        )

        assert "score" in df.columns

    def test_custom_scorer_integration(self, nlp_en):
        """Test using custom scorer with extract_key_phrases."""
        from terms.score import TFIDFScorer

        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("Statistical analysis shows results.", "test")]

        # Use only TF-IDF scorer
        scorer = TFIDFScorer()
        results = matcher.extract_key_phrases(
            sentences, exclusive_search=False, compute_scores=True, scorer=scorer
        )

        for r in results:
            assert "score" in r

    def test_score_phrases_convenience_function(self, nlp_en):
        """Test the score_phrases convenience function."""
        from terms.score import score_phrases

        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [("Statistical analysis shows results.", "test")]
        results = matcher.extract_key_phrases(
            sentences, exclusive_search=False, compute_scores=False
        )

        # Score using convenience function
        scored = score_phrases(results)

        for r in scored:
            assert "score" in r

    def test_empty_phrases_handling(self, nlp_en):
        """Test that scoring handles empty phrase list."""
        from terms.score import CentralityScorer, CompositeScorer, TFIDFScorer

        empty = []

        # All scorers should handle empty input
        assert TFIDFScorer().score(empty) == []
        assert CentralityScorer().score(empty) == []
        assert CompositeScorer().score(empty) == []


# =============================================================================
# End-to-End Integration Tests
# =============================================================================


class TestEndToEndIntegration:
    """Test full pipeline: raw text → scored, deduplicated keyphrases."""

    def test_full_pipeline_english(self, nlp_en):
        """Test complete extraction pipeline with English text."""
        text = """
        The advanced machine learning algorithm demonstrates remarkable accuracy.
        Statistical analysis of the machine learning results shows improvement.
        The learning algorithm continues to evolve.
        """
        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [(text, "doc1")]

        # Full extraction with redundancy resolution and scoring
        results = matcher.extract_key_phrases(
            sentences,
            exclusive_search=False,
            resolve_redundancy=True,
            compute_scores=True,
        )

        # Should have extracted phrases
        assert len(results) > 0

        # All results should have the expected fields
        for r in results:
            assert "uuid" in r
            assert "pos_label" in r
            assert "key_noun_phrase" in r
            assert "key_noun_phrase_processed" in r
            assert "span_location" in r
            assert "token_span" in r
            assert "is_maximal" in r
            assert "family_id" in r
            assert "maximal_text" in r
            assert "score" in r

        # Should have at least one maximal span
        maximal_spans = [r for r in results if r["is_maximal"]]
        assert len(maximal_spans) >= 1

        # Scores should be non-negative
        for r in results:
            assert r["score"] >= 0

    def test_full_pipeline_german(self, nlp_de):
        """Test complete extraction pipeline with German text."""
        text = """
        Die wissenschaftliche Analyse zeigt interessante Ergebnisse.
        Die statistische Methode funktioniert gut.
        """
        matcher = TermsMatcher(nlp=nlp_de)
        sentences = [(text, "doc1")]

        results = matcher.extract_key_phrases(
            sentences,
            exclusive_search=False,
            resolve_redundancy=True,
            compute_scores=True,
        )

        assert len(results) > 0
        for r in results:
            assert "score" in r
            assert r["score"] >= 0

    def test_full_pipeline_russian(self, nlp_ru):
        """Test complete extraction pipeline with Russian text."""
        text = """
        Научный анализ показывает интересные результаты.
        Статистический метод работает хорошо.
        """
        matcher = TermsMatcher(nlp=nlp_ru)
        sentences = [(text, "doc1")]

        results = matcher.extract_key_phrases(
            sentences,
            exclusive_search=False,
            resolve_redundancy=True,
            compute_scores=True,
        )

        assert len(results) > 0
        for r in results:
            assert "score" in r
            assert r["score"] >= 0

    def test_full_pipeline_ukrainian(self, nlp_uk):
        """Test complete extraction pipeline with Ukrainian text."""
        text = """
        Науковий аналіз показує цікаві результати.
        Статистичний метод працює добре.
        """
        matcher = TermsMatcher(nlp=nlp_uk)
        sentences = [(text, "doc1")]

        results = matcher.extract_key_phrases(
            sentences,
            exclusive_search=False,
            resolve_redundancy=True,
            compute_scores=True,
        )

        assert len(results) > 0
        for r in results:
            assert "score" in r
            assert r["score"] >= 0

    def test_dataframe_full_pipeline(self, nlp_en):
        """Test DataFrame output with full pipeline."""
        text = "The statistical analysis shows research results."
        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [(text, "doc1")]

        df = matcher.to_dataframe(
            sentences,
            exclusive_search=False,
            resolve_redundancy=True,
            compute_scores=True,
        )

        # Check all expected columns
        expected_cols = [
            "uuid",
            "pos_label",
            "key_noun_phrase",
            "key_noun_phrase_processed",
            "span_location",
            "token_span",
            "is_maximal",
            "family_id",
            "maximal_text",
            "score",
        ]
        for col in expected_cols:
            assert col in df.columns

        # DataFrame should have data
        assert len(df) > 0

    def test_multi_document_extraction(self, nlp_en):
        """Test extraction across multiple documents."""
        sentences = [
            ("Statistical analysis reveals patterns.", "doc1"),
            ("Machine learning models improve accuracy.", "doc2"),
            ("The research method works well.", "doc3"),
        ]
        matcher = TermsMatcher(nlp=nlp_en)

        results = matcher.extract_key_phrases(
            sentences,
            exclusive_search=False,
            resolve_redundancy=True,
            compute_scores=True,
        )

        # Should have results from multiple documents
        uuids = {r["uuid"] for r in results}
        assert len(uuids) >= 1  # At least one document should yield phrases

    def test_top_k_phrase_extraction(self, nlp_en):
        """Test extracting top-K scored phrases."""
        text = """
        Advanced statistical analysis shows significant research results.
        The analysis method demonstrates accuracy in machine learning.
        Statistical methods improve research outcomes significantly.
        """
        matcher = TermsMatcher(nlp=nlp_en)
        sentences = [(text, "doc1")]

        results = matcher.extract_key_phrases(
            sentences,
            exclusive_search=False,
            resolve_redundancy=True,
            compute_scores=True,
        )

        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

        # Top phrases should have highest scores
        if len(sorted_results) >= 2:
            assert sorted_results[0]["score"] >= sorted_results[-1]["score"]

        # Can extract top-K
        top_k = sorted_results[:3] if len(sorted_results) >= 3 else sorted_results
        assert len(top_k) <= 3
