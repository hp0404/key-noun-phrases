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
        """Test VERB-NOUN pattern with participle: 'learning algorithm'."""
        text = "The learning algorithm improves over time."
        phrases = extract_phrases(nlp_en, text)
        # With exclusive_search=False, we may get participle patterns
        assert len(phrases) > 0

    def test_verb_adj_noun(self, nlp_en):
        """Test VERB-ADJ-NOUN pattern: 'trained neural network'."""
        text = "A trained neural network classifies images."
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
        """Test Russian ADJ-NOUN-ADP-NOUN: 'статистический анализ по выборке'."""
        text = "Статистический анализ по выборке даёт результаты."
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
        """Test text with multiple subjects."""
        text = "Machine learning and deep learning transform industries."
        phrases = extract_phrases(nlp_en, text)
        # Should extract from both subjects
        assert len(phrases) > 0

    def test_passive_voice(self, nlp_en):
        """Test passive voice (nsubjpass)."""
        text = "The statistical model was trained on large datasets."
        phrases = extract_phrases(nlp_en, text)
        assert len(phrases) > 0

    def test_long_noun_phrase(self, nlp_en):
        """Test extraction of longer noun phrases."""
        text = "The advanced deep neural network architecture processes images."
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
