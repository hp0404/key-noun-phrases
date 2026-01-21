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
ENGLISH_STOPWORDS = frozenset(
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

# Ukrainian function words, pronouns, auxiliaries
UKRAINIAN_STOPWORDS = frozenset(
    {
        # Prepositions
        "в",
        "у",
        "на",
        "з",
        "із",
        "зі",
        "за",
        "до",
        "від",
        "про",
        "при",
        "для",
        "під",
        "над",
        "між",
        "через",
        "після",
        "перед",
        "біля",
        "коло",
        "без",
        "крім",
        "замість",
        "поза",
        # Conjunctions
        "та",
        "і",
        "й",
        "але",
        "а",
        "або",
        "чи",
        "ні",
        "що",
        "як",
        "коли",
        "якщо",
        "бо",
        "тому",
        "хоч",
        "хоча",
        "проте",
        "однак",
        # Pronouns
        "він",
        "вона",
        "воно",
        "вони",
        "я",
        "ми",
        "ти",
        "ви",
        "це",
        "той",
        "та",
        "те",
        "ті",
        "цей",
        "ця",
        "це",
        "ці",
        "свій",
        "своя",
        "своє",
        "свої",
        "мій",
        "моя",
        "моє",
        "мої",
        "твій",
        "твоя",
        "твоє",
        "твої",
        "наш",
        "наша",
        "наше",
        "наші",
        "ваш",
        "ваша",
        "ваше",
        "ваші",
        "їх",
        "їхній",
        "їхня",
        "їхнє",
        "їхні",
        "який",
        "яка",
        "яке",
        "які",
        "котрий",
        "котра",
        "котре",
        "котрі",
        "хто",
        "що",
        "чий",
        "чия",
        "чиє",
        "чиї",
        "сам",
        "сама",
        "саме",
        "самі",
        "весь",
        "вся",
        "все",
        "всі",
        "кожен",
        "кожна",
        "кожне",
        "кожні",
        "інший",
        "інша",
        "інше",
        "інші",
        # Auxiliaries and common verbs
        "бути",
        "є",
        "був",
        "була",
        "було",
        "були",
        "буде",
        "будуть",
        "буду",
        "будеш",
        "будемо",
        "будете",
        # Particles and adverbs
        "не",
        "ні",
        "так",
        "ще",
        "вже",
        "теж",
        "також",
        "тільки",
        "лише",
        "дуже",
        "ось",
        "там",
        "тут",
        "де",
        "куди",
        "звідки",
        "коли",
        "тоді",
        "потім",
        "зараз",
        "завжди",
        "ніколи",
        # Numerals (generic)
        "один",
        "одна",
        "одне",
        "два",
        "дві",
        "три",
        "обидва",
        "обидві",
        "обох",
    }
)

# Russian function words (similar to Ukrainian)
RUSSIAN_STOPWORDS = frozenset(
    {
        # Prepositions
        "в",
        "на",
        "с",
        "со",
        "за",
        "к",
        "до",
        "от",
        "из",
        "про",
        "при",
        "для",
        "под",
        "над",
        "между",
        "через",
        "после",
        "перед",
        "около",
        "без",
        "кроме",
        # Conjunctions
        "и",
        "а",
        "но",
        "или",
        "что",
        "как",
        "когда",
        "если",
        "потому",
        "хотя",
        "однако",
        # Pronouns
        "он",
        "она",
        "оно",
        "они",
        "я",
        "мы",
        "ты",
        "вы",
        "это",
        "тот",
        "та",
        "то",
        "те",
        "этот",
        "эта",
        "это",
        "эти",
        "свой",
        "своя",
        "своё",
        "свои",
        "мой",
        "моя",
        "моё",
        "мои",
        "твой",
        "твоя",
        "твоё",
        "твои",
        "наш",
        "наша",
        "наше",
        "наши",
        "ваш",
        "ваша",
        "ваше",
        "ваши",
        "их",
        "который",
        "которая",
        "которое",
        "которые",
        "кто",
        "что",
        "чей",
        "чья",
        "чьё",
        "чьи",
        "сам",
        "сама",
        "само",
        "сами",
        "весь",
        "вся",
        "всё",
        "все",
        "каждый",
        "каждая",
        "каждое",
        "каждые",
        "другой",
        "другая",
        "другое",
        "другие",
        # Auxiliaries
        "быть",
        "есть",
        "был",
        "была",
        "было",
        "были",
        "будет",
        "будут",
        # Particles
        "не",
        "ни",
        "да",
        "ещё",
        "уже",
        "тоже",
        "также",
        "только",
        "лишь",
        "очень",
        "вот",
        "там",
        "тут",
        "здесь",
        "где",
        "куда",
        "откуда",
        "когда",
        "тогда",
        "потом",
        "сейчас",
        "всегда",
        "никогда",
    }
)

# German function words
GERMAN_STOPWORDS = frozenset(
    {
        # Articles
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "ein",
        "eine",
        "einen",
        "einem",
        "einer",
        "eines",
        # Prepositions
        "in",
        "an",
        "auf",
        "mit",
        "bei",
        "von",
        "zu",
        "für",
        "um",
        "durch",
        "aus",
        "nach",
        "über",
        "unter",
        "vor",
        "zwischen",
        "hinter",
        "neben",
        # Conjunctions
        "und",
        "oder",
        "aber",
        "wenn",
        "als",
        "dass",
        "weil",
        "obwohl",
        "denn",
        # Pronouns
        "ich",
        "du",
        "er",
        "sie",
        "es",
        "wir",
        "ihr",
        "sie",
        "mein",
        "dein",
        "sein",
        "ihr",
        "unser",
        "euer",
        "dieser",
        "diese",
        "dieses",
        "jener",
        "jene",
        "jenes",
        "welcher",
        "welche",
        "welches",
        "wer",
        "was",
        # Auxiliaries
        "sein",
        "ist",
        "sind",
        "war",
        "waren",
        "haben",
        "hat",
        "hatte",
        "hatten",
        "werden",
        "wird",
        "wurde",
        "wurden",
        # Particles
        "nicht",
        "auch",
        "nur",
        "noch",
        "schon",
        "sehr",
        "hier",
        "dort",
        "wo",
        "wann",
        "wie",
        "warum",
    }
)

# Combined default stopwords (all languages)
DEFAULT_STOPWORDS = (
    ENGLISH_STOPWORDS | UKRAINIAN_STOPWORDS | RUSSIAN_STOPWORDS | GERMAN_STOPWORDS
)

# Language-specific stopword sets for fine-grained control
STOPWORDS_BY_LANGUAGE = {
    "en": ENGLISH_STOPWORDS,
    "uk": UKRAINIAN_STOPWORDS,
    "ru": RUSSIAN_STOPWORDS,
    "de": GERMAN_STOPWORDS,
}

# POS tags typically associated with content words (higher IDF)
CONTENT_POS = frozenset({"NOUN", "PROPN", "ADJ", "VERB"})

# Generic head noun lemmas (configurable, domain-agnostic)
# These are nouns that often appear as phrase heads but carry low semantic value
# Used to downweight phrases whose dependency-root lemma is in this set
GENERIC_HEAD_LEMMAS_UK = frozenset(
    {
        "ситуація",
        "питання",
        "проблема",
        "фактор",
        "тенденція",
        "процес",
        "випадок",
        "момент",
        "період",
        "місяць",
        "рік",
        "річ",
        "справа",
        "частина",
        "сторона",
        "пункт",
        "точка",
        "рівень",
        "вид",
        "тип",
        "форма",
        "спосіб",
        "варіант",
        "елемент",
        "аспект",
    }
)

GENERIC_HEAD_LEMMAS_RU = frozenset(
    {
        "ситуация",
        "вопрос",
        "проблема",
        "фактор",
        "тенденция",
        "процесс",
        "случай",
        "момент",
        "период",
        "месяц",
        "год",
        "вещь",
        "дело",
        "часть",
        "сторона",
        "пункт",
        "точка",
        "уровень",
        "вид",
        "тип",
        "форма",
        "способ",
        "вариант",
        "элемент",
        "аспект",
    }
)

GENERIC_HEAD_LEMMAS_EN = frozenset(
    {
        "situation",
        "question",
        "problem",
        "factor",
        "tendency",
        "process",
        "case",
        "moment",
        "period",
        "month",
        "year",
        "thing",
        "matter",
        "part",
        "side",
        "point",
        "level",
        "kind",
        "type",
        "form",
        "way",
        "variant",
        "element",
        "aspect",
    }
)

GENERIC_HEAD_LEMMAS_DE = frozenset(
    {
        "Situation",
        "Frage",
        "Problem",
        "Faktor",
        "Tendenz",
        "Prozess",
        "Fall",
        "Moment",
        "Zeitraum",
        "Monat",
        "Jahr",
        "Sache",
        "Angelegenheit",
        "Teil",
        "Seite",
        "Punkt",
        "Ebene",
        "Art",
        "Typ",
        "Form",
        "Weise",
        "Variante",
        "Element",
        "Aspekt",
    }
)

# Combined default generic head lemmas
DEFAULT_GENERIC_HEAD_LEMMAS = (
    GENERIC_HEAD_LEMMAS_UK
    | GENERIC_HEAD_LEMMAS_RU
    | GENERIC_HEAD_LEMMAS_EN
    | GENERIC_HEAD_LEMMAS_DE
)

# Language-specific generic head lemma sets
GENERIC_HEAD_LEMMAS_BY_LANGUAGE = {
    "en": GENERIC_HEAD_LEMMAS_EN,
    "uk": GENERIC_HEAD_LEMMAS_UK,
    "ru": GENERIC_HEAD_LEMMAS_RU,
    "de": GENERIC_HEAD_LEMMAS_DE,
}

# POS-label quality weights for pattern types
# Higher weights indicate more domain-specific patterns
PATTERN_WEIGHTS: dict[str, float] = {
    # High quality noun phrase patterns
    "ADJ-NOUN": 1.0,
    "ADJ-ADJ-NOUN": 1.1,
    "NOUN-ADJ-NOUN": 1.0,
    "ADJ-NOUN-NOUN": 1.0,
    "ADJ-ADJ-ADJ-NOUN": 1.1,
    # Proper noun patterns (domain-specific)
    "PROPN": 1.2,
    "PROPN-PROPN": 1.3,
    "ADJ-PROPN": 1.2,
    "PROPN-NOUN": 1.1,
    "NOUN-PROPN": 1.1,
    # Noun compound patterns
    "NOUN-NOUN": 0.9,
    "NOUN-NOUN-NOUN": 0.9,
    "NOUN-NOUN-NOUN-NOUN": 0.85,
    "NOUN-NOUN-NOUN-NOUN-NOUN": 0.8,
    # Prepositional patterns (often fragmentary)
    "NOUN-ADP-NOUN": 0.75,
    "ADJ-NOUN-ADP-NOUN": 0.8,
    "NOUN-ADP-ADJ-NOUN": 0.8,
    "ADJ-NOUN-ADP-ADJ-NOUN": 0.85,
    "NOUN-ADP-NOUN-NOUN": 0.75,
    "NOUN-ADP-NOUN-ADP-NOUN": 0.7,
    # Verb/participle patterns
    "VERB-NOUN": 0.7,
    "VERB-ADJ-NOUN": 0.7,
    "VERB-NOUN-NOUN": 0.7,
    # Participle patterns (restricted VERB with VerbForm=Part)
    "PART-NOUN": 0.85,
    "PART-ADJ-NOUN": 0.85,
    "PART-NOUN-NOUN": 0.85,
    "NOUN-PART-NOUN": 0.8,
    "ADJ-PART-NOUN": 0.85,
    "PART-NOUN-ADP-NOUN": 0.75,
    # Additional PROPN patterns
    "PROPN-NUM": 1.1,
    "NUM-PROPN": 1.1,
    "PROPN-ADJ": 1.1,
    # Coordinated adjective pattern
    "ADJ-CCONJ-ADJ-NOUN": 0.9,
    # Hyphenated patterns
    "ADJ-PUNCT-NOUN-NOUN": 0.85,
    "ADJ-PUNCT-ADJ-NOUN": 0.85,
    "NOUN-PUNCT-NOUN-NOUN": 0.8,
    # Generic/low-value patterns (if kept)
    "DET-NOUN": 0.4,
    "DET-ADJ-NOUN": 0.5,
    "NUM-NOUN": 0.4,
    "NUM-ADJ-NOUN": 0.5,
    "NOUN-VERB": 0.3,
}

# Default weight for unknown patterns
DEFAULT_PATTERN_WEIGHT = 0.7


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
    downweights common/stopword tokens, plus POS-label quality weights
    and PROPN boost.

    Attributes
    ----------
    stopwords : frozenset[str]
        Words to assign low IDF weight
    propn_boost : float
        Multiplier for phrases containing proper nouns (PROPN in pos_label)
    use_pattern_weights : bool
        Whether to apply POS-label quality weights
    generic_head_lemmas : frozenset[str]
        Lemmas considered generic/low-value as phrase heads
    generic_head_penalty : float
        Penalty multiplier for phrases with generic head lemmas (0-1, lower = stronger penalty)
    adp_penalty : float
        Penalty multiplier for phrases containing ADP tokens (0-1, lower = stronger penalty)
    propn_min_frequency : int
        Minimum frequency threshold for standalone PROPN patterns
    propn_all_caps_boost : float
        Bonus multiplier for all-caps PROPN (likely acronyms)
    """

    stopwords: frozenset[str] = field(default_factory=lambda: DEFAULT_STOPWORDS)
    propn_boost: float = 1.3
    use_pattern_weights: bool = True
    generic_head_lemmas: frozenset[str] = field(
        default_factory=lambda: DEFAULT_GENERIC_HEAD_LEMMAS
    )
    generic_head_penalty: float = 0.5
    adp_penalty: float = 0.6
    propn_min_frequency: int = 2
    propn_all_caps_boost: float = 1.2

    def _compute_tf(self, phrases: list[dict]) -> dict[str, int]:
        """Compute term frequency for each unique phrase."""
        return Counter(p["key_noun_phrase_processed"] for p in phrases)

    def _compute_idf_proxy(self, phrase_processed: str) -> float:
        """Compute IDF proxy score for a phrase.

        Phrases with more stopwords get lower scores.
        Uses content ratio without length bias.
        """
        tokens = phrase_processed.split()
        if not tokens:
            return 0.0

        content_count = sum(1 for t in tokens if t.lower() not in self.stopwords)
        # Score: ratio of content tokens to total tokens
        # No length bias - quality over quantity
        content_ratio = content_count / len(tokens)

        return content_ratio

    def _get_pattern_weight(self, pos_label: str | None) -> float:
        """Get quality weight for a POS pattern.

        Parameters
        ----------
        pos_label : str | None
            POS pattern label (e.g., "ADJ-NOUN", "NOUN-ADP-NOUN")

        Returns
        -------
        float
            Weight multiplier for this pattern type
        """
        if not self.use_pattern_weights or not pos_label:
            return 1.0
        return PATTERN_WEIGHTS.get(pos_label, DEFAULT_PATTERN_WEIGHT)

    def _has_propn(self, pos_label: str | None) -> bool:
        """Check if pattern contains a proper noun.

        Parameters
        ----------
        pos_label : str | None
            POS pattern label

        Returns
        -------
        bool
            True if pattern contains PROPN
        """
        if not pos_label:
            return False
        return "PROPN" in pos_label

    def _has_adp(self, pos_label: str | None) -> bool:
        """Check if pattern contains a preposition (ADP).

        Parameters
        ----------
        pos_label : str | None
            POS pattern label

        Returns
        -------
        bool
            True if pattern contains ADP
        """
        if not pos_label:
            return False
        return "ADP" in pos_label

    def _is_standalone_propn(self, pos_label: str | None) -> bool:
        """Check if pattern is a standalone proper noun.

        Parameters
        ----------
        pos_label : str | None
            POS pattern label

        Returns
        -------
        bool
            True if pattern is exactly "PROPN"
        """
        return pos_label == "PROPN"

    def _has_generic_head(self, phrase: dict) -> bool:
        """Check if phrase has a generic head noun.

        Uses head_lemma field if available, otherwise extracts last
        token as heuristic head (head-final assumption for Slavic languages).

        Parameters
        ----------
        phrase : dict
            Phrase dict with 'key_noun_phrase_processed' and optionally 'head_lemma'

        Returns
        -------
        bool
            True if phrase head is a generic lemma
        """
        # Prefer explicit head_lemma if available
        head_lemma = phrase.get("head_lemma")
        if head_lemma:
            return head_lemma.lower() in self.generic_head_lemmas

        # Fallback: use last token as heuristic head
        processed = phrase.get("key_noun_phrase_processed", "")
        tokens = processed.split()
        if tokens:
            last_token = tokens[-1].lower()
            return last_token in self.generic_head_lemmas

        return False

    def _is_all_caps(self, phrase_text: str) -> bool:
        """Check if phrase text is all uppercase (likely acronym).

        Parameters
        ----------
        phrase_text : str
            The phrase text to check

        Returns
        -------
        bool
            True if phrase is all uppercase letters
        """
        # Only consider alphabetic characters
        alpha_chars = [c for c in phrase_text if c.isalpha()]
        return len(alpha_chars) > 0 and all(c.isupper() for c in alpha_chars)

    def score(
        self, phrases: list[dict], document_tokens: list[str] | None = None
    ) -> list[dict]:
        """Score phrases using TF-IDF proxy with pattern weights and penalties.

        Applies:
        - Base TF-IDF proxy scoring
        - POS pattern quality weights
        - PROPN boost (for phrases containing proper nouns)
        - Generic head penalty (for phrases with generic head nouns)
        - ADP penalty (for phrases containing prepositions)
        - PROPN mitigation (frequency threshold and all-caps boost for standalone PROPN)

        Parameters
        ----------
        phrases : list[dict]
            Extracted phrases with 'key_noun_phrase_processed' and optionally 'pos_label'
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
            pos_label = p.get("pos_label")
            phrase_freq = tf[processed]

            # Base TF-IDF proxy score
            term_freq = phrase_freq / max_tf  # Normalize to [0, 1]
            idf_proxy = self._compute_idf_proxy(processed)

            # Apply pattern quality weight
            pattern_weight = self._get_pattern_weight(pos_label)

            # Apply PROPN boost if applicable
            propn_multiplier = self.propn_boost if self._has_propn(pos_label) else 1.0

            # Apply generic head penalty (§3.1)
            generic_head_multiplier = (
                self.generic_head_penalty if self._has_generic_head(p) else 1.0
            )

            # Apply ADP penalty (§3.2)
            adp_multiplier = self.adp_penalty if self._has_adp(pos_label) else 1.0

            # Apply PROPN mitigation for standalone PROPN (§3.3)
            propn_mitigation_multiplier = 1.0
            if self._is_standalone_propn(pos_label):
                # Penalize low-frequency standalone PROPNs
                if phrase_freq < self.propn_min_frequency:
                    propn_mitigation_multiplier = 0.5
                # Boost all-caps PROPNs (likely acronyms)
                elif self._is_all_caps(processed):
                    propn_mitigation_multiplier = self.propn_all_caps_boost

            p["score"] = (
                term_freq
                * idf_proxy
                * pattern_weight
                * propn_multiplier
                * generic_head_multiplier
                * adp_multiplier
                * propn_mitigation_multiplier
            )

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
