"""CiteSpace-style Concept Tree Builder.

This module implements the concept tree extraction algorithm described in
Chen (2012) "Turning Points: The Nature of Creativity", Chapter 7.

The algorithm:
1. POS-tags input text
2. Extracts noun phrases using pattern matching
3. Splits phrases into head nouns (concepts) and modifiers (attributes)
4. Builds a hierarchical tree where heads are parents and modifiers are children
5. Tracks frequency and sentence context for each concept
"""

from __future__ import annotations

import html
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import spacy
from spacy.tokens import Doc, Span, Token


@dataclass
class ConceptNode:
    """A node in the concept tree."""

    name: str
    freq: int = 0
    source: str = "p"
    contexts: list[str] = field(default_factory=list)
    children: dict[str, ConceptNode] = field(default_factory=dict)

    def add_context(self, context: str) -> None:
        """Add a context sentence if not already present."""
        if context and context not in self.contexts:
            self.contexts.append(context)

    def increment_freq(self, amount: int = 1) -> None:
        """Increment frequency counter."""
        self.freq += amount


@dataclass
class ExtractedPhrase:
    """An extracted noun phrase with metadata."""

    text: str
    head: str
    modifiers: list[str]
    sentence: str
    is_proper_noun: bool = False


class ConceptTreeBuilder:
    """Build a concept tree from text using the CiteSpace algorithm."""

    # Articles to strip from phrases
    ARTICLES = {"a", "an", "the"}

    # Words to filter out entirely (pronouns, determiners, etc.)
    FILTER_WORDS = {
        "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
        "my", "our", "your", "his", "her", "its", "their",
        "this", "that", "these", "those",
        "which", "who", "whom", "what", "whose",
        "itself", "himself", "herself", "themselves", "ourselves",
        "some", "any", "no", "every", "all", "both", "each", "few", "more", "most",
        "other", "such", "only", "example", "fact",
    }

    # Phrases to skip entirely (common idioms that aren't meaningful concepts)
    SKIP_PHRASES = {"good times", "times"}

    # POS tags to skip when they appear alone
    SKIP_POS = {"PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART", "INTJ", "X", "SYM"}

    def __init__(self, nlp: spacy.language.Language | None = None):
        """Initialize the concept tree builder.

        Parameters
        ----------
        nlp : spacy.language.Language, optional
            A spaCy language model. If not provided, loads en_core_web_sm.
        """
        if nlp is None:
            nlp = spacy.load("en_core_web_sm")
        self.nlp = nlp
        self.root = ConceptNode(name="Concepts", freq=1, source="p")

    def process_text(self, text: str, source: str = "p", doc_title: str = "Untitled") -> None:
        """Process text and extract concepts into the tree.

        Parameters
        ----------
        text : str
            The input text to process
        source : str
            Source identifier (default "p" for primary)
        doc_title : str
            Document title for context links
        """
        doc = self.nlp(text)

        for sent in doc.sents:
            sent_text = sent.text.strip()
            # Clean the sentence text for context (remove smart quotes, normalize)
            clean_sent = self._clean_sentence(sent_text)

            phrases = self._extract_phrases_from_sentence(sent, clean_sent)

            for phrase in phrases:
                self._add_phrase_to_tree(phrase, source)

    def _clean_sentence(self, text: str) -> str:
        """Clean sentence text for storage in context."""
        # Replace smart quotes with regular ones
        text = text.replace("'", "'").replace("'", "'")
        text = text.replace(""", '"').replace(""", '"')
        return text

    def _extract_phrases_from_sentence(
        self, sent: Span, sent_text: str
    ) -> list[ExtractedPhrase]:
        """Extract noun phrases from a sentence.

        Parameters
        ----------
        sent : Span
            spaCy sentence span
        sent_text : str
            Cleaned sentence text

        Returns
        -------
        list[ExtractedPhrase]
            List of extracted phrases
        """
        phrases = []
        seen_spans = set()

        # First, detect "X of Y" patterns at sentence level
        of_patterns = self._detect_of_patterns(sent, sent_text)
        for phrase in of_patterns:
            phrases.append(phrase)
            # Mark these spans as seen
            seen_spans.add(phrase.text.lower())

        # Extract standalone possessive forms (e.g., "Schmitt's" → "schmitts")
        possessive_phrases = self._extract_possessives(sent, sent_text)
        for phrase in possessive_phrases:
            if phrase.head not in seen_spans:
                phrases.append(phrase)
                seen_spans.add(phrase.head)

        # Process noun chunks
        for chunk in sent.noun_chunks:
            # Skip if entirely covered by filter words
            content_tokens = [t for t in chunk if t.text.lower() not in self.FILTER_WORDS
                           and t.text.lower() not in self.ARTICLES
                           and t.pos_ not in self.SKIP_POS]
            if not content_tokens:
                continue

            phrase = self._process_noun_chunk(chunk, sent_text)
            if phrase and phrase.head:
                # Skip if this head was already captured in an "of" pattern
                if phrase.head in seen_spans:
                    continue
                span_key = (chunk.start, chunk.end)
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    phrases.append(phrase)

        return phrases

    def _detect_of_patterns(self, sent: Span, sent_text: str) -> list[ExtractedPhrase]:
        """Detect and extract 'X of Y' patterns from a sentence.

        In CiteSpace, "X of Y" creates Y as head with X as modifier.
        E.g., "guarantor of peace" → head="peace", modifier="guarantor"
              "times of war" → head="war", modifier="times"

        Only extracts patterns where Y is a concrete concept (not abstract/action nouns).

        Parameters
        ----------
        sent : Span
            spaCy sentence span
        sent_text : str
            Cleaned sentence text

        Returns
        -------
        list[ExtractedPhrase]
            List of "of" pattern phrases
        """
        phrases = []

        # Words that should NOT be heads in "of" patterns (too abstract or actions)
        skip_heads = {
            "departure", "becoming", "understanding", "understating", "conceiving",
            "relying", "geopolitics", "political", "example",
        }

        # Words that should NOT be modifiers in "of" patterns
        skip_modifiers = {"point", "concept", "kind", "sort", "type", "form"}

        # Find all "of" tokens in the sentence
        for i, token in enumerate(sent):
            if token.text.lower() == "of" and token.pos_ == "ADP":
                # Look for noun before "of" and noun after "of"
                before_noun = self._find_noun_before(sent, i)
                after_noun = self._find_noun_after(sent, i)

                if before_noun and after_noun:
                    # Create phrase with after_noun as head, before_noun as modifier
                    head = self._clean_token_text(after_noun)
                    modifier = self._clean_token_text(before_noun)

                    # Skip unwanted patterns
                    if head.lower() in skip_heads:
                        continue
                    if modifier.lower() in skip_modifiers:
                        continue

                    if (head and modifier
                        and head.lower() not in self.FILTER_WORDS
                        and modifier.lower() not in self.FILTER_WORDS):
                        phrases.append(ExtractedPhrase(
                            text=f"{modifier} of {head}",
                            head=head,
                            modifiers=[modifier],
                            sentence=sent_text,
                        ))

        return phrases

    def _extract_possessives(self, sent: Span, sent_text: str) -> list[ExtractedPhrase]:
        """Extract standalone possessive forms as concepts.

        E.g., "Schmitt's understating" → extract "schmitts" as standalone concept.
        Only extracts when the possessive is NOT part of a larger noun phrase.

        Parameters
        ----------
        sent : Span
            spaCy sentence span
        sent_text : str
            Cleaned sentence text

        Returns
        -------
        list[ExtractedPhrase]
            List of possessive phrases
        """
        phrases = []

        # Get all noun chunk boundaries to check if possessive is inside a chunk
        chunk_ranges = set()
        for chunk in sent.noun_chunks:
            for j in range(chunk.start, chunk.end):
                chunk_ranges.add(j)

        for i, token in enumerate(sent):
            # Look for possessive marker (handles both straight and curly apostrophes)
            # Check if token looks like a possessive marker
            token_lower = token.text.lower()
            is_possessive = (
                token.pos_ == "PART"
                and (token_lower.endswith("s") or token_lower in ("'", "\u2019"))
                and ("'" in token.text or "\u2019" in token.text)
            )
            if is_possessive:
                # Check if previous token is a proper noun
                if i > 0:
                    prev_token = sent[i - 1]
                    prev_idx = prev_token.i

                    # Skip if this possessive is part of a noun chunk
                    # (e.g., "EU's numerous deficiencies" - EU's is part of chunk)
                    if prev_idx in chunk_ranges:
                        continue

                    if prev_token.pos_ == "PROPN":
                        # Create concept from possessive form
                        # "Schmitt's" → "schmitts" (name + s)
                        name = prev_token.text.lower() + "s"
                        if name and name not in self.FILTER_WORDS:
                            phrases.append(ExtractedPhrase(
                                text=prev_token.text + token.text,
                                head=name,
                                modifiers=[],
                                sentence=sent_text,
                            ))

        return phrases

    def _find_noun_before(self, sent: Span, of_idx: int) -> Token | None:
        """Find the noun token before 'of' in the sentence."""
        # Look backwards from of_idx
        for i in range(of_idx - 1, max(of_idx - 5, -1), -1):
            token = sent[i]
            if token.pos_ in ("NOUN", "PROPN"):
                # Skip if it's a filter word
                if token.text.lower() not in self.FILTER_WORDS:
                    return token
            elif token.pos_ in ("DET", "ADJ", "ADV"):
                continue  # Skip articles/adjectives
            else:
                break  # Stop on other POS
        return None

    def _find_noun_after(self, sent: Span, of_idx: int) -> Token | None:
        """Find the noun token after 'of' in the sentence."""
        # Look forwards from of_idx
        for i in range(of_idx + 1, min(of_idx + 5, len(sent))):
            token = sent[i]
            if token.pos_ in ("NOUN", "PROPN"):
                # Skip if it's a filter word
                if token.text.lower() not in self.FILTER_WORDS:
                    return token
            elif token.pos_ in ("DET", "ADJ"):
                continue  # Skip articles/adjectives
            else:
                break  # Stop on other POS
        return None

    def _process_noun_chunk(
        self, chunk: Span, sent_text: str
    ) -> ExtractedPhrase | None:
        """Process a noun chunk into an ExtractedPhrase.

        Parameters
        ----------
        chunk : Span
            spaCy noun chunk
        sent_text : str
            Original sentence text

        Returns
        -------
        ExtractedPhrase | None
            Extracted phrase or None if should be skipped
        """
        # Get tokens, handling special cases
        tokens = list(chunk)

        # Skip common idioms/phrases that shouldn't be concepts
        chunk_lower = chunk.text.lower().strip()
        # Remove articles for comparison
        for art in self.ARTICLES:
            chunk_lower = chunk_lower.replace(art + " ", "")
        chunk_lower = chunk_lower.strip()
        if chunk_lower in self.SKIP_PHRASES:
            return None

        # Check if this is a proper noun phrase (named entity)
        if self._is_proper_noun_phrase(chunk):
            text = self._normalize_phrase_text(tokens)
            if not text or text.lower() in self.FILTER_WORDS:
                return None
            return ExtractedPhrase(
                text=text,
                head=text.lower(),
                modifiers=[],
                sentence=sent_text,
                is_proper_noun=True,
            )

        # Check if this is a hyphenated compound (e.g., "self-understanding")
        if self._is_hyphenated_compound(chunk):
            # Filter out leading possessives/determiners
            content_tokens = [t for t in tokens
                              if t.text.lower() not in self.ARTICLES
                              and t.text.lower() not in self.FILTER_WORDS]
            text = self._normalize_phrase_text(content_tokens)
            if not text or text.lower() in self.FILTER_WORDS:
                return None
            # Keep as single concept without modifiers
            return ExtractedPhrase(
                text=text,
                head=text.lower(),
                modifiers=[],
                sentence=sent_text,
            )

        # Filter out articles and filter words at the beginning
        filtered_tokens = []
        for t in tokens:
            lower = t.text.lower()
            if lower in self.ARTICLES:
                continue
            if lower in self.FILTER_WORDS and t.pos_ in ("PRON", "DET"):
                continue
            filtered_tokens.append(t)

        if not filtered_tokens:
            return None

        # Check for "X of Y" pattern
        of_idx = None
        for i, t in enumerate(filtered_tokens):
            if t.text.lower() == "of":
                of_idx = i
                break

        if of_idx is not None and of_idx > 0 and of_idx < len(filtered_tokens) - 1:
            return self._process_of_pattern(filtered_tokens, of_idx, sent_text)

        # Standard pattern: modifiers + head noun
        return self._process_standard_pattern(filtered_tokens, sent_text)

    def _is_proper_noun_phrase(self, chunk: Span) -> bool:
        """Check if chunk is a pure proper noun phrase (named entity or multi-word proper noun).

        Returns True only for phrases like "Carl Schmitt", "European Union", "Ukraine".
        Returns False for mixed patterns like "soft power Europe" (ADJ + NOUN + PROPN).
        """
        # Check if it's a named entity
        if chunk.root.ent_type_:
            # GPE, ORG, PERSON, etc.
            if chunk.root.ent_type_ in ("GPE", "ORG", "PERSON", "LOC", "FAC", "NORP"):
                # But check if there are non-proper-noun content words before it
                content_tokens = [t for t in chunk if t.pos_ not in ("DET", "ADP", "PUNCT")
                                  and t.text.lower() not in self.ARTICLES]
                # If there are adjectives or common nouns before the proper noun, don't treat as pure proper noun
                non_propn = [t for t in content_tokens if t.pos_ not in ("PROPN",)]
                if non_propn:
                    return False
                return True

        # Check if all content words are proper nouns
        content_tokens = [t for t in chunk if t.pos_ not in ("DET", "ADP", "PUNCT")
                        and t.text.lower() not in self.ARTICLES]
        if content_tokens and all(t.pos_ == "PROPN" for t in content_tokens):
            return True

        return False

    def _is_hyphenated_compound(self, chunk: Span) -> bool:
        """Check if chunk is a hyphenated compound noun (not adjective) to keep as one concept.

        Returns True only for noun-noun hyphenated compounds like "self-understanding".
        Returns False for adjective-noun patterns like "ever-growing ambition" so they
        can be split into head + modifier.
        """
        # Filter to content tokens (skip articles and possessive pronouns)
        tokens = [t for t in chunk if t.text.lower() not in self.ARTICLES
                  and t.text.lower() not in self.FILTER_WORDS]

        if len(tokens) < 2:
            return False

        # Check if there's a hyphen in the tokens
        has_hyphen = any(t.text == "-" or "-" in t.text for t in tokens)
        if not has_hyphen:
            return False

        # Find the hyphen position
        hyphen_idx = None
        for i, t in enumerate(tokens):
            if t.text == "-":
                hyphen_idx = i
                break

        if hyphen_idx is None:
            return False

        # Get tokens before and after hyphen
        before_hyphen = tokens[:hyphen_idx] if hyphen_idx > 0 else []
        after_hyphen = tokens[hyphen_idx + 1:] if hyphen_idx + 1 < len(tokens) else []

        if not before_hyphen or not after_hyphen:
            return False

        # Check the POS of tokens around the hyphen
        # If before is NOUN and after is NOUN/VERB, it's a compound like "self-understanding"
        # If before is ADV/ADJ and after is VERB, it's an adjective modifier like "ever-growing"
        last_before = before_hyphen[-1]
        first_after = after_hyphen[0]

        # "self-understanding" pattern: NOUN-NOUN or NOUN-VERB(gerund)
        if last_before.pos_ == "NOUN" and first_after.pos_ in ("NOUN", "VERB"):
            # Check if there are more tokens after the hyphenated sequence
            # If so, those would be separate head nouns
            if len(after_hyphen) > 1:
                # Check if remaining tokens are just the hyphenated continuation
                # vs separate head nouns
                remaining = after_hyphen[1:]
                if remaining and remaining[-1].pos_ in ("NOUN", "PROPN"):
                    # There's a separate noun after - this is "X-Y noun" pattern
                    return False
            return True

        # "ever-growing" pattern: ADV-VERB followed by a noun
        # This should NOT be a hyphenated compound - return False
        if last_before.pos_ in ("ADV", "ADJ") and first_after.pos_ == "VERB":
            return False

        return False

    def _normalize_phrase_text(self, tokens: list[Token]) -> str:
        """Normalize phrase text, handling hyphenation and spacing."""
        # Filter out articles
        filtered = [t for t in tokens if t.text.lower() not in self.ARTICLES]
        if not filtered:
            return ""

        parts = []
        for i, t in enumerate(filtered):
            if t.text == "-":
                # Keep hyphen attached
                if parts:
                    parts[-1] = parts[-1] + "-"
            elif t.text.startswith("-"):
                if parts:
                    parts[-1] = parts[-1] + t.text
                else:
                    parts.append(t.text)
            elif i > 0 and filtered[i-1].text == "-":
                if parts:
                    parts[-1] = parts[-1] + t.text
                else:
                    parts.append(t.text)
            else:
                parts.append(t.text)

        return " ".join(parts)

    def _process_of_pattern(
        self, tokens: list[Token], of_idx: int, sent_text: str
    ) -> ExtractedPhrase | None:
        """Process an "X of Y" pattern.

        In CiteSpace, "X of Y" → Y is the head and X is the modifier.
        E.g., "guarantor of peace" → head="peace", modifier="guarantor"

        Parameters
        ----------
        tokens : list[Token]
            All tokens in the phrase (already filtered)
        of_idx : int
            Index of "of" token
        sent_text : str
            Original sentence

        Returns
        -------
        ExtractedPhrase | None
            The processed phrase
        """
        before_of = tokens[:of_idx]
        after_of = tokens[of_idx + 1:]

        # Filter articles from after_of
        after_of = [t for t in after_of if t.text.lower() not in self.ARTICLES]

        if not after_of or not before_of:
            # Fallback to standard processing
            return self._process_standard_pattern(tokens, sent_text)

        # Find head in after_of part (rightmost noun/noun phrase)
        head_tokens = self._extract_head_sequence(after_of)
        modifier_tokens = before_of

        # Build head string
        head = self._build_concept_string(head_tokens)

        # Build modifier string
        modifier = self._build_concept_string(modifier_tokens)

        if not head:
            return None

        # Skip if head is a filter word
        if head.lower() in self.FILTER_WORDS:
            return None

        text = self._normalize_phrase_text(tokens)

        modifiers = [modifier] if modifier and modifier.lower() not in self.FILTER_WORDS else []

        return ExtractedPhrase(
            text=text,
            head=head,
            modifiers=modifiers,
            sentence=sent_text,
        )

    def _process_standard_pattern(
        self, tokens: list[Token], sent_text: str
    ) -> ExtractedPhrase | None:
        """Process a standard noun phrase (modifiers + head).

        Parameters
        ----------
        tokens : list[Token]
            Tokens in the phrase
        sent_text : str
            Original sentence

        Returns
        -------
        ExtractedPhrase | None
            The processed phrase or None
        """
        if not tokens:
            return None

        # Filter out pure determiners/pronouns from beginning
        while tokens and tokens[0].pos_ in ("DET", "PRON") and tokens[0].text.lower() in self.FILTER_WORDS:
            tokens = tokens[1:]

        if not tokens:
            return None

        # Find the head noun(s) - may be compound (e.g., "power europe")
        # Start from the end and collect consecutive nouns/proper nouns
        head_tokens = []
        head_start_idx = len(tokens)

        for i in range(len(tokens) - 1, -1, -1):
            t = tokens[i]
            if t.pos_ in ("NOUN", "PROPN"):
                head_tokens.insert(0, t)
                head_start_idx = i
            elif t.pos_ == "CCONJ" and head_tokens:
                # Include conjunction in compound head (e.g., "exception and concept")
                head_tokens.insert(0, t)
                head_start_idx = i
            elif head_tokens:
                # Stop when we hit non-noun/non-conjunction
                break

        if not head_tokens:
            return None

        # Build head string
        head = self._build_concept_string(head_tokens)

        # Skip if head is a filter word
        if head.lower() in self.FILTER_WORDS:
            return None

        # Modifiers are everything before the head
        modifiers = []
        modifier_tokens = tokens[:head_start_idx]

        # Collect modifiers:
        # - Individual adjectives become separate modifiers: "formidable global" → [formidable, global]
        # - Conjunction-linked items stay together: "destiny and limited" → [destiny and limited]
        # - Possessive + adjective stay together: "EU's numerous" → [eus numerous]
        # - Hyphenated compounds stay together: "ever-growing" → [ever-growing]

        i = 0
        current_group = []  # For grouping conjunction-linked or possessive+adj patterns
        has_conjunction = False
        has_possessive = False

        while i < len(modifier_tokens):
            t = modifier_tokens[i]

            # Skip articles
            if t.text.lower() in self.ARTICLES:
                i += 1
                continue

            # Skip pure filter words (pronouns/determiners)
            if t.text.lower() in self.FILTER_WORDS and t.pos_ in ("PRON", "DET"):
                i += 1
                continue

            # Handle possessive marker ('s) - mark that we should group
            # Handles both straight and curly apostrophes
            if t.pos_ == "PART" and t.text in ("'s", "'", "'s", "'"):
                has_possessive = True
                i += 1
                continue

            # Check for conjunction - triggers grouping
            if t.pos_ == "CCONJ":
                has_conjunction = True
                current_group.append(t.text.lower())
                i += 1
                continue

            # Check for hyphenated compound (e.g., "ever-growing")
            if t.pos_ in ("ADJ", "VERB", "ADV"):
                hyphenated = self._collect_hyphenated(modifier_tokens, i)
                if hyphenated:
                    mod_text = self._build_concept_string(hyphenated)
                    if mod_text and mod_text.lower() not in self.FILTER_WORDS:
                        if has_conjunction or has_possessive:
                            current_group.append(mod_text)
                        else:
                            modifiers.append(mod_text)
                    i += len(hyphenated)
                else:
                    mod_text = self._clean_token_text(t)
                    if mod_text and mod_text.lower() not in self.FILTER_WORDS:
                        if has_conjunction or has_possessive:
                            current_group.append(mod_text)
                        else:
                            modifiers.append(mod_text)
                    i += 1
            elif t.pos_ in ("NOUN", "PROPN"):
                # Noun/proper noun as modifier (e.g., "EU" in "EU's numerous")
                # Check if next token is possessive - if so, combine them
                if i + 1 < len(modifier_tokens) and modifier_tokens[i + 1].pos_ == "PART":
                    # Combine noun with 's' (e.g., "EU" + "'s" → "eus")
                    mod_text = t.text.lower() + "s"
                    if mod_text and mod_text not in self.FILTER_WORDS:
                        current_group.append(mod_text)
                        has_possessive = True
                    i += 2  # Skip both noun and possessive marker
                else:
                    mod_text = self._clean_token_text(t)
                    if mod_text and mod_text.lower() not in self.FILTER_WORDS:
                        if has_conjunction or has_possessive:
                            current_group.append(mod_text)
                        else:
                            modifiers.append(mod_text)
                    i += 1
            elif t.text == "-":
                i += 1  # Skip standalone hyphens
            elif t.pos_ == "PUNCT":
                i += 1  # Skip punctuation
            else:
                i += 1

        # Add any grouped modifiers
        if current_group:
            combined_modifier = " ".join(current_group)
            if combined_modifier and combined_modifier.lower() not in self.FILTER_WORDS:
                modifiers.append(combined_modifier)

        text = self._normalize_phrase_text(tokens)

        return ExtractedPhrase(
            text=text,
            head=head,
            modifiers=modifiers,
            sentence=sent_text,
        )

    def _collect_hyphenated(self, tokens: list[Token], start_idx: int) -> list[Token] | None:
        """Collect a hyphenated sequence starting at start_idx.

        Returns the tokens if a hyphenated sequence is found, else None.
        """
        if start_idx >= len(tokens):
            return None

        result = [tokens[start_idx]]
        i = start_idx + 1

        while i < len(tokens):
            if tokens[i].text == "-" and i + 1 < len(tokens):
                result.append(tokens[i])      # hyphen
                result.append(tokens[i + 1])  # next word
                i += 2
            elif tokens[i].text.startswith("-"):
                result.append(tokens[i])
                i += 1
            else:
                break

        return result if len(result) > 1 else None

    def _extract_head_sequence(self, tokens: list[Token]) -> list[Token]:
        """Extract the head noun sequence from tokens.

        This handles compound heads like "exception and concept" or "power europe".
        """
        # Find content tokens (nouns, proper nouns)
        content = []
        for t in tokens:
            if t.pos_ in ("NOUN", "PROPN"):
                content.append(t)
            elif t.pos_ == "CCONJ" and content:
                content.append(t)
            elif t.pos_ in ("DET", "ADP"):
                continue
            elif t.text.lower() in self.ARTICLES:
                continue

        # If we have multiple nouns with conjunction, include all
        return content if content else [tokens[-1]]

    def _build_concept_string(self, tokens: list[Token]) -> str:
        """Build a concept string from tokens, handling hyphenation."""
        if not tokens:
            return ""

        parts = []
        i = 0
        while i < len(tokens):
            t = tokens[i]

            # Skip articles
            if t.text.lower() in self.ARTICLES:
                i += 1
                continue

            # Handle hyphenation
            if t.text == "-":
                if parts and i + 1 < len(tokens):
                    # Attach hyphen to previous and next
                    next_text = self._clean_token_text(tokens[i + 1])
                    parts[-1] = parts[-1] + "-" + next_text
                    i += 2
                    continue
                i += 1
                continue

            text = self._clean_token_text(t)
            if text:
                parts.append(text)
            i += 1

        return " ".join(parts)

    def _clean_token_text(self, token: Token) -> str:
        """Clean token text for use in tree.

        Parameters
        ----------
        token : Token
            The token

        Returns
        -------
        str
            Cleaned text (lowercased, possessives normalized)
        """
        text = token.text.lower()
        # Remove possessive 's but keep the rest (handles both straight and curly apostrophes)
        text = re.sub(r"['']s$", "s", text)
        text = re.sub(r"['']$", "", text)
        return text

    def _add_phrase_to_tree(self, phrase: ExtractedPhrase, source: str) -> None:
        """Add an extracted phrase to the concept tree.

        Parameters
        ----------
        phrase : ExtractedPhrase
            The phrase to add
        source : str
            Source identifier
        """
        head = phrase.head
        context = phrase.sentence

        if not head or head in self.FILTER_WORDS:
            return

        if not phrase.modifiers:
            # Direct leaf under root
            if head not in self.root.children:
                self.root.children[head] = ConceptNode(name=head, source=source)
            self.root.children[head].increment_freq()
            self.root.children[head].add_context(context)
        else:
            # Branch with modifier children
            if head not in self.root.children:
                self.root.children[head] = ConceptNode(name=head, source=source)

            branch = self.root.children[head]
            branch.increment_freq()

            for modifier in phrase.modifiers:
                if modifier and modifier not in self.FILTER_WORDS:
                    if modifier not in branch.children:
                        branch.children[modifier] = ConceptNode(name=modifier, source=source)
                    branch.children[modifier].increment_freq()
                    branch.children[modifier].add_context(context)

    def to_treeml(self, output_path: Path | str, doc_title: str = "Untitled") -> None:
        """Export the concept tree to TreeML XML format.

        Parameters
        ----------
        output_path : Path | str
            Output file path
        doc_title : str
            Document title for context links
        """
        output_path = Path(output_path)

        # Create XML structure
        tree_elem = ET.Element("tree")

        # Add declarations
        declarations = ET.SubElement(tree_elem, "declarations")
        for attr_name, attr_type in [
            ("name", "String"),
            ("freq", "Int"),
            ("source", "String"),
            ("context", "String"),
        ]:
            attr_decl = ET.SubElement(declarations, "attributeDecl")
            attr_decl.set("name", attr_name)
            attr_decl.set("type", attr_type)

        # Add root branch
        root_branch = ET.SubElement(tree_elem, "branch")
        self._add_attributes(root_branch, self.root.name, self.root.freq, self.root.source, "")

        # Add children
        self._add_children_to_element(root_branch, self.root, doc_title)

        # Write XML with proper formatting
        self._write_xml(tree_elem, output_path)

    def _add_children_to_element(
        self, parent_elem: ET.Element, node: ConceptNode, doc_title: str
    ) -> None:
        """Add child nodes to an XML element.

        Parameters
        ----------
        parent_elem : ET.Element
            Parent XML element
        node : ConceptNode
            Parent concept node
        doc_title : str
            Document title
        """
        # Sort children: branches with children first, then leaves, by frequency desc
        children_items = list(node.children.items())

        for name, child in children_items:
            if child.children:
                # This is a branch (has children)
                branch_elem = ET.SubElement(parent_elem, "branch")
                # Branch may have its own context if it appears without modifiers too
                context_str = ""
                if child.contexts:
                    context_str = "null" + self._format_context(child.contexts, doc_title)
                self._add_attributes(branch_elem, child.name, child.freq, child.source, context_str)
                # Add modifier children as leaves
                for mod_name, mod_node in child.children.items():
                    leaf_elem = ET.SubElement(branch_elem, "leaf")
                    mod_context = self._format_context(mod_node.contexts, doc_title)
                    self._add_attributes(
                        leaf_elem, mod_node.name, mod_node.freq, mod_node.source, mod_context
                    )
            else:
                # This is a leaf (no children)
                leaf_elem = ET.SubElement(parent_elem, "leaf")
                context_str = self._format_context(child.contexts, doc_title)
                self._add_attributes(leaf_elem, child.name, child.freq, child.source, context_str)

    def _add_attributes(
        self, elem: ET.Element, name: str, freq: int, source: str, context: str
    ) -> None:
        """Add attribute elements to a tree node.

        Parameters
        ----------
        elem : ET.Element
            The element to add attributes to
        name : str
            Node name
        freq : int
            Frequency count
        source : str
            Source identifier
        context : str
            Context string (HTML)
        """
        for attr_name, attr_value in [
            ("name", name),
            ("freq", str(freq)),
            ("source", source),
            ("context", context),
        ]:
            attr_elem = ET.SubElement(elem, "attribute")
            attr_elem.set("name", attr_name)
            attr_elem.set("value", attr_value)

    def _format_context(self, contexts: list[str], doc_title: str) -> str:
        """Format context sentences as HTML list.

        Parameters
        ----------
        contexts : list[str]
            List of context sentences
        doc_title : str
            Document title

        Returns
        -------
        str
            HTML-formatted context string
        """
        if not contexts:
            return ""

        items = []
        for ctx in contexts:
            # Escape special characters
            escaped_ctx = ctx.replace("'", "'")
            # Format as list item with link
            link = f'<a href="file://null\\{doc_title}">{doc_title}</a>'
            items.append(f"<li>[{link}] {escaped_ctx}</li>")

        return f'<ul>{"".join(items)}</ul>\n'

    def _write_xml(self, tree_elem: ET.Element, output_path: Path) -> None:
        """Write XML tree to file with proper formatting.

        Parameters
        ----------
        tree_elem : ET.Element
            Root XML element
        output_path : Path
            Output file path
        """
        # Generate timestamp comment
        timestamp = datetime.now().strftime("%a %b %d %H:%M:%S CET %Y")
        comment = f"<!-- prefuse TreeML Writer | {timestamp} -->"

        # Add XML declaration and comment
        output = f'<?xml version="1.0" encoding="UTF-8"?>\n{comment}\n'

        # Pretty print the XML
        output += self._pretty_print_xml(tree_elem)

        output_path.write_text(output, encoding="utf-8")

    def _pretty_print_xml(self, elem: ET.Element, level: int = 0) -> str:
        """Pretty print XML element with proper indentation.

        Parameters
        ----------
        elem : ET.Element
            Element to print
        level : int
            Indentation level

        Returns
        -------
        str
            Formatted XML string
        """
        indent = "  " * level
        result = f"{indent}<{elem.tag}"

        # Add attributes
        for key, value in elem.attrib.items():
            # Escape for XML attribute
            escaped_value = (value
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#8216;"))
            result += f' {key}="{escaped_value}"'

        if len(elem) == 0:
            result += "/>\n"
        else:
            result += ">\n"
            for child in elem:
                result += self._pretty_print_xml(child, level + 1)
            result += f"{indent}</{elem.tag}>\n"

        return result


def build_concept_tree(
    text: str,
    output_path: Path | str | None = None,
    source: str = "p",
    doc_title: str = "Untitled",
    nlp: spacy.language.Language | None = None,
) -> ConceptTreeBuilder:
    """Convenience function to build a concept tree from text.

    Parameters
    ----------
    text : str
        Input text
    output_path : Path | str | None
        If provided, write TreeML XML to this path
    source : str
        Source identifier
    doc_title : str
        Document title
    nlp : spacy.language.Language | None
        spaCy model (loads en_core_web_sm if not provided)

    Returns
    -------
    ConceptTreeBuilder
        The builder with the constructed tree
    """
    builder = ConceptTreeBuilder(nlp)
    builder.process_text(text, source=source, doc_title=doc_title)

    if output_path:
        builder.to_treeml(output_path, doc_title=doc_title)

    return builder


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        text = input_path.read_text(encoding="utf-8")
    else:
        text = """The Russian war against Ukraine presents the most serious challenge
        to the European Union and its self-understanding."""

    output_path = Path(__file__).parent / "output_concept_tree.xml"
    builder = build_concept_tree(text, output_path)

    print(f"Concept tree written to: {output_path}")
    print(f"Number of concepts: {len(builder.root.children)}")
