"""Concept tree builder following CiteSpace's approach.

This module extracts noun phrases and builds hierarchical concept trees where:
- Head nouns become parent nodes
- Modifiers become child nodes
- Phrases sharing the same head are grouped together
- Each phrase contributes to exactly ONE location (no duplication)

Uses the `terms` module for POS-pattern-based phrase extraction,
supporting multiple languages (en, de, ru, uk).

Based on Chen (2012) "Turning Points: The Nature of Creativity" Chapter 7.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import spacy
from spacy.tokens import Doc, Span, Token

# Import the terms module for phrase extraction
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from terms import ExtractionScope, TermsMatcher


@dataclass
class ExtractedPhrase:
    """A noun phrase with its head and modifiers identified."""

    text: str  # Original phrase text
    head: str  # Head noun (surface form)
    modifiers: list[str]  # Modifier words (surface forms)
    sentence: str = ""  # Source sentence for context
    doc_index: int = 0  # Document index
    pos_label: str = ""  # POS pattern label (e.g., "ADJ-NOUN")
    is_proper_noun_unit: bool = False  # Whether this is a proper noun sequence


@dataclass
class ConceptNode:
    """A node in the concept tree."""

    name: str
    frequency: int = 0
    contexts: list[str] = field(default_factory=list)
    children: dict[str, "ConceptNode"] = field(default_factory=dict)
    source: str = "p"  # Source identifier

    def add_context(self, context: str) -> None:
        """Add a context sentence if not already present."""
        if context and context not in self.contexts:
            self.contexts.append(context)

    def get_or_create_child(self, name: str) -> "ConceptNode":
        """Get existing child or create new one."""
        if name not in self.children:
            self.children[name] = ConceptNode(name=name)
        return self.children[name]


# Language-specific spaCy models
LANG_MODELS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "ru": "ru_core_news_sm",
    "uk": "uk_core_news_sm",
}


class ConceptTreeBuilder:
    """Builds hierarchical concept trees from text.

    The tree construction follows CiteSpace's approach:
    1. Extract noun phrases via POS pattern matching (using terms module)
    2. Use redundancy resolution to keep only maximal spans
    3. Identify head noun (rightmost noun in most languages)
    4. Group phrases by head noun
    5. Modifiers become children of their head
    6. Each phrase contributes to exactly ONE location (no duplication)

    Example:
        "Russian war" → head="war", modifier="russian"
        "times of war" → head="war", modifier="times"
        Both grouped under "war" parent node.
    """

    # Prepositions that connect noun phrases (for "X of Y" pattern detection)
    CONNECTORS = {
        "en": {"of", "for", "in", "on", "to", "with", "by", "from", "at"},
        "de": {"von", "für", "in", "an", "auf", "mit", "bei", "aus", "nach"},
        "ru": {"в", "на", "с", "от", "до", "для", "по", "из", "о", "об"},
        "uk": {"в", "на", "з", "від", "до", "для", "по", "із", "про"},
    }

    # Tokens to always filter out from modifiers
    FILTER_TOKENS = {"'s", "'s", "its", "their", "our", "your", "his", "her"}

    def __init__(
        self,
        lang: str = "en",
        nlp: spacy.language.Language | None = None,
        lemmatize: bool = False,  # Default to False to preserve surface forms
        scope: ExtractionScope = ExtractionScope.SENTENCE,
    ):
        """Initialize the concept tree builder.

        Parameters
        ----------
        lang : str
            Language code: 'en', 'de', 'ru', 'uk'
        nlp : spacy.language.Language | None
            Optional pre-loaded spaCy model. If None, loads based on lang.
        lemmatize : bool
            Whether to lemmatize tokens. Default False to preserve surface forms
            like "capabilities", "growing", "deficiencies".
        scope : ExtractionScope
            Extraction scope for TermsMatcher (default SENTENCE for full coverage).
        """
        self.lang = lang
        self.lemmatize = lemmatize
        self.scope = scope

        # Load spaCy model
        if nlp is None:
            model_name = LANG_MODELS.get(lang, LANG_MODELS["en"])
            nlp = spacy.load(model_name)
        self.nlp = nlp

        # Initialize TermsMatcher for phrase extraction
        self.terms_matcher = TermsMatcher(nlp=self.nlp)

        # Root of the concept tree
        self.root = ConceptNode(name="Concepts", frequency=1)

        # Get language-specific connectors
        self._connectors = self.CONNECTORS.get(lang, self.CONNECTORS["en"])

    def _get_token_text(self, token: Token) -> str:
        """Get the text representation of a token."""
        if self.lemmatize:
            return token.lemma_.lower()
        return token.text.lower()

    def _is_content_pos(self, token: Token) -> bool:
        """Check if token has a content POS tag."""
        allowed_pos = {"NOUN", "PROPN", "ADJ", "NUM"}
        if token.pos_ in allowed_pos:
            return True
        # Allow participles (VERB with VerbForm=Part)
        if token.pos_ == "VERB":
            morph = token.morph.to_dict()
            if morph.get("VerbForm", "") == "Part":
                return True
        return False

    def _is_content_token(self, token: Token) -> bool:
        """Check if token is valid content for the concept tree."""
        if not self._is_content_pos(token):
            return False
        # Filter out specific tokens
        text_lower = token.text.lower()
        if text_lower in self._connectors:
            return False
        if text_lower in self.FILTER_TOKENS:
            return False
        # Filter out possessive markers
        if token.tag_ in ("POS", "PRP$", "WP$"):
            return False
        return True

    def _is_proper_noun_sequence(self, tokens: list[Token]) -> bool:
        """Check if tokens form a proper noun sequence (e.g., 'European Union')."""
        if len(tokens) < 2:
            return False
        # All tokens should be PROPN or capitalized content
        propn_count = sum(1 for t in tokens if t.pos_ == "PROPN")
        # At least half should be PROPN for it to be a proper noun sequence
        return propn_count >= len(tokens) / 2 and all(
            t.pos_ == "PROPN" or (t.pos_ == "ADJ" and t.text[0].isupper())
            for t in tokens
        )

    def _find_head_noun(self, tokens: list[Token]) -> Token | None:
        """Find the head noun of a phrase.

        For most noun phrases, the head is the rightmost noun.
        For "X of Y" patterns, Y is the semantic head.
        """
        if not tokens:
            return None

        # Check for "X of/von/... Y" pattern
        for i, token in enumerate(tokens):
            if token.text.lower() in self._connectors and i > 0 and i < len(tokens) - 1:
                # Find rightmost noun after the connector
                for j in range(len(tokens) - 1, i, -1):
                    if tokens[j].pos_ in ("NOUN", "PROPN"):
                        return tokens[j]

        # Default: rightmost noun
        for token in reversed(tokens):
            if token.pos_ in ("NOUN", "PROPN"):
                return token

        return None

    def _decompose_phrase(
        self,
        doc: Doc,
        start: int,
        end: int,
        sentence_text: str,
        pos_label: str,
    ) -> ExtractedPhrase | None:
        """Decompose a noun phrase span into head and modifiers.

        Parameters
        ----------
        doc : Doc
            The spaCy document
        start, end : int
            Token indices for the span
        sentence_text : str
            The source sentence for context
        pos_label : str
            The POS pattern label (e.g., "ADJ-NOUN")

        Returns
        -------
        ExtractedPhrase | None
            Decomposed phrase, or None if invalid
        """
        span = doc[start:end]

        # Filter to content tokens only
        tokens = [t for t in span if self._is_content_token(t)]

        if not tokens:
            return None

        # Check for proper noun sequences - keep as single unit, no decomposition
        if self._is_proper_noun_sequence(tokens):
            head = " ".join(self._get_token_text(t) for t in tokens)
            return ExtractedPhrase(
                text=span.text,
                head=head,
                modifiers=[],
                sentence=sentence_text,
                pos_label=pos_label,
                is_proper_noun_unit=True,
            )

        # Find head noun
        head_token = self._find_head_noun(tokens)
        if head_token is None:
            return None

        head = self._get_token_text(head_token)

        # Modifiers are other content tokens (excluding the head)
        modifiers = []
        for t in tokens:
            if t == head_token:
                continue
            token_text = self._get_token_text(t)
            if token_text != head:  # Skip if same as head
                modifiers.append(token_text)

        return ExtractedPhrase(
            text=span.text,
            head=head,
            modifiers=modifiers,
            sentence=sentence_text,
            pos_label=pos_label,
            is_proper_noun_unit=False,
        )

    def add_phrase_to_tree(self, phrase: ExtractedPhrase) -> None:
        """Add a decomposed phrase to the concept tree.

        For proper noun units: add as leaf directly under root
        For regular phrases: head becomes branch, modifiers become children
        """
        # Proper noun sequences become leaves with no children
        if phrase.is_proper_noun_unit:
            node = self.root.get_or_create_child(phrase.head)
            node.frequency += 1
            node.add_context(phrase.sentence)
            return

        # Regular phrases: head is branch, modifiers are children
        head_node = self.root.get_or_create_child(phrase.head)
        head_node.frequency += 1
        head_node.add_context(phrase.sentence)

        # Add each modifier as a separate child
        for modifier in phrase.modifiers:
            modifier_node = head_node.get_or_create_child(modifier)
            modifier_node.frequency += 1
            modifier_node.add_context(phrase.sentence)

    def _get_phrase_key(self, phrase: ExtractedPhrase) -> tuple:
        """Get a unique key for a phrase based on its decomposed form."""
        return (phrase.head, tuple(sorted(phrase.modifiers)))

    def _is_valid_phrase(self, doc: Doc, start: int, end: int) -> bool:
        """Check if a span represents a valid complete phrase.

        Filters out incomplete phrases like "challenge to the European"
        where the final token is an adjective without its noun.
        """
        span = doc[start:end]
        if not span:
            return False

        # Get the last content token
        last_content = None
        for token in reversed(list(span)):
            if token.pos_ in ("NOUN", "PROPN", "ADJ", "NUM"):
                last_content = token
                break
            if token.pos_ == "VERB":
                morph = token.morph.to_dict()
                if morph.get("VerbForm", "") == "Part":
                    last_content = token
                    break

        if last_content is None:
            return False

        # Reject if ends with an adjective that typically needs a noun
        # (like "European" without "Union")
        if last_content.pos_ == "ADJ" and last_content.text[0].isupper():
            # Check if there's a noun immediately after this span in the doc
            if end < len(doc) and doc[end].pos_ in ("NOUN", "PROPN"):
                return False  # Incomplete phrase

        return True

    def process_text(self, text: str, doc_index: int = 0) -> list[ExtractedPhrase]:
        """Process text and extract phrases into the concept tree.

        Strategy:
        1. Extract all maximal spans
        2. Filter out invalid/incomplete phrases
        3. Decompose each into (head, modifiers)
        4. Aggregate by decomposed form - same (head, modifiers) = same phrase type
        5. All occurrences of same phrase contribute to its frequency

        Parameters
        ----------
        text : str
            Input text to process
        doc_index : int
            Document index for tracking

        Returns
        -------
        list[ExtractedPhrase]
            All extracted phrases
        """
        # Clean text
        text = text.replace("'", "'").replace("'", "'")
        text = text.replace(""", '"').replace(""", '"')

        doc = self.nlp(text)

        # Use extract_key_phrases with redundancy resolution
        results = self.terms_matcher.extract_key_phrases(
            text,
            scope=self.scope,
            exclusive_search=False,
            resolve_redundancy=True,
        )

        # Build sentence lookup for context
        sent_map = {}
        for sent in doc.sents:
            for i in range(sent.start, sent.end):
                sent_map[i] = sent.text.strip()

        # Process maximal spans and aggregate by phrase decomposition
        phrase_occurrences: dict[tuple, list[ExtractedPhrase]] = {}
        seen_spans = set()

        for result in results:
            if not result.get("is_maximal", False):
                continue

            start, end = result["token_span"]
            span_key = (start, end)

            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)

            # Filter out invalid phrases
            if not self._is_valid_phrase(doc, start, end):
                continue

            sentence_text = sent_map.get(start, "")
            phrase = self._decompose_phrase(
                doc, start, end, sentence_text, result["pos_label"]
            )

            if phrase:
                phrase.doc_index = doc_index
                key = self._get_phrase_key(phrase)

                if key not in phrase_occurrences:
                    phrase_occurrences[key] = []
                phrase_occurrences[key].append(phrase)

        # Add each unique phrase type to the tree (with aggregated frequency)
        all_phrases = []
        for key, occurrences in phrase_occurrences.items():
            # Use the first occurrence as the representative
            phrase = occurrences[0]
            all_phrases.append(phrase)

            # Add to tree with correct frequency
            self._add_phrase_to_tree_with_count(phrase, len(occurrences))

        return all_phrases

    def _add_phrase_to_tree_with_count(self, phrase: ExtractedPhrase, count: int) -> None:
        """Add a phrase to the tree with specified occurrence count."""
        # Proper noun sequences become leaves with no children
        if phrase.is_proper_noun_unit:
            node = self.root.get_or_create_child(phrase.head)
            node.frequency += count
            node.add_context(phrase.sentence)
            return

        # Regular phrases: head is branch, modifiers are children
        head_node = self.root.get_or_create_child(phrase.head)
        head_node.frequency += count
        head_node.add_context(phrase.sentence)

        # Add each modifier as a separate child
        for modifier in phrase.modifiers:
            modifier_node = head_node.get_or_create_child(modifier)
            modifier_node.frequency += count
            modifier_node.add_context(phrase.sentence)

    def to_treeml_xml(self, source_name: str = "Untitled") -> str:
        """Generate TreeML XML output compatible with prefuse/CiteSpace."""
        tree = ET.Element("tree")

        # Declarations
        declarations = ET.SubElement(tree, "declarations")
        for attr_name, attr_type in [
            ("name", "String"),
            ("freq", "Int"),
            ("source", "String"),
            ("context", "String"),
        ]:
            attr_decl = ET.SubElement(declarations, "attributeDecl")
            attr_decl.set("name", attr_name)
            attr_decl.set("type", attr_type)

        def format_context(contexts: list[str], source: str) -> str:
            if not contexts:
                return ""
            items = []
            for ctx in contexts[:10]:
                escaped = ctx.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                items.append(f'<li>[<a href="file://null\\{source}">{source}</a>] {escaped}</li>')
            return "<ul>" + "".join(items) + "</ul>\n"

        def add_node(parent_elem: ET.Element, node: ConceptNode, is_root: bool = False) -> None:
            has_children = bool(node.children)
            tag = "branch" if has_children or is_root else "leaf"
            elem = ET.SubElement(parent_elem, tag)

            name_attr = ET.SubElement(elem, "attribute")
            name_attr.set("name", "name")
            name_attr.set("value", node.name)

            freq_attr = ET.SubElement(elem, "attribute")
            freq_attr.set("name", "freq")
            freq_attr.set("value", str(node.frequency))

            source_attr = ET.SubElement(elem, "attribute")
            source_attr.set("name", "source")
            source_attr.set("value", node.source)

            if node.contexts or not has_children:
                ctx_attr = ET.SubElement(elem, "attribute")
                ctx_attr.set("name", "context")
                ctx_attr.set("value", format_context(node.contexts, source_name))

            sorted_children = sorted(
                node.children.values(),
                key=lambda n: (-n.frequency, n.name),
            )
            for child in sorted_children:
                add_node(elem, child)

        add_node(tree, self.root, is_root=True)
        ET.indent(tree, space="  ")

        xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_str += f'<!-- prefuse TreeML Writer | {datetime.now().strftime("%c")} -->\n'
        xml_str += ET.tostring(tree, encoding="unicode")
        return xml_str

    def print_tree(self, node: ConceptNode | None = None, indent: int = 0) -> None:
        """Print the concept tree for debugging."""
        if node is None:
            node = self.root

        prefix = "  " * indent
        freq_str = f" ({node.frequency})" if node.frequency > 0 else ""
        print(f"{prefix}{node.name}{freq_str}")

        for child in sorted(node.children.values(), key=lambda n: (-n.frequency, n.name)):
            self.print_tree(child, indent + 1)


def build_concept_tree(
    text: str,
    lang: str = "en",
    nlp: spacy.language.Language | None = None,
    lemmatize: bool = False,
    scope: ExtractionScope = ExtractionScope.SENTENCE,
) -> ConceptTreeBuilder:
    """Convenience function to build a concept tree from text.

    Parameters
    ----------
    text : str
        Input text
    lang : str
        Language code: 'en', 'de', 'ru', 'uk'
    nlp : spacy.language.Language | None
        Optional pre-loaded spaCy model
    lemmatize : bool
        Whether to lemmatize tokens. Default False to preserve surface forms.
    scope : ExtractionScope
        Extraction scope for phrase matching

    Returns
    -------
    ConceptTreeBuilder
        The builder with populated tree
    """
    builder = ConceptTreeBuilder(
        lang=lang,
        nlp=nlp,
        lemmatize=lemmatize,
        scope=scope,
    )
    builder.process_text(text)
    return builder


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build concept tree from text file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported languages: en (English), de (German), ru (Russian), uk (Ukrainian)

Examples:
  python -m concept_tree.concept_tree input.txt --print
  python -m concept_tree.concept_tree input.txt --lang de -o output.xml
  python -m concept_tree.concept_tree input.txt --lemma --print
        """,
    )
    parser.add_argument("input_file", type=Path, help="Input text file")
    parser.add_argument("-o", "--output", type=Path, help="Output XML file")
    parser.add_argument(
        "--lang",
        choices=["en", "de", "ru", "uk"],
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument("--print", action="store_true", help="Print tree to console")
    parser.add_argument(
        "--lemma",
        action="store_true",
        help="Lemmatize tokens (default: keep surface forms)",
    )
    parser.add_argument(
        "--scope",
        choices=["subject", "object", "sentence"],
        default="sentence",
        help="Extraction scope (default: sentence)",
    )

    args = parser.parse_args()

    # Read input
    text = args.input_file.read_text(encoding="utf-8")

    # Map scope string to enum
    scope_map = {
        "subject": ExtractionScope.SUBJECT,
        "object": ExtractionScope.OBJECT,
        "sentence": ExtractionScope.SENTENCE,
    }

    # Build tree
    print(f"Processing: {args.input_file} (lang={args.lang})")
    builder = build_concept_tree(
        text,
        lang=args.lang,
        lemmatize=args.lemma,
        scope=scope_map[args.scope],
    )

    # Print if requested
    if args.print:
        print("\nConcept Tree:")
        print("=" * 50)
        builder.print_tree()
        print("=" * 50)

    # Generate XML
    source_name = args.input_file.stem
    xml_output = builder.to_treeml_xml(source_name=source_name)

    # Write output
    output_path = args.output or args.input_file.with_suffix(".xml")
    output_path.write_text(xml_output, encoding="utf-8")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
