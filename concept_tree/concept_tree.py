"""Concept tree builder following CiteSpace's approach.

This module extracts noun phrases and builds hierarchical concept trees where:
- Head nouns become parent nodes
- Modifiers become child nodes
- Phrases sharing the same head are grouped together

Based on Chen (2012) "Turning Points: The Nature of Creativity" Chapter 7.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import spacy
from spacy.tokens import Doc, Span, Token


@dataclass
class ExtractedPhrase:
    """A noun phrase with its head and modifiers identified."""

    text: str  # Original phrase text
    head: str  # Head noun (lemmatized, lowercase)
    modifiers: list[str]  # Modifier words (lemmatized, lowercase)
    head_token: Token | None = None  # The actual head token
    sentence: str = ""  # Source sentence for context
    doc_index: int = 0  # Document index


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


class ConceptTreeBuilder:
    """Builds hierarchical concept trees from text.

    The tree construction follows CiteSpace's approach:
    1. Extract noun phrases via POS pattern matching
    2. Identify head noun (rightmost noun in English)
    3. Group phrases by head noun
    4. Modifiers become children of their head

    Example:
        "Russian war" → head="war", modifier="russian"
        "times of war" → head="war", modifier="times"
        Both grouped under "war" parent node.
    """

    # Articles to strip
    ARTICLES = {"a", "an", "the"}

    # Prepositions that connect noun phrases
    CONNECTORS = {"of", "for", "in", "on", "to", "with", "by", "from", "at"}

    # Tokens to always filter out
    FILTER_TOKENS = {"'s", "'s", "its", "their", "our", "your", "his", "her"}

    def __init__(
        self,
        nlp: spacy.language.Language | None = None,
        min_phrase_words: int = 2,
        max_phrase_words: int = 4,
        lemmatize: bool = True,
    ):
        """Initialize the concept tree builder.

        Parameters
        ----------
        nlp : spacy.language.Language | None
            spaCy model to use. Defaults to en_core_web_sm.
        min_phrase_words : int
            Minimum words in a phrase (default 2, as in CiteSpace).
        max_phrase_words : int
            Maximum words in a phrase (default 4, as in CiteSpace).
        lemmatize : bool
            Whether to lemmatize tokens (default True). Set to False to preserve
            surface forms like "narratives" instead of "narrative".
        """
        if nlp is None:
            nlp = spacy.load("en_core_web_sm")
        self.nlp = nlp
        self.min_phrase_words = min_phrase_words
        self.max_phrase_words = max_phrase_words
        self.lemmatize = lemmatize

        # Root of the concept tree
        self.root = ConceptNode(name="Concepts", frequency=1)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove smart quotes and normalize
        text = text.replace("'", "'").replace("'", "'")
        text = text.replace(""", '"').replace(""", '"')
        return text

    def _get_head_noun(self, span: Span) -> Token | None:
        """Find the head noun of a noun phrase span.

        In English NPs, the head is typically:
        1. The syntactic root of the span (if it's a noun)
        2. Otherwise, the rightmost noun

        Parameters
        ----------
        span : Span
            A spaCy Span representing a noun phrase

        Returns
        -------
        Token | None
            The head noun token, or None if not found
        """
        # First, try to find the syntactic root that's a noun
        for token in span:
            if token.head not in span or token == span.root:
                if token.pos_ in ("NOUN", "PROPN"):
                    return token

        # Fallback: rightmost noun (English is head-final for NPs)
        for token in reversed(list(span)):
            if token.pos_ in ("NOUN", "PROPN"):
                return token

        return None

    def _is_proper_noun_sequence(self, span: Span) -> bool:
        """Check if span is a proper noun sequence (e.g., 'European Union')."""
        content_tokens = [t for t in span if t.pos_ not in ("DET", "PUNCT")]
        if len(content_tokens) < 2:
            return False
        # Check if all content words are PROPN or capitalized ADJ + PROPN
        return all(t.pos_ == "PROPN" or (t.pos_ == "ADJ" and t.text[0].isupper())
                   for t in content_tokens)

    def _is_content_token(self, token: Token) -> bool:
        """Check if token is valid content for the concept tree."""
        # Filter out by POS - only allow NOUN, PROPN, ADJ, and participles
        allowed_pos = {"NOUN", "PROPN", "ADJ"}
        if token.pos_ not in allowed_pos:
            # Special case: allow participles (VERB with VerbForm=Part)
            if token.pos_ == "VERB":
                morph = token.morph.to_dict()
                verb_form = morph.get("VerbForm", "")
                # Allow both past and present participles when used adjectivally
                # e.g., "limited capabilities", "growing ambition"
                if verb_form == "Part":
                    return True
            return False

        # Filter out by text
        text_lower = token.text.lower()
        if text_lower in self.ARTICLES:
            return False
        if text_lower in self.CONNECTORS:
            return False
        if text_lower in self.FILTER_TOKENS:
            return False

        # Filter out possessive markers
        if token.tag_ in ("POS", "PRP$", "WP$"):
            return False

        return True

    def _get_token_text(self, token: Token) -> str:
        """Get the text representation of a token (lemma or surface form)."""
        if self.lemmatize:
            return token.lemma_.lower()
        else:
            return token.text.lower()

    def _find_of_pattern_head(self, span: Span) -> Token | None:
        """Handle 'X of Y' patterns where Y is the semantic head.

        In CiteSpace, 'guarantor of peace' has head='peace'.
        Returns the semantic head token for such patterns.
        """
        tokens = list(span)
        # Look for "of" in the span
        for i, token in enumerate(tokens):
            if token.text.lower() == "of" and i > 0 and i < len(tokens) - 1:
                # Find rightmost noun after "of"
                for j in range(len(tokens) - 1, i, -1):
                    if tokens[j].pos_ in ("NOUN", "PROPN"):
                        return tokens[j]
        return None

    def _decompose_phrase(self, span: Span, sentence: str) -> ExtractedPhrase | None:
        """Decompose a noun phrase into head and modifiers.

        Parameters
        ----------
        span : Span
            A spaCy Span representing a noun phrase
        sentence : str
            The source sentence for context

        Returns
        -------
        ExtractedPhrase | None
            Decomposed phrase, or None if invalid
        """
        # Filter to content tokens only
        tokens = [t for t in span if self._is_content_token(t)]

        if not tokens:
            return None

        # Check for proper noun sequences - keep as single unit
        if self._is_proper_noun_sequence(span):
            # Use full phrase as head, no modifiers
            head = " ".join(self._get_token_text(t) for t in tokens)
            return ExtractedPhrase(
                text=span.text,
                head=head,
                modifiers=[],
                head_token=tokens[-1],
                sentence=sentence,
            )

        # Check for "X of Y" pattern - semantic head is Y
        of_head = self._find_of_pattern_head(span)

        # Find head noun
        if of_head is not None:
            head_token = of_head
        else:
            head_token = self._get_head_noun(span)

        if head_token is None:
            return None

        # Head is the token text (lemmatized or surface form based on setting)
        head = self._get_token_text(head_token)

        # Modifiers are other content tokens (excluding the head)
        modifiers = []
        for t in tokens:
            if t == head_token:
                continue

            token_text = self._get_token_text(t)

            # Skip if same as head (can happen with different surface forms)
            if token_text == head:
                continue

            modifiers.append(token_text)

        return ExtractedPhrase(
            text=span.text,
            head=head,
            modifiers=modifiers,
            head_token=head_token,
            sentence=sentence,
        )

    def _extract_noun_phrases(self, doc: Doc) -> Iterator[Span]:
        """Extract noun phrases from a document using spaCy's noun_chunks.

        Parameters
        ----------
        doc : Doc
            A spaCy Doc

        Yields
        ------
        Span
            Noun phrase spans
        """
        for chunk in doc.noun_chunks:
            # Filter by word count (excluding articles/det)
            content_tokens = [t for t in chunk if t.pos_ not in ("DET", "PUNCT")]
            word_count = len(content_tokens)

            if self.min_phrase_words <= word_count <= self.max_phrase_words:
                yield chunk

    def _extract_pattern_phrases(self, doc: Doc) -> Iterator[Span]:
        """Extract noun phrases using pattern matching for better coverage.

        This captures phrases that noun_chunks might miss, especially:
        - Compound nouns
        - Adjective chains
        - Proper noun sequences
        - X of Y patterns

        Parameters
        ----------
        doc : Doc
            A spaCy Doc

        Yields
        ------
        Span
            Noun phrase spans
        """
        i = 0
        while i < len(doc):
            token = doc[i]

            # Start of potential NP: ADJ, NOUN, PROPN only (no verbs at start)
            if token.pos_ in ("ADJ", "NOUN", "PROPN"):
                # Extend to find full phrase
                j = i + 1
                has_of = False
                while j < len(doc):
                    next_token = doc[j]
                    # Continue if: ADJ, NOUN, PROPN
                    if next_token.pos_ in ("ADJ", "NOUN", "PROPN"):
                        j += 1
                    elif next_token.text.lower() == "of" and j + 1 < len(doc) and not has_of:
                        # "X of Y" pattern - only one "of" allowed
                        following = doc[j + 1]
                        if following.pos_ in ("NOUN", "PROPN", "DET"):
                            j += 1
                            has_of = True
                        else:
                            break
                    elif next_token.pos_ == "DET" and has_of and j + 1 < len(doc):
                        # Skip determiner after "of" ("X of the Y")
                        j += 1
                    else:
                        break

                # Check if we found a valid phrase
                span = doc[i:j]
                content_tokens = [t for t in span if self._is_content_token(t)]

                # Must end with noun and meet length requirements
                if (len(content_tokens) >= self.min_phrase_words and
                    len(content_tokens) <= self.max_phrase_words and
                    span[-1].pos_ in ("NOUN", "PROPN")):
                    yield span
                    i = j
                    continue

            i += 1

    def add_phrase_to_tree(self, phrase: ExtractedPhrase) -> None:
        """Add a decomposed phrase to the concept tree.

        The head noun becomes a branch under root (or connects to existing).
        Modifiers become children of the head - each modifier as separate child.

        Parameters
        ----------
        phrase : ExtractedPhrase
            The decomposed phrase to add
        """
        # Get or create head node under root
        head_node = self.root.get_or_create_child(phrase.head)
        head_node.frequency += 1
        head_node.add_context(phrase.sentence)

        # Add each modifier as a separate child (CiteSpace style)
        for modifier in phrase.modifiers:
            modifier_node = head_node.get_or_create_child(modifier)
            modifier_node.frequency += 1
            modifier_node.add_context(phrase.sentence)

    def process_text(self, text: str, doc_index: int = 0) -> list[ExtractedPhrase]:
        """Process text and extract phrases into the concept tree.

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
        text = self._clean_text(text)
        doc = self.nlp(text)

        phrases = []
        seen_spans = set()  # Track (start, end) to avoid duplicates

        # Process each sentence
        for sent in doc.sents:
            sentence_text = sent.text.strip()

            # Extract from noun_chunks
            for chunk in sent.noun_chunks:
                content_tokens = [t for t in chunk if t.pos_ not in ("DET", "PUNCT")]
                word_count = len(content_tokens)

                if self.min_phrase_words <= word_count <= self.max_phrase_words:
                    span_key = (chunk.start, chunk.end)
                    if span_key not in seen_spans:
                        seen_spans.add(span_key)
                        phrase = self._decompose_phrase(chunk, sentence_text)
                        if phrase:
                            phrase.doc_index = doc_index
                            phrases.append(phrase)
                            self.add_phrase_to_tree(phrase)

            # Also try pattern-based extraction for better coverage
            sent_doc = sent.as_doc()
            for span in self._extract_pattern_phrases(sent_doc):
                # Map back to original doc indices
                orig_start = sent.start + span.start
                orig_end = sent.start + span.end
                span_key = (orig_start, orig_end)

                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    # Get the actual span from the original doc
                    orig_span = doc[orig_start:orig_end]
                    phrase = self._decompose_phrase(orig_span, sentence_text)
                    if phrase:
                        phrase.doc_index = doc_index
                        phrases.append(phrase)
                        self.add_phrase_to_tree(phrase)

        return phrases

    def to_treeml_xml(self, source_name: str = "Untitled") -> str:
        """Generate TreeML XML output compatible with prefuse/CiteSpace.

        Parameters
        ----------
        source_name : str
            Name of the source document

        Returns
        -------
        str
            TreeML XML string
        """
        # Create XML structure
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
            """Format contexts as HTML list."""
            if not contexts:
                return ""
            items = []
            for ctx in contexts[:10]:  # Limit contexts
                escaped = ctx.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                items.append(
                    f'<li>[<a href="file://null\\{source}">{source}</a>] {escaped}</li>'
                )
            return "<ul>" + "".join(items) + "</ul>\n"

        def add_node(parent_elem: ET.Element, node: ConceptNode, is_root: bool = False) -> None:
            """Recursively add nodes to XML."""
            # Determine if branch or leaf
            has_children = bool(node.children)
            tag = "branch" if has_children or is_root else "leaf"

            elem = ET.SubElement(parent_elem, tag)

            # Add attributes
            name_attr = ET.SubElement(elem, "attribute")
            name_attr.set("name", "name")
            name_attr.set("value", node.name)

            freq_attr = ET.SubElement(elem, "attribute")
            freq_attr.set("name", "freq")
            freq_attr.set("value", str(node.frequency))

            source_attr = ET.SubElement(elem, "attribute")
            source_attr.set("name", "source")
            source_attr.set("value", node.source)

            # Add context for nodes with contexts
            if node.contexts or not has_children:
                ctx_attr = ET.SubElement(elem, "attribute")
                ctx_attr.set("name", "context")
                ctx_attr.set("value", format_context(node.contexts, source_name))

            # Add children sorted by frequency (descending)
            sorted_children = sorted(
                node.children.values(),
                key=lambda n: (-n.frequency, n.name)
            )
            for child in sorted_children:
                add_node(elem, child)

        # Build tree from root
        add_node(tree, self.root, is_root=True)

        # Generate XML string with proper formatting
        ET.indent(tree, space="  ")

        # Add XML declaration and comment
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
    nlp: spacy.language.Language | None = None,
    min_words: int = 2,
    max_words: int = 4,
    lemmatize: bool = True,
) -> ConceptTreeBuilder:
    """Convenience function to build a concept tree from text.

    Parameters
    ----------
    text : str
        Input text
    nlp : spacy.language.Language | None
        spaCy model (optional)
    min_words : int
        Minimum phrase length
    max_words : int
        Maximum phrase length
    lemmatize : bool
        Whether to lemmatize tokens

    Returns
    -------
    ConceptTreeBuilder
        The builder with populated tree
    """
    builder = ConceptTreeBuilder(
        nlp=nlp,
        min_phrase_words=min_words,
        max_phrase_words=max_words,
        lemmatize=lemmatize,
    )
    builder.process_text(text)
    return builder


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build concept tree from text file"
    )
    parser.add_argument("input_file", type=Path, help="Input text file")
    parser.add_argument("-o", "--output", type=Path, help="Output XML file")
    parser.add_argument("--min-words", type=int, default=2, help="Min phrase words")
    parser.add_argument("--max-words", type=int, default=4, help="Max phrase words")
    parser.add_argument("--print", action="store_true", help="Print tree to console")
    parser.add_argument("--no-lemma", action="store_true",
                        help="Don't lemmatize (keep surface forms like 'narratives')")

    args = parser.parse_args()

    # Read input
    text = args.input_file.read_text(encoding="utf-8")

    # Build tree
    print(f"Processing: {args.input_file}")
    builder = build_concept_tree(
        text,
        min_words=args.min_words,
        max_words=args.max_words,
        lemmatize=not args.no_lemma,
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
