#!/usr/bin/env python3
"""Visualize spaCy parse tree for text files."""

from pathlib import Path

import spacy

MODELS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "ru": "ru_core_news_sm",
    "uk": "uk_core_news_sm",
}


def detect_language(file_path: Path) -> str:
    stem = file_path.stem
    if "_" in stem:
        prefix = stem.split("_")[0]
        if prefix in MODELS:
            return prefix
    return "en"


def print_token_table(doc):
    """Print token attributes in a table format."""
    print(
        f"{'IDX':<4} {'TOKEN':<20} {'LEMMA':<20} {'POS':<6} {'TAG':<10} {'DEP':<12} {'HEAD':<20} {'MORPH'}"
    )
    print("-" * 130)
    for token in doc:
        print(
            f"{token.i:<4} {token.text:<20} {token.lemma_:<20} {token.pos_:<6} "
            f"{token.tag_:<10} {token.dep_:<12} {token.head.text:<20} {str(token.morph)}"
        )


def print_subtrees(doc):
    """Print subtree information for subjects."""
    print("\n" + "=" * 80)
    print("SUBJECT SUBTREES (what the extractor searches)")
    print("=" * 80)

    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass", "sb", "sbp"):
            if token.head.pos_ in ("VERB", "AUX"):
                left = token.left_edge.i
                right = token.right_edge.i + 1
                subtree_span = doc[left:right]
                print(f"\nSubject: '{token.text}' (idx={token.i})")
                print(f"  Head verb: '{token.head.text}' (pos={token.head.pos_})")
                print(f"  Subtree span: [{left}:{right}]")
                print(f"  Subtree text: '{subtree_span.text}'")
                print(f"  Subtree tokens:")
                for t in subtree_span:
                    print(f"    {t.i}: {t.text} ({t.pos_})")


def print_sentence_tree(sent):
    """Print dependency tree for a sentence."""

    def print_tree(token, indent=0):
        print("  " * indent + f"└─ {token.text} [{token.pos_}, {token.dep_}]")
        for child in token.children:
            print_tree(child, indent + 1)

    print(
        f'\nDependency tree for: "{sent.text[:60]}..."'
        if len(sent.text) > 60
        else f'\nDependency tree for: "{sent.text}"'
    )
    print_tree(sent.root)


def main() -> int:
    txt_path = Path(__file__).parent / "texts" / "uk_telegram.txt"

    if not txt_path.exists():
        print(f"File not found: {txt_path}")
        return 1

    lang = detect_language(txt_path)
    model_name = MODELS.get(lang, "en_core_web_sm")

    print(f"File: {txt_path}")
    print(f"Language: {lang}")
    print(f"Model: {model_name}")
    print()

    nlp = spacy.load(model_name)
    text = txt_path.read_text(encoding="utf-8")
    doc = nlp(text)

    print("=" * 80)
    print("TOKEN ANALYSIS")
    print("=" * 80)
    print_token_table(doc)

    print_subtrees(doc)

    print("\n" + "=" * 80)
    print("DEPENDENCY TREES (first 5 sentences)")
    print("=" * 80)
    for i, sent in enumerate(doc.sents):
        if i >= 5:
            print("\n... (truncated)")
            break
        print_sentence_tree(sent)


if __name__ == "__main__":
    main()
