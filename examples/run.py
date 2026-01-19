#!/usr/bin/env python3
"""Run key noun phrase extraction on text files in examples/texts/."""

import json
from pathlib import Path

import spacy

from terms import TermsMatcher


def detect_language(file_path: Path) -> str:
    """Detect language from filename prefix (e.g., uk_telegram.txt -> uk)."""
    stem = file_path.stem
    # Check for language prefix pattern: lang_*
    if "_" in stem:
        prefix = stem.split("_")[0]
        if prefix in {"en", "de", "ru", "uk"}:
            return prefix
    # Default to English
    return "en"


def get_spacy_model(lang: str) -> str:
    """Get spacy model name for a language."""
    models = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
        "ru": "ru_core_news_sm",
        "uk": "uk_core_news_sm",
    }
    return models.get(lang, "en_core_web_sm")


def process_file(input_path: Path, output_path: Path, nlp) -> None:
    """Process a single text file and save results as JSON."""
    # Read the text
    text = input_path.read_text(encoding="utf-8")

    # Create the matcher
    terms = TermsMatcher(nlp=nlp)

    # Prepare sentences - use the filename as the document ID
    # The library expects list of (text, uuid) tuples
    sentences = [(text, input_path.stem)]

    # Extract key phrases with all features enabled
    results = terms.extract_key_phrases(
        sentences,
        exclusive_search=False,  # Include all phrases, not just subject-rooted
        resolve_redundancy=True,  # Group overlapping phrases
        compute_scores=True,  # Add importance scores
    )

    # Save results as JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processed: {input_path.name} -> {output_path.name}")
    print(f"  Found {len(results)} phrases")
    if results:
        maximal_count = sum(1 for r in results if r.get("is_maximal"))
        print(f"  Maximal phrases: {maximal_count}")


def main():
    """Main entry point."""
    # Get the examples directory
    script_dir = Path(__file__).resolve().parent
    texts_dir = script_dir / "texts"

    if not texts_dir.exists():
        print(f"Error: texts directory not found at {texts_dir}")
        return

    # Find all .txt files
    txt_files = list(texts_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {texts_dir}")
        return

    print(f"Found {len(txt_files)} text file(s) to process\n")

    # Group files by language to minimize model loading
    files_by_lang: dict[str, list[Path]] = {}
    for txt_file in txt_files:
        lang = detect_language(txt_file)
        files_by_lang.setdefault(lang, []).append(txt_file)

    # Process files grouped by language
    for lang, files in files_by_lang.items():
        model_name = get_spacy_model(lang)
        print(f"Loading spaCy model: {model_name}")
        try:
            nlp = spacy.load(model_name)
        except OSError:
            print(f"  Error: Model '{model_name}' not installed.")
            print(f"  Install with: python -m spacy download {model_name}")
            continue

        for txt_file in files:
            output_file = txt_file.with_suffix(".json")
            process_file(txt_file, output_file, nlp)

        print()

    print("Done!")


if __name__ == "__main__":
    main()
