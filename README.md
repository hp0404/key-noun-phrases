# key-noun-phrases

Key Noun Phrase extraction using dependency parsing and POS tag patterns, inspired by CiteSpace's term extraction approach.

## Supported Languages

| Language | Model | Patterns |
|----------|-------|----------|
| English | `en_core_web_sm` | `default_patterns.json` |
| German | `de_core_news_sm` | `de_patterns.json` |
| Russian | `ru_core_news_sm` | `ru_patterns.json` |
| Ukrainian | `uk_core_news_sm` | `uk_patterns.json` |

## Installation

```console
git clone https://github.com/hp0404/key-noun-phrases.git
cd key-noun-phrases
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install spacy models for your target language(s):

```console
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python -m spacy download ru_core_news_sm
python -m spacy download uk_core_news_sm
```

## Usage

```python
import spacy
from terms import TermsMatcher

nlp = spacy.load("en_core_web_sm")
terms = TermsMatcher(nlp=nlp)

sentences = [
    ("Statistical analysis of data reveals patterns.", "doc1"),
    ("The neural network learns features.", "doc2"),
]

# As DataFrame
df = terms.to_dataframe(sentences)

# As iterator
for phrase in terms.yield_key_phrases(sentences):
    print(phrase)
```

Output includes:
- `uuid`: document identifier
- `pos_label`: matched pattern (e.g., `ADJ-NOUN`, `NOUN-ADP-NOUN`)
- `key_noun_phrase`: original text
- `key_noun_phrase_processed`: lemmatized, lowercased form
- `span_location`: character offsets

## Pattern Types

The extraction uses CiteSpace-style POS patterns:

| Category | Examples |
|----------|----------|
| Basic NP | `ADJ-NOUN`, `NOUN-NOUN`, `ADJ-ADJ-NOUN` |
| Prepositional | `NOUN-ADP-NOUN`, `ADJ-NOUN-ADP-NOUN` |
| Verbal | `VERB-NOUN`, `VERB-ADJ-NOUN` |
| Hyphenated | `ADJ-PUNCT-NOUN-NOUN` |

Custom patterns can be provided via a JSON file:

```python
from pathlib import Path
from terms import TermsMatcher, build_matcher

nlp = spacy.load("en_core_web_sm")
custom_matcher = build_matcher(nlp, Path("my_patterns.json"))
terms = TermsMatcher(nlp=nlp, matcher=custom_matcher)
```

## Testing

```console
pytest tests/ -v
```

## License

MIT
