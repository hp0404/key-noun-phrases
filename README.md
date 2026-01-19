# key-noun-phrases

Key Noun Phrase extraction using dependency parsing and POS tag patterns, inspired by CiteSpace's term extraction approach.

## Supported Languages

| Language | Model | Patterns |
|----------|-------|----------|
| English | `en_core_web_sm` | `en_patterns.json` |
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

### Basic Extraction

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

### Redundancy Resolution

The extractor can identify overlapping phrases and group them into families. A *maximal span* is a phrase not contained by any other phrase. Subphrases are grouped under their smallest containing maximal span.

```python
# Extract with redundancy resolution (enabled by default)
results = terms.extract_key_phrases(
    sentences,
    resolve_redundancy=True  # default
)

# Each result includes:
# - is_maximal: True if this phrase is not contained by another
# - family_id: ID of the family this phrase belongs to
# - maximal_text: Text of the maximal phrase containing this one

# Filter to only maximal phrases
maximal_phrases = [r for r in results if r["is_maximal"]]
```

### Scoring

Phrases can be scored by importance using TF-IDF and co-occurrence graph centrality:

```python
# Extract with scoring
results = terms.extract_key_phrases(
    sentences,
    compute_scores=True
)

# Sort by score to get most important phrases
top_phrases = sorted(results, key=lambda x: x["score"], reverse=True)[:10]

# DataFrame with scores
df = terms.to_dataframe(sentences, compute_scores=True)
df_sorted = df.sort_values("score", ascending=False)
```

#### Custom Scorers

```python
from terms.score import TFIDFScorer, CentralityScorer, CompositeScorer

# Use only TF-IDF scoring
tfidf_scorer = TFIDFScorer()
results = terms.extract_key_phrases(sentences, compute_scores=True, scorer=tfidf_scorer)

# Use only centrality scoring
centrality_scorer = CentralityScorer(window_size=5)
results = terms.extract_key_phrases(sentences, compute_scores=True, scorer=centrality_scorer)

# Custom weights for composite scoring
custom_scorer = CompositeScorer(
    tfidf_weight=0.7,
    centrality_weight=0.3,
    maximal_bonus=1.5  # Bonus for maximal spans
)
results = terms.extract_key_phrases(sentences, compute_scores=True, scorer=custom_scorer)
```

### Full Pipeline Example

```python
import spacy
from terms import TermsMatcher

nlp = spacy.load("en_core_web_sm")
terms = TermsMatcher(nlp=nlp)

text = """
The advanced machine learning algorithm demonstrates remarkable accuracy.
Statistical analysis of the machine learning results shows improvement.
"""

sentences = [(text, "doc1")]

# Full extraction with all features
df = terms.to_dataframe(
    sentences,
    exclusive_search=False,      # Include all phrases, not just subject-rooted
    resolve_redundancy=True,     # Group overlapping phrases
    compute_scores=True          # Add importance scores
)

# Get top 5 maximal phrases by score
top_maximal = (
    df[df["is_maximal"] == True]
    .sort_values("score", ascending=False)
    .head(5)
)
print(top_maximal[["key_noun_phrase", "score"]])
```

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | str | Document identifier |
| `pos_label` | str | Matched pattern (e.g., `ADJ-NOUN`) |
| `key_noun_phrase` | str | Original phrase text |
| `key_noun_phrase_processed` | str | Lemmatized, lowercased form |
| `span_location` | list[int] | Character offsets `[start, end]` |
| `token_span` | list[int] | Token indices `[start, end)` |
| `is_maximal` | bool | True if not contained by another phrase |
| `family_id` | str | Family group ID (e.g., `family_1`) |
| `maximal_text` | str | Text of containing maximal phrase |
| `score` | float | Importance score (if `compute_scores=True`) |

## Pattern Types

The extraction uses CiteSpace-style POS patterns:

| Category | Examples |
|----------|----------|
| Basic NP | `ADJ-NOUN`, `NOUN-NOUN`, `ADJ-ADJ-NOUN` |
| Prepositional | `NOUN-ADP-NOUN`, `ADJ-NOUN-ADP-NOUN` |
| Verbal | `VERB-NOUN`, `VERB-ADJ-NOUN` |
| Hyphenated | `ADJ-PUNCT-NOUN-NOUN` |
| Coordinated | `NOUN-CCONJ-NOUN` |
| Long chains | `NOUN-NOUN-NOUN-NOUN-NOUN` |

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
