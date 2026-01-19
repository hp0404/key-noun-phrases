# plan.md — Keyphrase Extraction (CiteSpace-to-Python Port)

## 1. Context

We are porting core ideas from **CiteSpace** into Python, but adapting them to a different operating mode:

- **Input reality:** arbitrary raw text (often without metadata such as author keywords, abstracts, venue, etc.).
- **Operational unit:** the system must reliably extract **high-quality keyphrases from a single text at a time**.
- **Later aggregation:** multi-document signals (e.g., burstiness) may be computed later, after many documents have been processed, but they are **not assumed available at first-pass extraction time**.

## 2. Current Approach and Constraints

### 2.1 Extraction method
- We use **spaCy `Matcher`** with **explicit patterns** for candidate keyphrases.
- We **do not** rely on spaCy’s built-in `.noun_chunks` / noun phrase attribute for English.
- Patterns are **language-specific**; target architecture is **one file per target language**.
    - `terms/assets/default_patterns.json` should be refactored into English-specific patterns

### 2.2 Pattern source
- The older patterns are stored in the `terms/assets/` folder.
- These patterns may contain errors:
  - Some useful patterns were likely **omitted**.
  - Some patterns are likely **incorrect or brittle** for modern text.
  - The current pattern set produces **nested / redundant matches** that complicate downstream phrase selection.
  - The pattern set needs to be reviewed and revised.

## 3. Primary Problems to Solve

### 3.1 Pattern quality gaps
- Coverage gaps: important constructions are missing.
- Precision issues: some matches feel “wrong,” even if technically syntactically valid.

### 3.2 Nested / redundant matches (maximal span + all subspans)
A representative example:

> “We implemented a robust data security incident response plan for rapid recovery.”

The matcher can yield the maximal noun phrase and many contained subspans:

- **robust data security incident response plan** → `ADJ-NOUN-NOUN-NOUN-NOUN-NOUN`
- **robust data security incident response** → `ADJ-NOUN-NOUN-NOUN-NOUN`
- **robust data security incident** → `ADJ-NOUN-NOUN-NOUN`
- **robust data security** → `ADJ-NOUN-NOUN`
- **robust data** → `ADJ-NOUN`

This creates two practical requirements:
1) Differentiate the **maximal phrase** from its **related subspans** (within the same local region of text).  
2) Preserve other phrases elsewhere in the document as **unrelated candidates** (not mistakenly collapsed).

### 3.3 Scoring stage mismatch with CiteSpace defaults
CiteSpace emphasizes cross-document dynamics (e.g., burstiness in scientific corpora with strong metadata). In our use case:
- We cannot assume metadata or author-supplied keywords.
- We must prioritize **within-document scoring** at extraction time.

Therefore:
- **Burstiness (Kleinberg burst detection)** is not a first-pass dependency (it requires a document stream) — ignore for now
- **TF-IDF (per document)** and **graph-based centrality (per document)** are promising first-pass signals

## 4. Proposed Direction (High-Level)

We should improve two steps first and foremost:

1) **Develop better, language-specific patterns**
2) **Implement within-document scoring of extracted keyphrases**

In addition, we must introduce a deterministic **redundancy-resolution layer** to handle nested matches.

---

## 5. Maximal Spans Logic (Redundancy Resolution)

### 5.1 Definitions
Represent each match as a span object:

- `start`: token start index (inclusive)
- `end`: token end index (exclusive)
- `text`
- `label` (pattern label)
- `length = end - start`

A span **A contains B** if:
- `A.start <= B.start` and `B.end <= A.end`

A span **A strictly contains B** if it contains B and is not identical.

### 5.2 Maximal span
A span is **maximal** if **no other span strictly contains it**.

This partitions matches into:
- **Maximal candidates** (top-level phrases)
- **Subspans** (contained phrases)

### 5.3 Grouping: “families” of related subspans
For each maximal span **M**, define its family:

- `family(M) = { s | M contains s }`

To avoid attaching subspans to an overly broad maximal span when a tighter maximal exists, assign each non-maximal span `s` to the **smallest maximal** that contains it (minimal length among containers).

**Result:**  
- Each maximal span becomes a “parent.”  
- Contained spans are “related subspans.”  
- Spans not contained in any maximal span remain **unrelated** and form their own groups.

### 5.4 Differentiation outputs

The current output structure (from `terms/__init__.py`) yields flat dictionaries:

```python
{
    "uuid": uuid,
    "pos_label": pos_label,
    "key_noun_phrase": span.text,
    "key_noun_phrase_processed": " ".join(t.lemma_.lower() for t in span if not t.is_punct),
    "span_location": [span.start_char, span.end_char],
}
```

To expose families of related keyphrases, **add the following fields**:

| Field | Type | Description |
|-------|------|-------------|
| `token_span` | `[int, int]` | Token indices `[start, end)` — required for containment checks (character offsets alone are insufficient for reliable nesting logic) |
| `is_maximal` | `bool` | `True` if no other span strictly contains this one |
| `family_id` | `str \| None` | Unique identifier for the maximal span this belongs to; equals own ID if maximal, parent's ID if subspan |
| `maximal_text` | `str \| None` | Text of the maximal parent (for quick lookup); `None` if this span is itself maximal |

Alternatively, provide a **grouped output mode** that returns nested structures:

```python
{
    "uuid": uuid,
    "families": [
        {
            "family_id": "...",
            "maximal": {
                "pos_label": "ADJ-NOUN-NOUN-NOUN-NOUN-NOUN",
                "key_noun_phrase": "robust data security incident response plan",
                "key_noun_phrase_processed": "robust data security incident response plan",
                "token_span": [2, 8],
                "span_location": [20, 62],
            },
            "subspans": [
                {
                    "pos_label": "ADJ-NOUN-NOUN-NOUN-NOUN",
                    "key_noun_phrase": "robust data security incident response",
                    ...
                },
                ...
            ]
        },
        ...
    ],
    "unrelated": [...]  # spans that do not nest inside any maximal span
}
```

This yields an explicit separation between:
- "This is the canonical extraction"
- "These are redundant/derived candidates we may keep for evidence, aliasing, or later scoring"

### 5.5 Optional: direct-child containment tree
If needed, compute a containment tree where `s` is a **direct child** of `M` if:
- `M` strictly contains `s`, and
- no intermediate span exists between them.

This supports compact UIs and reduces combinatorial listing of all nested spans.

### 5.6 Automatic “pick one” policy (default)
Given the user requirement (“we need to pick one pattern in each such case automatically”), the default policy should be:

- **Keep the maximal span** as the primary candidate.
- Treat subspans as **secondary** unless they score significantly higher on within-document signals.

This can be implemented as:
- **Canonical list:** maximal spans only
- **Expanded list:** maximal spans plus subspans with metadata linking them to their maximal parent

### 5.7 Edge cases to handle explicitly
- Overlapping spans that are not strict containments (crossing spans): treat as separate groups; optionally resolve by span-score and/or label priority.
- Duplicate surface text at different locations: treat as separate occurrences initially; merge later during scoring/normalization if desired.
- Hyphenation and tokenization variation: ensure ORTH-based patterns align with tokenizer output.

---

## 6. Scoring (Within-Document First Pass)

### 6.1 Goals
- Rank candidate phrases extracted from a single document.
- Reduce noise and improve top-K phrase quality.

### 6.2 Signals to implement first
1) **TF-IDF (per document)**
   - Use within-document term frequency.
   - Use an IDF estimate if available from a background corpus; otherwise use a local proxy (e.g., downweight very common tokens via stoplists and POS constraints).

2) **Betweenness centrality on a within-document co-occurrence graph**
   - Build a co-occurrence graph of candidate phrases and/or their normalized tokens.
   - Compute betweenness centrality as a proxy for “bridge concepts” inside the document narrative.

### 6.3 Interaction with maximal spans logic
Scoring should operate at two levels:
- **Span-instance scoring:** each occurrence in context
- **Canonical scoring:** aggregate occurrences for the canonical phrase (maximal) and optionally roll-up related subspans

Recommended default:
- Score all span instances.
- Promote maximal spans as canonical and:
  - either take their own score,
  - or compute a roll-up score that incorporates selected subspan evidence.

## 7. Workstreams and Deliverables

### 7.1 Workstream A — Pattern library (per language)
- Consolidate existing patterns.
- Audit for coverage gaps and false positives.
- Maintain **one pattern file per language** with:
  - labels
  - matcher patterns
  - notes/examples per pattern (optional but recommended)
    - Use these examples for the tests — each language and each pattern should be tested

Deliverables:
- `terms/assets/<lang>_patterns.json` files
- Unit tests with representative sentences

### 7.2 Workstream B — Extraction engine
- Review `terms/__init__.py` — improve if needed, including the output schema
- Update the output schema to capture all the nuances

### 7.3 Workstream C — Redundancy resolution (maximal spans)
- Compute maximal spans
- Build families (maximal + related subspans)
- Output canonical list + optional expanded list

Deliverables:
- `terms/group_spans.py` with import into `terms/__init__.py`
- Tests for containment / overlap edge cases

### 7.4 Workstream D — Scoring (within-document)
- Implement TF-IDF-like score for candidates
- Implement co-occurrence graph and betweenness centrality
- Combine into a composite score (configurable weights)

Deliverables:
- `terms/score.py` with import into `terms/__init__.py`
- Evaluation script for top-K inspection
- Unit tests in `tests/`
