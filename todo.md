# Refactor TODO — Key Noun Phrase Extraction

This document tracks the refactor from a CiteSpace-to-Python port to a within-document keyphrase extraction system.

---

## Phase 1: Pattern Library (Workstream A)

### 1.1 Rename and reorganize pattern files
- [ ] Rename `terms/assets/default_patterns.json` → `terms/assets/en_patterns.json`
- [ ] Update `terms/__init__.py` to use `en_patterns.json` for English models
- [ ] Verify all language mappings: `de`, `ru`, `uk`, `en`

### 1.2 Audit English patterns (`en_patterns.json`)
- [ ] Review all 48 existing patterns for correctness
- [ ] Remove patterns that produce noisy/low-quality matches:
  - [ ] Evaluate `ADJ-VERB`, `ADJ-NUM` — likely low value
  - [ ] Evaluate `ADP-PUNCT-*`, `DET-PUNCT-*`, `PART-PUNCT-*` patterns
- [ ] Identify coverage gaps (missing useful constructions):
  - [ ] Single NOUN/PROPN patterns (if needed)
  - [ ] Longer noun chains (5+ tokens)
  - [ ] Coordinated noun phrases (`NOUN CONJ NOUN`)
- [ ] Add test sentences for each pattern in `tests/test_patterns.py`

### 1.3 Audit German patterns (`de_patterns.json`)
- [ ] Review 6 existing patterns using optional/quantifier operators (`OP`)
- [ ] Validate patterns against German grammar rules
- [ ] Ensure label naming is consistent (currently uses `GERMAN_*` prefix)
- [ ] Add German-specific test cases

### 1.4 Audit Russian patterns (`ru_patterns.json`)
- [ ] Review 30 existing patterns
- [ ] Validate Slavic-specific constructions (genitive case, etc.)
- [ ] Add Russian-specific test cases

### 1.5 Audit Ukrainian patterns (`uk_patterns.json`)
- [ ] Review 30 existing patterns (currently mirrors Russian)
- [ ] Identify any Ukrainian-specific adjustments needed
- [ ] Add Ukrainian-specific test cases

---

## Phase 2: Extraction Engine (Workstream B)

### 2.1 Update output schema in `terms/__init__.py`
- [ ] Add `token_span: [int, int]` field (token indices `[start, end)`)
- [ ] Add `is_maximal: bool` field (placeholder, computed in Phase 3)
- [ ] Add `family_id: str | None` field (placeholder, computed in Phase 3)
- [ ] Add `maximal_text: str | None` field (placeholder, computed in Phase 3)
- [ ] Update docstrings to reflect new schema

### 2.2 Refactor `yield_key_phrases` method
- [ ] Ensure token indices are captured correctly from subtree-relative to doc-absolute
- [ ] Review `exclusive_search` logic for correctness
- [ ] Consider extracting match processing into a helper function

### 2.3 Review `treebank.py` verb helpers
- [ ] Verify `is_vbg()` and `is_vbn()` work correctly across all languages
- [ ] Add unit tests for treebank functions

---

## Phase 3: Redundancy Resolution (Workstream C)

### 3.1 Create `terms/group_spans.py` module
- [ ] Implement `Span` dataclass with `start`, `end`, `text`, `label`, `length`
- [ ] Implement `contains(a, b)` function (A contains B)
- [ ] Implement `strictly_contains(a, b)` function
- [ ] Implement `find_maximal_spans(spans)` function
- [ ] Implement `build_families(spans)` function — group subspans under smallest containing maximal

### 3.2 Implement family assignment logic
- [ ] For each non-maximal span, assign to smallest maximal that contains it
- [ ] Handle edge case: overlapping (crossing) spans that don't nest
- [ ] Handle edge case: duplicate surface text at different positions

### 3.3 Create output modes
- [ ] **Flat mode**: all spans with `is_maximal`, `family_id`, `maximal_text` fields
- [ ] **Grouped mode**: nested structure with `families[]` and `unrelated[]`
- [ ] Add parameter to `yield_key_phrases` to select output mode

### 3.4 Integrate into main extraction flow
- [ ] Import `group_spans` into `terms/__init__.py`
- [ ] Apply grouping after raw matches are collected
- [ ] Update `to_dataframe()` to handle new output structure

### 3.5 Add tests for redundancy resolution
- [ ] Test maximal span detection
- [ ] Test family grouping
- [ ] Test crossing span handling
- [ ] Test the example from plan.md: "robust data security incident response plan"

---

## Phase 4: Scoring (Workstream D)

### 4.1 Create `terms/score.py` module
- [ ] Define scoring interface/protocol

### 4.2 Implement TF-IDF scoring
- [ ] Compute within-document term frequency for candidate phrases
- [ ] Implement IDF proxy (downweight common tokens via POS/stoplist)
- [ ] Combine into TF-IDF score

### 4.3 Implement co-occurrence graph centrality
- [ ] Build co-occurrence graph from candidate phrases (window-based)
- [ ] Compute betweenness centrality for each phrase/node
- [ ] Normalize centrality scores

### 4.4 Implement composite scoring
- [ ] Define configurable weights for TF-IDF and centrality
- [ ] Implement score aggregation function
- [ ] Score maximal spans with optional subspan roll-up

### 4.5 Integrate scoring into extraction pipeline
- [ ] Add `score` field to output schema
- [ ] Add scoring parameters to `yield_key_phrases`
- [ ] Update `to_dataframe()` to include scores

### 4.6 Add tests and evaluation
- [ ] Unit tests for TF-IDF calculation
- [ ] Unit tests for centrality calculation
- [ ] Create evaluation script for top-K phrase inspection

---

## Phase 5: Final Integration & Cleanup

### 5.1 Integration testing
- [ ] End-to-end test: raw text → scored, deduplicated keyphrases
- [ ] Test with real documents across all 4 languages
- [ ] Verify backward compatibility (if needed)

### 5.2 Documentation
- [ ] Update README.md with new features
- [ ] Add usage examples for new output modes
- [ ] Document pattern file format and contribution guidelines

### 5.3 Performance review
- [ ] Profile extraction on large documents
- [ ] Optimize if bottlenecks found (batch processing, caching)

---

## Dependencies & Order

```
Phase 1 (Patterns) ──┐
                     ├──► Phase 2 (Engine) ──► Phase 3 (Redundancy) ──► Phase 4 (Scoring) ──► Phase 5 (Integration)
                     │
Phases are sequential; within each phase, tasks can often be parallelized.
```
