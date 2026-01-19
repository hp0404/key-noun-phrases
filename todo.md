# Refactor TODO — Key Noun Phrase Extraction

This document tracks the refactor from a CiteSpace-to-Python port to a within-document keyphrase extraction system.

---

## Phase 1: Pattern Library (Workstream A) ✅ COMPLETE

### 1.1 Rename and reorganize pattern files
- [x] Rename `terms/assets/default_patterns.json` → `terms/assets/en_patterns.json`
- [x] Update `terms/__init__.py` to use `en_patterns.json` for English models
- [x] Verify all language mappings: `de`, `ru`, `uk`, `en`

### 1.2 Audit English patterns (`en_patterns.json`)
- [x] Review all 48 existing patterns for correctness
- [x] Remove patterns that produce noisy/low-quality matches:
  - [x] Removed `ADJ-VERB`, `ADJ-NUM` — low value (not noun phrases)
  - [x] Kept `ADP-PUNCT-*`, `DET-PUNCT-*` (useful for hyphenated compounds)
  - [x] Removed `PART-PUNCT-*` patterns (extremely rare)
- [x] Identify coverage gaps (missing useful constructions):
  - [x] Added `NOUN-NOUN-NOUN-NOUN-NOUN` (5-noun chains)
  - [x] Added coordinated noun phrases (`NOUN-CCONJ-NOUN` and variants)

### 1.3 Audit German patterns (`de_patterns.json`)
- [x] Review 6 existing patterns using optional/quantifier operators (`OP`)
- [x] Validate patterns against German grammar rules
- [x] Ensure label naming is consistent (uses `GERMAN_*` prefix)
- [x] German-specific tests exist

### 1.4 Audit Russian patterns (`ru_patterns.json`)
- [x] Review 30 existing patterns
- [x] Validate Slavic-specific constructions (genitive case works correctly)
- [x] Russian-specific tests exist

### 1.5 Audit Ukrainian patterns (`uk_patterns.json`)
- [x] Review 30 existing patterns (mirrors Russian appropriately)
- [x] Ukrainian-specific tests exist

---

## Phase 2: Extraction Engine (Workstream B) ✅ COMPLETE

### 2.1 Update output schema in `terms/__init__.py`
- [x] Add `token_span: [int, int]` field (token indices `[start, end)`)
- [x] Add `is_maximal: bool` field (placeholder, computed in Phase 3)
- [x] Add `family_id: str | None` field (placeholder, computed in Phase 3)
- [x] Add `maximal_text: str | None` field (placeholder, computed in Phase 3)
- [x] Update docstrings to reflect new schema

### 2.2 Refactor `yield_key_phrases` method
- [x] Token indices captured correctly (spacy Span has doc-absolute indices)
- [x] `exclusive_search` logic reviewed and working correctly

### 2.3 Review `treebank.py` verb helpers
- [x] Verify `is_vbg()` and `is_vbn()` work correctly (primarily for English)
- Note: German/Russian participles tagged as ADJ, not VERB, so filters don't apply

---

## Phase 3: Redundancy Resolution (Workstream C) ✅ COMPLETE

### 3.1 Create `terms/group_spans.py` module
- [x] Implement `Span` dataclass with `start`, `end`, `text`, `label`, `length`
- [x] Implement `contains(a, b)` function (A contains B)
- [x] Implement `strictly_contains(a, b)` function
- [x] Implement `find_maximal_spans(spans)` function
- [x] Implement `build_families(spans)` function — group subspans under smallest containing maximal

### 3.2 Implement family assignment logic
- [x] For each non-maximal span, assign to smallest maximal that contains it
- [x] Handle edge case: overlapping (crossing) spans placed in `_unrelated`

### 3.3 Create output modes
- [x] **Flat mode**: all spans with `is_maximal`, `family_id`, `maximal_text` fields
- [x] `resolve_redundancy` parameter controls annotation

### 3.4 Integrate into main extraction flow
- [x] Import `group_spans` into `terms/__init__.py`
- [x] Added `extract_key_phrases()` method for batch extraction with annotation
- [x] Updated `to_dataframe()` with `resolve_redundancy` parameter

### 3.5 Add tests for redundancy resolution
- [x] Test maximal span detection
- [x] Test family grouping
- [x] Test token_span field
- [x] Test resolve_redundancy flag

---

## Phase 4: Scoring (Workstream D) ✅ COMPLETE

### 4.1 Create `terms/score.py` module
- [x] Define scoring interface/protocol (Scorer Protocol class)

### 4.2 Implement TF-IDF scoring
- [x] Compute within-document term frequency for candidate phrases
- [x] Implement IDF proxy (downweight common tokens via POS/stoplist)
- [x] Combine into TF-IDF score

### 4.3 Implement co-occurrence graph centrality
- [x] Build co-occurrence graph from candidate phrases (window-based)
- [x] Compute betweenness centrality for each phrase/node
- [x] Normalize centrality scores

### 4.4 Implement composite scoring
- [x] Define configurable weights for TF-IDF and centrality
- [x] Implement score aggregation function
- [x] Score maximal spans with optional bonus multiplier

### 4.5 Integrate scoring into extraction pipeline
- [x] Add `score` field to output schema
- [x] Add scoring parameters to `extract_key_phrases()` and `to_dataframe()`
- [x] Add `score_phrases()` convenience function

### 4.6 Add tests and evaluation
- [x] Unit tests for TF-IDF calculation
- [x] Unit tests for centrality calculation
- [x] Integration tests for scoring pipeline

---

## Phase 5: Final Integration & Cleanup ✅ COMPLETE

### 5.1 Integration testing
- [x] End-to-end test: raw text → scored, deduplicated keyphrases
- [x] Test with real documents across all 4 languages
- [x] Backward compatibility maintained (existing APIs unchanged, new features opt-in)

### 5.2 Documentation
- [x] Update README.md with new features
- [x] Add usage examples for new output modes
- [x] Document pattern file format and custom scorer usage

### 5.3 Performance review
- [x] Existing batch processing via spacy.pipe maintained
- [x] No additional optimization needed for typical use cases

---

## Dependencies & Order

```
Phase 1 (Patterns) ──┐
                     ├──► Phase 2 (Engine) ──► Phase 3 (Redundancy) ──► Phase 4 (Scoring) ──► Phase 5 (Integration)
                     │
Phases are sequential; within each phase, tasks can often be parallelized.
```

## Progress Summary

- **Phase 1**: ✅ Complete - Pattern files reorganized, English patterns audited (removed 3 low-value, added 6 new for coverage)
- **Phase 2**: ✅ Complete - Output schema updated with token_span and placeholder fields
- **Phase 3**: ✅ Complete - group_spans.py created, integrated into extraction flow, tests added
- **Phase 4**: ✅ Complete - score.py created with TF-IDF, centrality, and composite scorers; integrated into pipeline
- **Phase 5**: ✅ Complete - End-to-end tests added, README updated with new features and usage examples

**All phases complete! Total: 66 tests passing.**
