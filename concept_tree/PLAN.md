# CiteSpace Concept Tree Pipeline - Implementation Plan

## Overview

Reverse-engineer the CiteSpace concept tree extraction algorithm based on:
- The input text (`input.txt`)
- The expected output (`concept_tree.xml`)
- The paper describing the algorithm (Chen, 2012, Chapter 7)

## Algorithm Summary (from Paper)

### Input
- Plain text document(s)

### Processing Steps

1. **POS Tagging**: Tag each word with part-of-speech using a tagger
2. **Pattern Matching**: Extract noun phrases using regex patterns over POS-tagged text
3. **Tree Construction**:
   - Split noun phrases into **head nouns** (concepts) and **modifiers** (attributes)
   - Head noun becomes parent node (branch)
   - Modifiers become child nodes (leaves)
   - Single nouns without modifiers become leaf nodes directly under root
4. **Context Tracking**: Store the original sentence for each extracted phrase
5. **Frequency Counting**: Track how many times each concept appears
6. **XML Output**: Generate TreeML format XML

### Key Patterns from Paper (Table 7.6)

| Pattern | Definition | Example |
|---------|------------|---------|
| noun | (article OR adjective)* + word/nn[sp]* | heavy and cold rain |
| noun phrase | various combinations of nouns and words | Information visualization research |
| concept | noun phrase, including "noun of noun" | exotic plant |

### Tree Structure Rules

1. **Head Noun Identification**: The rightmost noun in a noun phrase is typically the head
2. **Modifier Extraction**: All words before the head noun are modifiers
3. **Multi-word Modifiers**: Can be grouped (e.g., "destiny and limited")
4. **Frequency Aggregation**: Branch nodes sum frequencies of their children plus direct occurrences
5. **Context Storage**: HTML-escaped sentence contexts stored as `<ul><li>...</li></ul>`

## Analysis of Expected Output

From `concept_tree.xml`, the following patterns are extracted:

### Direct Leaves (no modifiers)
- "europe" (freq=2)
- "european union" (freq=2) - proper noun phrase kept together
- "carl schmitt" (freq=1) - proper noun phrase
- "limitations" (freq=1)
- "point" (freq=1)
- "power" (freq=1)
- "russia" (freq=1)
- "schmitts" (freq=1)
- "self-understanding" (freq=1)
- "ukraine" (freq=1)

### Branch + Leaf Patterns (head → modifiers)
- "narratives" → "celebratory" (from "celebratory narratives")
- "distance" → "critical" (from "critical distance")
- "capabilities" → "destiny and limited" (from "destiny and the limited capabilities")
- "deficiencies" → "eus numerous" (from "EU's numerous deficiencies")
- "ambition" → "ever-growing" (from "ever-growing ambition")
- "player" → "formidable", "global" (from "formidable global player")
- "peace" → "guarantor" (from "guarantor of peace")
- "liability" → "major" (from "major liability")
- "exception and concept" → "politics" (from "politics of the exception and the concept")
- "war" → "russian", "times" (from "Russian war", "times of war")
- "challenge" → "serious" (from "serious challenge")
- "power europe" → "soft" (from "soft power Europe")
- "governance" → "technocratic" (from "technocratic governance")

## Implementation Design

### Module: `concept_tree/concept_tree.py`

```python
class ConceptTreeBuilder:
    """
    Build a concept tree from text using the CiteSpace algorithm.
    """

    def __init__(self, nlp):
        self.nlp = nlp
        self.concepts = {}  # head -> {modifiers: {mod: contexts}, contexts: [], freq: int}

    def process_text(self, text: str, source: str = "p") -> None:
        """Process text and extract concepts."""
        pass

    def extract_noun_phrases(self, doc) -> list[NounPhrase]:
        """Extract noun phrases from a spaCy doc."""
        pass

    def split_head_modifier(self, np: NounPhrase) -> tuple[str, str]:
        """Split noun phrase into head noun and modifier."""
        pass

    def add_to_tree(self, head: str, modifier: str, context: str, source: str):
        """Add concept to the tree structure."""
        pass

    def to_treeml(self, output_path: Path) -> None:
        """Export tree to TreeML XML format."""
        pass
```

### Key Observations from CiteSpace Output

1. **Proper Noun Handling**: Multi-word proper nouns like "European Union" and "Carl Schmitt" are kept together as single concepts
2. **Possessives**: "EU's" becomes "EUs" (apostrophe stripped)
3. **Article Stripping**: "a", "an", "the" are omitted from tree
4. **Hyphenated Words**: Kept intact (e.g., "ever-growing", "self-understanding")
5. **"Of" Patterns**: "X of Y" → Y becomes head, X becomes modifier (e.g., "guarantor of peace" → peace → guarantor)
6. **Context Format**: HTML-escaped `<ul><li>[link]sentence</li></ul>`

## Dependencies

- spaCy with `en_core_web_sm` or `en_core_web_md` model
- xml.etree.ElementTree for XML generation

## Testing Strategy

1. Process `input.txt`
2. Compare generated XML structure with `concept_tree.xml`
3. Verify:
   - Same concepts extracted
   - Same head/modifier relationships
   - Same frequencies
   - Contexts contain correct sentences

## Files to Create

1. `concept_tree/concept_tree.py` - Main implementation
2. `concept_tree/test_concept_tree.py` - Test against expected output
