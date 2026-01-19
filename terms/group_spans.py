"""Span grouping and redundancy resolution for key noun phrase extraction.

This module provides functionality to identify maximal spans and group
nested (sub)spans into families for deduplication purposes.

Example: Given the phrase "robust data security incident response plan",
multiple overlapping spans may be extracted:
    - "data security" (tokens 1-3)
    - "data security incident" (tokens 1-4)
    - "security incident" (tokens 2-4)
    - "data security incident response plan" (tokens 1-6) <- maximal

The maximal span contains all others, and subspans are grouped under it.
"""

from dataclasses import dataclass


@dataclass
class Span:
    """A token span with metadata.

    Attributes:
        start: Start token index (inclusive)
        end: End token index (exclusive)
        text: The surface text of the span
        label: The POS pattern label (e.g., "ADJ-NOUN")
        doc_index: The index of the document in the input list
    """

    start: int
    end: int
    text: str
    label: str
    doc_index: int

    @property
    def length(self) -> int:
        """Return the span length in tokens."""
        return self.end - self.start

    def __hash__(self):
        """Make Span hashable for use in sets."""
        return hash((self.start, self.end, self.text, self.label, self.doc_index))


def contains(a: Span, b: Span) -> bool:
    """Check if span A contains span B (A >= B).

    A span A contains span B if A's token range fully encompasses B's range.
    A span contains itself.

    Parameters
    ----------
    a : Span
        The potentially containing span
    b : Span
        The potentially contained span

    Returns
    -------
    bool
        True if A contains B
    """
    return a.start <= b.start and a.end >= b.end


def strictly_contains(a: Span, b: Span) -> bool:
    """Check if span A strictly contains span B (A > B).

    A span A strictly contains span B if A contains B and A != B
    (i.e., A is strictly larger).

    Parameters
    ----------
    a : Span
        The potentially containing span
    b : Span
        The potentially contained span

    Returns
    -------
    bool
        True if A strictly contains B
    """
    return contains(a, b) and (a.start < b.start or a.end > b.end)


def find_maximal_spans(spans: list[Span]) -> list[Span]:
    """Find all maximal spans (spans not contained by any other span).

    A span is maximal if no other span strictly contains it.

    Parameters
    ----------
    spans : list[Span]
        List of spans to analyze

    Returns
    -------
    list[Span]
        List of maximal spans, sorted by start position
    """
    if not spans:
        return []

    maximal = []
    for candidate in spans:
        is_maximal = True
        for other in spans:
            if other is not candidate and strictly_contains(other, candidate):
                is_maximal = False
                break
        if is_maximal:
            maximal.append(candidate)

    return sorted(maximal, key=lambda s: (s.start, -s.end))


def build_families(spans: list[Span]) -> dict[str, dict]:
    """Group spans into families based on containment relationships.

    Each maximal span becomes the head of a family. Non-maximal spans are
    assigned to the smallest maximal span that contains them.

    Parameters
    ----------
    spans : list[Span]
        List of spans to group

    Returns
    -------
    dict[str, dict]
        Dictionary mapping family_id to family data:
        {
            "family_1": {
                "maximal": Span,
                "members": [Span, ...],  # includes maximal span itself
            },
            ...
        }

        Spans that don't belong to any family (e.g., crossing spans that
        don't nest properly) are placed in a special "_unrelated" key.
    """
    if not spans:
        return {}

    maximal_spans = find_maximal_spans(spans)

    # Create family for each maximal span
    families = {}
    for i, maximal in enumerate(maximal_spans):
        family_id = f"family_{i + 1}"
        families[family_id] = {
            "maximal": maximal,
            "members": [maximal],
        }

    # Assign non-maximal spans to families
    unrelated = []
    for span in spans:
        if span in maximal_spans:
            continue

        # Find the smallest maximal span that contains this span
        containing_maximals = [
            (fam_id, fam_data)
            for fam_id, fam_data in families.items()
            if contains(fam_data["maximal"], span)
        ]

        if containing_maximals:
            # Assign to smallest containing maximal (by length)
            containing_maximals.sort(key=lambda x: x[1]["maximal"].length)
            best_family_id = containing_maximals[0][0]
            families[best_family_id]["members"].append(span)
        else:
            # Span doesn't fit into any family (crossing span)
            unrelated.append(span)

    if unrelated:
        families["_unrelated"] = {
            "maximal": None,
            "members": unrelated,
        }

    return families


def annotate_spans(
    results: list[dict],
) -> list[dict]:
    """Annotate extraction results with maximal/family information.

    Takes the raw results from yield_key_phrases and fills in the
    is_maximal, family_id, and maximal_text fields.

    Parameters
    ----------
    results : list[dict]
        List of result dicts from yield_key_phrases, each containing
        at minimum: doc_index, token_span, key_noun_phrase, pos_label

    Returns
    -------
    list[dict]
        Same results with is_maximal, family_id, maximal_text populated
    """
    if not results:
        return []

    # Group results by doc_index (sentence)
    by_doc_index: dict[int, list[dict]] = {}
    for r in results:
        doc_index = r["doc_index"]
        if doc_index not in by_doc_index:
            by_doc_index[doc_index] = []
        by_doc_index[doc_index].append(r)

    # Process each sentence's spans
    annotated = []
    for doc_index, sentence_results in by_doc_index.items():
        # Convert to Span objects
        spans = [
            Span(
                start=r["token_span"][0],
                end=r["token_span"][1],
                text=r["key_noun_phrase"],
                label=r["pos_label"],
                doc_index=doc_index,
            )
            for r in sentence_results
        ]

        # Build families
        families = build_families(spans)

        # Create lookup from span to family info
        span_to_family = {}
        for family_id, family_data in families.items():
            maximal = family_data["maximal"]
            maximal_text = maximal.text if maximal else None
            for member in family_data["members"]:
                span_key = (member.start, member.end, member.text)
                is_maximal = member is maximal
                span_to_family[span_key] = {
                    "is_maximal": is_maximal,
                    "family_id": family_id if family_id != "_unrelated" else None,
                    "maximal_text": maximal_text,
                }

        # Annotate original results
        for r in sentence_results:
            span_key = (r["token_span"][0], r["token_span"][1], r["key_noun_phrase"])
            family_info = span_to_family.get(span_key, {})
            r["is_maximal"] = family_info.get("is_maximal")
            r["family_id"] = family_info.get("family_id")
            r["maximal_text"] = family_info.get("maximal_text")
            annotated.append(r)

    return annotated
