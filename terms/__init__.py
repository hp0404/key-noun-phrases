"""HTU's implementation of Key Noun Phrase extraction."""

import json
import typing
from enum import Enum
from pathlib import Path

import pandas as pd
import spacy
from spacy.matcher import Matcher
from spacy.symbols import AUX, VERB, dobj, iobj, nsubj, nsubjpass, obj

from .group_spans import annotate_spans
from .score import CompositeScorer, Scorer, score_phrases
from .treebank import is_vbg, is_vbn

# German TIGER treebank uses different dependency labels
# sb = subject, sbp = passive subject
GERMAN_SUBJECT_DEPS = {"sb", "sbp"}
# German object dependencies
GERMAN_OBJECT_DEPS = {"oa", "oa2", "og", "da"}  # accusative, genitive, dative objects


class ExtractionScope(str, Enum):
    """Scope for key noun phrase extraction.

    Controls which parts of the sentence are searched for noun phrases.

    Attributes
    ----------
    SUBJECT : str
        Search only within subject subtrees (tokens with nsubj/nsubjpass
        dependency whose head is a verb). This is the most restrictive
        mode and focuses on "key" noun phrases that are grammatical subjects.
    OBJECT : str
        Search within both subject AND object subtrees (tokens with
        obj/dobj/iobj dependencies). Captures more phrases including
        direct and indirect objects.
    SENTENCE : str
        Search the entire sentence without subtree restrictions.
        Matches all patterns found anywhere in the sentence.
    """

    SUBJECT = "subject"
    OBJECT = "object"
    SENTENCE = "sentence"


class Rule(typing.NamedTuple):
    """Patterns structure."""

    label: str
    pattern: list[list[dict[str, typing.Any]]]


def read_pattern(path: Path) -> list[Rule]:
    """Reads patterns JSON file."""
    with path.open("r", encoding="utf-8") as file_content:
        content = json.load(file_content)
    return [Rule(label=p["label"], pattern=p["pattern"]) for p in content]


def build_matcher(nlp: spacy.language.Language, patterns: Path) -> Matcher:
    """Builds custom matcher.

    Parameters
    ----------
    nlp: spacy.language.Language
        The matcher will operate on the vocabulary object (spacy's model vocab
        attribute)
    patterns: Path
        Path to a .json file with predefined POS patterns; it should follow
        this schema:
        [
            {
                "label": "ADJ-NOUN",
                "pattern": [
                    [{"POS": "ADJ"}, {"POS": "NOUN"}]
                ]
            },
            {
                "label": "ADJ-ADJ-NOUN",
                "pattern": [
                    [{"POS": "ADJ"}, {"POS": "ADJ"}, {"POS": "NOUN"}]
                ]
            }
        ]
    """
    matcher = Matcher(nlp.vocab)
    combinations = read_pattern(patterns)
    for combination in combinations:
        matcher.add(combination.label, combination.pattern)
    return matcher


class TermsMatcher:
    """Key Noun Phrase matcher."""

    def __init__(self, nlp: spacy.language.Language, matcher: Matcher | None = None):
        """Initializes TermsMatcher class.

        Parameters
        ----------
        nlp: spacy.language.Language
            spacy's model
        matcher: spacy.match.Matcher
            spacy's rule-based Matcher;
            defaults to our own implementation but could be replaced with a custom one
        """
        self.nlp = nlp
        assets_dir = Path(__file__).resolve().parent / "assets"
        lang_patterns = {
            "de": "de_patterns.json",
            "en": "en_patterns.json",
            "ru": "ru_patterns.json",
            "uk": "uk_patterns.json",
        }
        pattern_file = lang_patterns.get(self.nlp.lang, "en_patterns.json")
        self._default_patterns = assets_dir / pattern_file
        self.matcher = (
            matcher
            if matcher is not None
            else build_matcher(nlp, self._default_patterns)
        )

    def yield_key_phrases(
        self,
        sentences: list[tuple[str, str]],
        batch_size: int = 25,
        exclusive_search: bool = True,
        scope: ExtractionScope = ExtractionScope.SUBJECT,
    ) -> typing.Iterator[dict[str, typing.Any]]:
        """Yields key noun phrases found in sentences.

        Parameters
        ----------
        sentences: list[tuple[uuid, text]]
            list of pairs, each consisting of text and its identifier (so that
            we could 'place' exact phrase within some context (found by uuid);
            it must follow this structure: [("Some text", "uuid1"), ("Another sentence", "uuid2"), ...]
        batch_size: int
            the number of texts to buffer
        exclusive_search: bool
            whether to yield
              - phrases with nsubj being part of them (True) and
              - VERB-based phrases with finegrained VERB subtypes (True)
            to yield any phrases found within the nsubj's subtree (even without
            nsubj token being a part of the phrase) (False) and any VERB-based phrases
            Note: Only applies when scope is SUBJECT or OBJECT, not SENTENCE.
        scope: ExtractionScope
            Controls which parts of the sentence are searched:
              - SUBJECT: Only search within subject subtrees (default, original behavior)
              - OBJECT: Search within both subject and object subtrees
              - SENTENCE: Search the entire sentence without restrictions

        Usage
        -----
        >>> from spacy.lang.ru.examples import sentences
        >>> nlp = spacy.load("ru_core_news_md")
        >>> terms = TermsMatcher(nlp=nlp)
        >>> transformed_sentences = [(sent, idx) for idx, sent in enumerate(sentences)]
        >>> for key_noun_phrase in terms.yield_key_phrases(transformed_sentences):
        ...     print(key_noun_phrase)
        ...

        Yields
        ------
        dict with the following fields:
            uuid: str
                The identifier passed in with the sentence
            pos_label: str
                The POS pattern label that matched (e.g., "ADJ-NOUN")
            key_noun_phrase: str
                The matched phrase text
            key_noun_phrase_processed: str
                Lemmatized, lowercased version of the phrase (punctuation removed)
            span_location: list[int, int]
                Character offsets [start_char, end_char] in the original text
            token_span: list[int, int]
                Token indices [start, end) in the document
            is_maximal: bool | None
                Whether this span is maximal (not contained by another span).
                Placeholder; computed in redundancy resolution phase.
            family_id: str | None
                ID of the maximal span that contains this span.
                Placeholder; computed in redundancy resolution phase.
            maximal_text: str | None
                Text of the maximal span that contains this span.
                Placeholder; computed in redundancy resolution phase.

        Notes
        -----
        The idea behind _key_ noun phrase is that we only care about phrases
        that stem from the token which has nsubj dependency tag and whose
        head token has VERB pos tag (as it's (arguably) more important than other phrases).
        Thus, we limit the context -- from the full document to a limited
        subtree or even token's direct children -- within which we're going to
        match phrases according to our pos combinations.

        With scope=OBJECT, object subtrees are also searched, capturing phrases
        like direct objects ("бойових втрат") that would otherwise be missed.

        With scope=SENTENCE, no subtree restriction is applied, and all matching
        patterns in the sentence are extracted.
        """
        # Handle string input for scope (for convenience)
        if isinstance(scope, str):
            scope = ExtractionScope(scope)

        for sentence, uuid in self.nlp.pipe(
            sentences, as_tuples=True, batch_size=batch_size
        ):
            if scope == ExtractionScope.SENTENCE:
                # Search the entire sentence
                yield from self._match_span(
                    sentence,
                    sentence[:],
                    uuid,
                    exclusive_search=False,  # No anchor token for sentence scope
                    anchor_token=None,
                )
            else:
                # Track which spans we've already yielded to avoid duplicates
                seen_spans: set[tuple[int, int]] = set()

                for token in sentence:
                    # Check for subject dependency (Universal or German TIGER)
                    is_subject = (
                        token.dep in [nsubj, nsubjpass]
                        or token.dep_ in GERMAN_SUBJECT_DEPS
                    )
                    # Check for object dependency (Universal or German TIGER)
                    is_object = (
                        token.dep in [obj, dobj, iobj]
                        or token.dep_ in GERMAN_OBJECT_DEPS
                    )
                    # Check if head is a verb (VERB or AUX for copular constructions)
                    head_is_verb = token.head.pos in [VERB, AUX]

                    should_search = False
                    if scope == ExtractionScope.SUBJECT:
                        should_search = is_subject and head_is_verb
                    elif scope == ExtractionScope.OBJECT:
                        should_search = (is_subject or is_object) and head_is_verb

                    if should_search:
                        subtree = sentence[
                            token.left_edge.i : token.right_edge.i + 1
                        ]
                        for result in self._match_span(
                            sentence,
                            subtree,
                            uuid,
                            exclusive_search,
                            anchor_token=token,
                        ):
                            span_key = tuple(result["token_span"])
                            if span_key not in seen_spans:
                                seen_spans.add(span_key)
                                yield result

    def _match_span(
        self,
        sentence: spacy.tokens.Doc,
        subtree: spacy.tokens.Span,
        uuid: str,
        exclusive_search: bool,
        anchor_token: spacy.tokens.Token | None,
    ) -> typing.Iterator[dict[str, typing.Any]]:
        """Match patterns within a span and yield results.

        Parameters
        ----------
        sentence: spacy.tokens.Doc
            The full sentence document
        subtree: spacy.tokens.Span
            The span to search within
        uuid: str
            Document identifier
        exclusive_search: bool
            If True and anchor_token is provided, only yield phrases containing
            the anchor token and with proper verb forms
        anchor_token: spacy.tokens.Token | None
            The subject/object token that anchors this subtree (None for sentence scope)
        """
        for match_id, start, end in self.matcher(subtree):
            span = subtree[start:end]
            pos_label = self.nlp.vocab[match_id].text
            if exclusive_search and anchor_token is not None:
                # phrases should stem from anchor directly
                if anchor_token not in span:
                    continue

                # VERB-based phrases should be of specific finegrained pos
                if "VERB" in pos_label and not any(
                    is_vbg(token) or is_vbn(token) for token in span
                ):
                    continue
            yield {
                "uuid": uuid,
                "pos_label": pos_label,
                "key_noun_phrase": span.text,
                "key_noun_phrase_processed": " ".join(
                    t.lemma_.lower() for t in span if not t.is_punct
                ),
                "span_location": [span.start_char, span.end_char],
                "token_span": [span.start, span.end],
                "is_maximal": None,
                "family_id": None,
                "maximal_text": None,
            }

    def extract_key_phrases(
        self,
        sentences: list[tuple[str, str]],
        batch_size: int = 25,
        exclusive_search: bool = True,
        scope: ExtractionScope = ExtractionScope.SUBJECT,
        resolve_redundancy: bool = True,
        compute_scores: bool = False,
        scorer: "Scorer | None" = None,
    ) -> list[dict[str, typing.Any]]:
        """Extract key noun phrases with optional redundancy resolution and scoring.

        This method collects all results and optionally annotates them with
        maximal span and family information, and computes importance scores.

        Parameters
        ----------
        sentences: list[tuple[uuid, text]]
            List of (text, uuid) pairs
        batch_size: int
            The number of texts to buffer
        exclusive_search: bool
            Whether to filter for phrases containing the subject/object token.
            Only applies when scope is SUBJECT or OBJECT.
        scope: ExtractionScope
            Controls which parts of the sentence are searched:
              - SUBJECT: Only search within subject subtrees (default)
              - OBJECT: Search within both subject and object subtrees
              - SENTENCE: Search the entire sentence without restrictions
        resolve_redundancy: bool
            If True, annotate results with is_maximal, family_id, maximal_text.
            If False, these fields remain None.
        compute_scores: bool
            If True, compute importance scores for each phrase.
            If False, no 'score' field is added.
        scorer: Scorer | None
            Custom scorer to use. Defaults to CompositeScorer if compute_scores
            is True and no scorer is provided.

        Returns
        -------
        list[dict]
            List of result dictionaries with all fields populated
        """
        results = list(
            self.yield_key_phrases(
                sentences,
                batch_size=batch_size,
                exclusive_search=exclusive_search,
                scope=scope,
            )
        )
        if resolve_redundancy and results:
            results = annotate_spans(results)
        if compute_scores and results:
            results = score_phrases(results, scorer=scorer)
        return results

    def to_dataframe(
        self,
        sentences: list[tuple[str, str]],
        batch_size: int = 25,
        exclusive_search: bool = True,
        scope: ExtractionScope = ExtractionScope.SUBJECT,
        resolve_redundancy: bool = True,
        compute_scores: bool = False,
        scorer: "Scorer | None" = None,
    ) -> pd.DataFrame:
        """Constructs a dataframe from key phrase extraction.

        Parameters
        ----------
        sentences: list[tuple[uuid, text]]
            list of pairs, each consisting of text and its identifier (so that
            we could 'place' exact phrase within some context (found by uuid);
            it must follow this structure: [("Some text", "uuid1"), ("Another sentence", "uuid2"), ...]
        batch_size: int
            the number of texts to buffer
        exclusive_search: bool
            whether to yield
              - phrases with nsubj being part of them (True) and
              - VERB-based phrases with finegrained VERB subtypes (True)
            to yield any phrases found within the nsubj's subtree (even without
            nsubj token being a part of the phrase) (False) and any VERB-based phrases
            Note: Only applies when scope is SUBJECT or OBJECT.
        scope: ExtractionScope
            Controls which parts of the sentence are searched:
              - SUBJECT: Only search within subject subtrees (default)
              - OBJECT: Search within both subject and object subtrees
              - SENTENCE: Search the entire sentence without restrictions
        resolve_redundancy: bool
            If True, annotate results with is_maximal, family_id, maximal_text.
            If False, these fields remain None.
        compute_scores: bool
            If True, compute importance scores for each phrase.
            If False, no 'score' field is added.
        scorer: Scorer | None
            Custom scorer to use. Defaults to CompositeScorer if compute_scores
            is True and no scorer is provided.

        Usage
        -----
        >>> from spacy.lang.ru.examples import sentences
        >>> nlp = spacy.load("ru_core_news_md")
        >>> terms = TermsMatcher(nlp=nlp)
        >>> transformed_sentences = [(sent, idx) for idx, sent in enumerate(sentences)]
        >>> df = terms.to_dataframe(transformed_sentences)

        Notes
        -----
        Intended usage for this method is when you have a small enough
        set of sentences so that you don't need to store them in the interim
        format like JSONLines.
        """
        data = self.extract_key_phrases(
            sentences,
            batch_size=batch_size,
            exclusive_search=exclusive_search,
            scope=scope,
            resolve_redundancy=resolve_redundancy,
            compute_scores=compute_scores,
            scorer=scorer,
        )
        return pd.DataFrame(data)
