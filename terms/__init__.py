# -*- coding: utf-8 -*-
"""HTU's implementation of Key Noun Phrase extraction."""
import json
import typing
from pathlib import Path

import pandas as pd
import spacy
from spacy.matcher import Matcher
from spacy.symbols import VERB, nsubj, nsubjpass


class Rule(typing.NamedTuple):
    """Patterns structure."""

    label: str
    pattern: typing.List[typing.List[typing.Dict[str, typing.Any]]]


def read_pattern(path: Path) -> typing.List[Rule]:
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

    def __init__(
        self, nlp: spacy.language.Language, matcher: typing.Optional[Matcher] = None
    ):
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
        self._default_patterns = (
            Path(__file__).resolve().parent / "assets" / "default_patterns.json"
        )
        self.matcher = (
            matcher
            if matcher is not None
            else build_matcher(nlp, self._default_patterns)
        )

    def yield_key_phrases(
        self,
        sentences: typing.List[typing.Tuple[str, str]],
        batch_size: int = 25,
        exclusive_search: bool = False,
    ) -> typing.Iterator[typing.Dict[str, typing.Any]]:
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
            whether to yield phrases with nsubj being part of them (True) or
            to yield any phrases found within the nsubj's subtree (even without
            nsubj token being a part of the phrase) (False)

        Usage
        -----
        >>> from spacy.lang.ru.examples import sentences
        >>> nlp = spacy.load("ru_core_news_md")
        >>> terms = TermsMatcher(nlp=nlp)
        >>> transformed_sentences = [(sent, idx) for idx, sent in enumerate(sentences)]
        >>> for key_noun_phrase in terms.yield_key_phrases(transformed_sentences):
        ...     print(key_noun_phrase)
        ...

        Notes
        -----
        The idea behind _key_ noun phrase is that we only care about phrases
        that stem from the token which has nsubj dependency tag and is whose
        head token has VERB pos tag (as it's more imporant than other phrases).
        Thus, we limit the context -- from the full document to a limited
        subtree or even token's direct children -- within which we're going to 
        match phrases according to our pos combinations.
        """
        for sentence, uuid in self.nlp.pipe(
            sentences, as_tuples=True, batch_size=batch_size
        ):
            for possible_subject in sentence:
                if (
                    possible_subject.dep in [nsubj, nsubjpass]
                    and possible_subject.head.pos == VERB
                ):
                    subtree = sentence[
                        possible_subject.left_edge.i : possible_subject.right_edge.i + 1
                    ]
                    for match_id, start, end in self.matcher(subtree):
                        span = subtree[start:end]
                        if exclusive_search and not possible_subject in span:
                            continue
                        yield {
                            "uuid": uuid,
                            "pos_label": self.nlp.vocab[match_id].text,
                            "key_noun_phrase": span.text,
                            "key_noun_phrase_processed": " ".join(
                                t.lemma_.lower() for t in span if not t.is_punct
                            ),
                            "span_location": [span.start_char, span.end_char],
                        }

    def to_dataframe(
        self, sentences: typing.List[typing.Tuple[str, str]]
    ) -> pd.DataFrame:
        """Constructs a dataframe directly from yield_key_phrases method.

        Parameters
        ----------
        sentences: list[tuple[uuid, text]]
            list of pairs, each consisting of text and its identifier (so that
            we could 'place' exact phrase within some context (found by uuid);
            it must follow this structure: [("Some text", "uuid1"), ("Another sentence", "uuid2"), ...]

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
        return pd.DataFrame(self.yield_key_phrases(sentences))
