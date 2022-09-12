import json
import typing
from pathlib import Path

import spacy
from spacy.symbols import nsubj, nsubjpass, VERB

import pandas as pd


class Rule(typing.NamedTuple):
    """Patterns structure."""

    label: str
    pattern: typing.List[typing.List[typing.Dict[str, typing.Any]]]


def read_pattern(path: Path) -> typing.List[Rule]:
    """Reads patterns JSON file."""
    with path.open("r", encoding="utf-8") as file_content:
        content = json.load(file_content)
    return [Rule(label=p["label"], pattern=p["pattern"]) for p in content]


def build_matcher(
    nlp: spacy.language.Language, patterns: Path
) -> spacy.matcher.Matcher:
    """Builds custom matcher."""
    matcher = spacy.matcher.Matcher(nlp.vocab)
    combinations = read_pattern(patterns)
    for combination in combinations:
        matcher.add(combination.label, combination.pattern)
    return matcher


class TermsMatcher:
    def __init__(
        self,
        nlp: spacy.language.Language,
        matcher: typing.Optional[spacy.matcher.Matcher] = None,
    ):
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
    ) -> typing.Iterator[typing.Dict[str, typing.Any]]:
        """ """
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
                        yield {
                            "uuid": uuid,
                            "label": self.nlp.vocab[match_id].text,
                            "phrase": span.text,
                            "phrase_norm": " ".join(
                                t.lemma_.lower() for t in span if not t.is_punct
                            ),
                            "location": [span.start_char, span.end_char],
                        }

    def to_dataframe(
        self, sentences: typing.List[typing.Tuple[str, str]]
    ) -> pd.DataFrame:
        return pd.DataFrame(self.yield_key_phrases(sentences))
