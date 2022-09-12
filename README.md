# key-noun-phrases

This repository contains HTU's implementation of Key Noun Phrase extraction 
(dependency & PoS tag parsing approach).


## Installation
```console
$ git clone https://github.com/hp0404/key-noun-phrases.git
$ cd key-noun-phrases
$ python3 -m venv env 
$ . env/bin/activate
$ pip install -e .
```

## Usage
```python
>>> from spacy.lang.ru.examples import sentences
>>> nlp = spacy.load("ru_core_news_md")
>>> terms = TermsMatcher(nlp=nlp)
>>> transformed_sentences = [(sent, idx) for idx, sent in enumerate(sentences)]
>>> df = terms.to_dataframe(transformed_sentences)
```

For more check out module's documentation!
