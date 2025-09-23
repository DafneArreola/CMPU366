#!/usr/bin/env python3

from collections import Counter
from pathlib import Path

from spacy.lang.en import English
from spacy.tokens import Doc

nlp = English(pipeline=[], max_length=5000000)


def make_docs(files: list[str | Path]) -> Doc:
    """Make a spaCy Doc out of the text in a list of file names."""
    docs = []
    for fname in files:
        with open(fname, "r", encoding="latin1") as f:
            text = f.read().replace("--", " -- ")
            docs.append(nlp(text))
    return Doc.from_docs(docs)


def get_unigrams(doc: Doc) -> list[str]:
    """Return a list of unigrams (individual tokens) in the given spaCy
    Doc, normalized to be lowercase."""
    return [t.text.lower() for t in doc if not t.text.isspace()]


def test_get_unigrams():
    assert get_unigrams(nlp("")) == []
    assert get_unigrams(nlp("hello world!")) == ["hello", "world", "!"]


def get_bigrams(doc: Doc) -> list[tuple[str, str]]:
    """Return a list of all bigrams in the given spaCy doc."""
    unigrams = get_unigrams(doc)
    return list(zip(unigrams[:-1], unigrams[1:]))


def test_get_bigrams():
    assert get_bigrams(nlp("")) == []
    doc = nlp("Hello darkness\nmy old friend")
    bigrams = [
        ("hello", "darkness"),
        ("darkness", "my"),
        ("my", "old"),
        ("old", "friend")
    ]
    assert get_bigrams(doc) == bigrams


def main():
    train_files = [
        "gutenberg/austen-emma.txt",
        "gutenberg/austen-persuasion.txt"
    ]
    train = make_docs(train_files)

    unigrams = get_unigrams(train)
    unigram_freqs = Counter(unigrams)

    bigrams = get_bigrams(train)
    bigram_freqs = Counter(bigrams)

    # Add your code here!
    ...


if __name__ == "__main__":
    main()
