#!/usr/bin/env python3

from typing import TextIO

import click

from tokenizer import tokenize


def baseline_segmenter(tokens: list[str]) -> list[list[str]]:
    """Segment the token list into a list of sentences, breaking every time
    we encounter one of the tokens . : ; ! ?
    """
    sentences = []
    current_sentence = []

    for token in tokens:
        current_sentence.append(token)
        if token in {".", ":", ";", "!", "?"}:
            sentences.append(current_sentence)
            current_sentence = []

    # Add any remaining tokens as final sentence
    if current_sentence:
        sentences.append(current_sentence)

    return sentences

def test_baseline_segmenter():
    result = baseline_segmenter(
        ["I", "am", "Sam", ".", "Sam", "I", "am", "."]
    )
    expected = [["I", "am", "Sam", "."], ["Sam", "I", "am", "."]]
    assert result == expected


def improved_segmenter(tokens: list[str]) -> list[list[str]]:
    """TODO: Replace this with an improved sentence segmenter."""
    ...

def test_improved_segmenter():
    ...


def write_sentence_boundaries(sentences: list[list[str]], out: TextIO):
    """TODO: Write out the token numbers of the sentence boundaries."""
    ...


@click.command()
@click.option(
    "-t",
    "--textfile",
    type=click.File("r"),
    required=True,
    help="Unlabeled text file.",
)
@click.option(
    "-y",
    "--hypothesis-file",
    type=click.File("w"),
    default="-",
    help="Write hypothesized boundaries to FILENAME (default: stdout).",
)
def main(textfile, hypothesis_file):
    """Segments text into sentences."""
    ...


if __name__ == "__main__":
    main()
