#!/usr/bin/env python3

from typing import TextIO

import click
import re

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
    # Terminators and helpers
    # Tokens that could terminate a sentence.
    terminating_punct = {".", ":", ";", "!", "?"}
    # Tokens that always end a sentence unless they're last-token edge cases.
    always_end = {"!", "?", ";"}
    # If a sentence ends, we attach these "closers" to the same sentence tail.
    closers_to_attach = {")", "]", "}", '"', "”", "’", "'"}

    # Common abbreviations that should NOT trigger sentence boundaries when followed by a dot
    abbreviations = re.compile(
        r"^(dr|mr|mrs|messrs|ms|prof|sr|jr|st|mt|vs|etc|fig|no|vol|rev|ed|"
        r"gov|sen|rep|col|gen|lt|atty|dept|adm|cmdr|sgt|capt|maj|"
        r"mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|"
        r"al)$",
        re.IGNORECASE,
    )
    # Single-letter initials like "A." or "b."
    initial_re = re.compile(r"^[A-Za-z]$") 
    # Digits
    digits = re.compile(r"^\d+$")

    # helper functions

    # previous token (or "" if none)
    def previous_token(i: int) -> str:
        return tokens[i - 1] if i - 1 >= 0 else ""

    # next alphabetic token (or None if none)
    def next_alpha(i: int) -> str | None:
        j = i + 1
        while j < len(tokens):
            if tokens[j].isalpha():
                return tokens[j]
            j += 1
        return None

    # next token (or None if none)
    def next_token(i: int) -> str | None:
        return tokens[i + 1] if i + 1 < len(tokens) else None

    # token type checks
    # is this token an abbreviation?
    def is_abbreviation(tok: str) -> bool:
        return bool(abbreviations.match(tok))

    # is this token a single-letter initial?
    def is_initial(tok: str) -> bool:
        return bool(initial_re.match(tok))

    # is this token a number?
    def is_number(tok: str) -> bool:
        return bool(digits.match(tok))

    # specific patterns for non-boundary cases
    # dot with single-letter both sides (e.g., U . S)
    def is_initialism_chain(i: int) -> bool:
        """Dot at i, with single-letter both sides (e.g., U . S)."""
        prev_tok = previous_token(i)
        nxt = next_token(i)
        return is_initial(prev_tok) and (is_initial(nxt) if nxt else False)

    # dot between two numbers (e.g., 3.14)
    def is_decimal(i: int) -> bool:
        prev_tok = previous_token(i)
        nxt = next_token(i)
        return is_number(prev_tok) and (is_number(nxt) if nxt else False)

    # dot in the middle of an ellipsis (e.g., "...")
    def is_ellipsis_mid(i: int) -> bool:
        nxt = next_token(i)
        return nxt == "."
    
    # dot at the end of a two-letter initialism (e.g., "U.S.")
    def is_two_letter_initialism_end(i: int) -> bool:
        return (
            i >= 3
            and is_initial(tokens[i - 1])
            and tokens[i - 2] == "."
            and is_initial(tokens[i - 3])
        )
    
    # dot after single-letter initial
    def is_middle_initial_case(i: int) -> bool:
        prev_tok = previous_token(i)
        nxt_alpha = next_alpha(i)
        return is_initial(prev_tok) and (nxt_alpha is not None and nxt_alpha[0].isupper() and len(nxt_alpha) > 1)

    # dot after single-letter initial, before another initial or uppercase word 
    def is_final_initialism_dot(i: int) -> bool:
        prev_tok = previous_token(i)
        if not is_initial(prev_tok):
            return False
        nxt = next_token(i)
        nxt_alpha = next_alpha(i)
        if nxt and is_initial(nxt):
            return True  # still inside chain
        if nxt in {"'", "’", '"', "”"}:
            return True
        if nxt_alpha is not None and nxt_alpha[0].isupper():
            return True
        return False

    # Determine if the token at index i is a sentence terminator
    def is_sentence_terminator(i: int) -> bool:
        tok = tokens[i]

        # Last token
        if i == len(tokens) - 1:
            return tok in terminating_punct

        # Always-enders
        if tok in always_end:
            return True

        # COLON: too many FPs in this corpus -> treat as NON-boundary.
        if tok == ":":
            return False

        # PERIODS
        if tok == ".":
            if is_ellipsis_mid(i):
                return False
            if is_decimal(i):
                return False
            if is_abbreviation(previous_token(i)):
                return False
            if is_two_letter_initialism_end(i):
                return False 
            if is_initialism_chain(i):
                return False
            if is_middle_initial_case(i):
                return False
            if is_final_initialism_dot(i):
                return False

            # Else: likely a boundary
            return True

        return False

    # main loop
    # create sentences by splitting at terminators
    sentences: list[list[str]] = []
    current: list[str] = []
    i = 0
    n = len(tokens)

    # Iterate through tokens
    while i < n:
        tok = tokens[i]
        current.append(tok)

        # Check for sentence boundary
        if tok in terminating_punct and is_sentence_terminator(i):
            # attach closers like quotes/parens to the same sentence
            j = i + 1
            while j < n and tokens[j] in closers_to_attach:
                current.append(tokens[j])
                j += 1
            # finalize current sentence
            sentences.append(current)
            current = []
            i = j
            continue

        i += 1

    # Add any remaining tokens as final sentence
    if current:
        sentences.append(current)

    return sentences

def test_improved_segmenter():
    tokens = tokenize("Dr. Smith went home. It was late.", lower=False)
    sents = improved_segmenter(tokens)
    assert len(sents) == 2 and sents[0][-1] == "." and sents[1][-1] == "."



def write_sentence_boundaries(sentences: list[list[str]], out: TextIO):
    """TODO: Write out the token numbers of the sentence boundaries."""
    offset = 0
    for sent in sentences:
        if not sent:
            # Skip empty sentence lists (shouldn't happen, but be safe)
            continue
        last_idx = offset + (len(sent) - 1)
        out.write(f"{last_idx}\n")
        offset += len(sent)


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
    tokenized = tokenize(textfile.read(), lower=False)
    sentences = improved_segmenter(tokenized)
    write_sentence_boundaries(sentences, hypothesis_file)   


if __name__ == "__main__":
    main()
