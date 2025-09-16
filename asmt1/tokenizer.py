#!/usr/bin/env python3

from collections import Counter


def get_words(text: str, lower: bool = False) -> list[str]:
    """Return a list of the words in the input string, splitting at spaces.
    Optionally lowercase all tokens.
    """
    if lower:
        text = text.lower()
    return text.split()

def test_get_words():
    sentence = "The cat in the hat ate the rat in the vat"
    words = [
        "The", "cat", "in", "the", "hat", "ate", "the", "rat", "in",
        "the", "vat"
    ]
    words_lower = [
        "the", "cat", "in", "the", "hat", "ate", "the", "rat", "in",
        "the", "vat"
    ]
    assert get_words(sentence) == words
    assert get_words(sentence, lower=True) == words_lower


def count_words(words: list[str]) -> Counter:
    """Return a Counter that maps a word to the frequency that it occurs
    in the list of words.
    """
    return Counter(words)

def test_count_words():
    words = [
        "The", "cat", "in", "the", "hat", "ate", "the", "rat", "in",
        "the", "vat"
    ]
    counts = Counter(
        {
            "the": 3,
            "in": 2,
            "The": 1,
            "cat": 1,
            "hat": 1,
            "ate": 1,
            "rat": 1,
            "vat": 1,
        }
    )

    assert count_words(words) == counts


def words_by_frequency(words: list[str], n=None) -> list[tuple[str, int]]:
    """Return a list of (word, count) tuples sorted by count, such that
    the first item in the list is the most frequent item.
    """
    counts = count_words(words)
    return counts.most_common(n)

def test_words_by_frequency():
    words = [
        "the", "cat", "in", "the", "hat", "ate", "the", "rat", "in",
        "the", "vat"
    ]

    word_freq = [
        ("the", 4),
        ("in", 2),
        ("cat", 1),
        ("hat", 1),
        ("ate", 1),
        ("rat", 1),
        ("vat", 1),
    ]
    word_freq_3 = [("the", 4), ("in", 2), ("cat", 1)]

    assert words_by_frequency(words) == word_freq

    assert words_by_frequency(words, n=3) == word_freq_3


def main():
    """All of your testing code should go in here. This code is only run if
    your program is run directly. If it's imported into another file
    (like segmenter.py), then this code will not run.
    """
    ...


if __name__ == "__main__":
    main()
