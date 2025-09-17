#!/usr/bin/env python3
import re
from pathlib import Path
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

def tokenize(text: str, lower: bool = False) -> list[str]:
    """Tokenize the input text into a list of tokens. Tokens are defined as
    contiguous sequences of characters (letters and digits).
    Optionally lowercase all tokens.
    """
    if lower:
        text = text.lower()

    return re.findall(r"\w+|[^\w\s]", text)

def filternonwords(tokens: list[str]) -> list[str]:
    """Filter out any tokens that are not purely alphanumeric."""
    return [t for t in tokens if t.isalpha()]

def main():
    """All of your testing code should go in here. This code is only run if
    your program is run directly. If it's imported into another file
    (like segmenter.py), then this code will not run.
    """
    carroll_text = open("gutenberg/carroll-alice.txt").read()
    
    carroll_frequency = words_by_frequency(get_words(carroll_text, lower = True), n=20)
    #print(carroll_frequency)

    carroll_tokens = tokenize(carroll_text, lower = True)
    carroll_token_frequency = words_by_frequency(carroll_tokens, n=5)
    #print(carroll_token_frequency)

    carroll_alpha_tokens = filternonwords(carroll_tokens)
    carroll_alpha_token_frequency = words_by_frequency(carroll_alpha_tokens, n=5)
    #print(carroll_alpha_token_frequency)

    all_words = []
    
    for book in Path("gutenberg").iterdir():
        with open(book, encoding="latin1") as f:
            text = f.read()
            tokens = filternonwords(tokenize(text, lower=True))
            token_freq = words_by_frequency(tokens, n=5)
            print(f"\nTop 5 words in {book.name}: {token_freq}")
        
        tokens = filternonwords(tokenize(text, lower=False))
        all_words.extend(tokens)

    all_word_freq = words_by_frequency(all_words, n=10)
    # print(f"\nTop 10 words in all books: {all_word_freq}")


if __name__ == "__main__":
    main()
