#!/usr/bin/env python3

from collections import Counter
from pathlib import Path

from spacy.lang.en import English
from spacy.tokens import Doc

import math
import random

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

# -------------------
# Part 1
# -------------------

def get_trigrams(doc: Doc) -> list[tuple[str, str, str]]:
    """Return a list of all trigrams in order."""
    unigrams = get_unigrams(doc)
    if len(unigrams) < 3:
        return []
    return list(zip(unigrams[:-2], unigrams[1:-1], unigrams[2:]))


def test_get_trigrams():
    assert get_trigrams(nlp("")) == []
    doc = nlp("To be or not to be .")
    # tokenized as: ["to","be","or","not","to","be","."]
    trigrams = [
        ("to", "be", "or"),
        ("be", "or", "not"),
        ("or", "not", "to"),
        ("not", "to", "be"),
        ("to", "be", "."),
    ]
    assert get_trigrams(doc) == trigrams


def get_ngram_counts(doc: Doc) -> tuple[Counter, Counter, Counter]:
    """Return Counters for unigrams, bigrams, trigrams from the training doc."""
    unigrams = get_unigrams(doc)
    bigrams = list(zip(unigrams[:-1], unigrams[1:])) if len(unigrams) >= 2 else []
    trigrams = list(zip(unigrams[:-2], unigrams[1:-1], unigrams[2:])) if len(unigrams) >= 3 else []
    return Counter(unigrams), Counter(bigrams), Counter(trigrams)


def calc_ngram_prob(trigram: tuple[str, str, str], bigram_counts: Counter, trigram_counts: Counter) -> float:
    """Return log P(w_n | w_{n-2} w_{n-1}) using MLE.
    If either count is zero/unseen, return -math.inf (do NOT log(0))."""
    w1, w2, w3 = trigram
    c_tri = trigram_counts.get((w1, w2, w3), 0)
    c_bi = bigram_counts.get((w1, w2), 0)
    if c_tri == 0 or c_bi == 0:
        return -math.inf
    return math.log(c_tri / c_bi)

# -------------------
# Part 2
# -------------------

def _last_two_tokens(sequence: str) -> tuple[str, str] | None:
    """Tokenize a string and return its last two non-space, lowercased tokens (or None if not enough)."""
    toks = [t.text.lower() for t in nlp(sequence) if not t.text.isspace()]
    if len(toks) < 2:
        return None
    return (toks[-2], toks[-1])


def get_possible_next_words(sequence: str, bigram_counts: Counter, trigram_counts: Counter) -> list[tuple[str, float]]:
    """Return [(candidate_word, log_prob)] for possible next words after the sequence."""
    ctx = _last_two_tokens(sequence)
    if ctx is None:
        return []
    w1, w2 = ctx
    # Candidates are all third words of trigrams that start with (w1, w2)
    candidates: list[tuple[str, float]] = []
    # Iterate only over relevant trigrams by checking prefix
    for (a, b, c), tri_count in trigram_counts.items():
        if a == w1 and b == w2:
            lp = calc_ngram_prob((a, b, c), bigram_counts, trigram_counts)
            if lp != -math.inf:
                candidates.append((c, lp))
    return candidates


def predict_next_word(sequence: str, bigram_counts: Counter, trigram_counts: Counter) -> str:
    """Return the most likely next word; 'UNK' if no continuation."""
    candidates = get_possible_next_words(sequence, bigram_counts, trigram_counts)
    if not candidates:
        return "UNK"
    # max by log-probability
    return max(candidates, key=lambda x: x[1])[0]


def sample_next_word(sequence: str, bigram_counts: Counter, trigram_counts: Counter) -> str:
    """Sample a next word according to its probability; 'UNK' if no valid continuation or zero weights."""
    candidates = get_possible_next_words(sequence, bigram_counts, trigram_counts)
    if not candidates:
        return "UNK"
    words, logps = zip(*candidates)
    weights = [math.exp(lp) for lp in logps]  # convert back to normal probs
    if sum(weights) == 0.0:
        return "UNK"
    return random.choices(words, weights=weights, k=1)[0]


def generate_text(sequence: str, n: int, bigram_counts: Counter, trigram_counts: Counter, mode: str = "top") -> str:
    """Generate n words after sequence, using either 'top' (argmax) or 'random' sampling."""
    out_tokens = [t.text for t in nlp(sequence)]
    for _ in range(n):
        if mode == "random":
            next_word = sample_next_word(" ".join(out_tokens), bigram_counts, trigram_counts)
        else:
            next_word = predict_next_word(" ".join(out_tokens), bigram_counts, trigram_counts)
        out_tokens.append(next_word)
    # Simple spacing: join tokens with spaces, but you could improve detokenization if desired
    return " ".join(out_tokens)


# -------------------
# Part 3
# -------------------

def calc_text_perplexity(doc: Doc, bigram_counts: Counter, trigram_counts: Counter) -> float:
    """Compute perplexity of doc under the trigram model defined by bigram/trigram counts.

    - If doc has no trigrams, return 1.0
    - If any trigram has zero prob (log=-inf), return inf
    """
    trigrams = get_trigrams(doc)
    if not trigrams:
        return 1.0
    logps: list[float] = []
    for tg in trigrams:
        lp = calc_ngram_prob(tg, bigram_counts, trigram_counts)
        if lp == -math.inf:
            return float("inf")
        logps.append(lp)
    avg_logp = sum(logps) / len(logps)
    # perplexity = exp( - average log-prob )
    return math.exp(-avg_logp)


def add1_smoothing(test_doc: Doc, bigram_counts: Counter, trigram_counts: Counter) -> None:
    """Laplace (add-1) smoothing using the test_doc to introduce unseen n-grams."""
    # 1) add-one to existing trigrams
    for tg in list(trigram_counts.keys()):
        trigram_counts[tg] += 1

    # 2) add unseen test trigrams with count 1
    test_trigrams = get_trigrams(test_doc)
    for tg in set(test_trigrams):
        if tg not in trigram_counts:
            trigram_counts[tg] = 1

    # 3) for every trigram present, increment its starting bigram by 1
    for (w1, w2, _w3), count in trigram_counts.items():
        # add exactly +1 for each trigram instance (not by count), per instructions “increase the count for the bigram it starts with by 1”
        # The check-in description implies +1 per trigram type, not per frequency. If you want +count, change the += 1 to += count.
        bigram_counts[(w1, w2)] += 1


def main():
    train_files = [
        "gutenberg/austen-emma.txt",
        "gutenberg/austen-persuasion.txt",
    ]
    train = make_docs(train_files)

    unigram_freqs, bigram_freqs, trigram_freqs = get_ngram_counts(train)

    # Part 1.3:
    # print(calc_ngram_prob(("the","day","was"), bigram_freqs, trigram_freqs))
    
    # Part 2.2:
    # print(predict_next_word("an agreeable", bigram_freqs, trigram_freqs))
    
    # Part 2.3/2.4:
    # print(generate_text("an agreeable", 10, bigram_freqs, trigram_freqs, mode="top"))
    # print(generate_text("an agreeable", 10, bigram_freqs, trigram_freqs, mode="random"))
    # print(generate_text("she could not", 10, bigram_freqs, trigram_freqs, mode="random"))
    # print(generate_text("just as she", 10, bigram_freqs, trigram_freqs, mode="random"))
    
    # Part 3.1:
    # sense = make_docs(["gutenberg/austen-sense.txt"])
    # print(calc_text_perplexity(train, bigram_freqs, trigram_freqs))
    # print(calc_text_perplexity(sense, bigram_freqs, trigram_freqs))
    
    # Part 3.2:
    # bigram_sm = bigram_freqs.copy()
    # trigram_sm = trigram_freqs.copy()
    # add1_smoothing(sense, bigram_sm, trigram_sm)
    # Inspect first/last four bigrams as per handout instructions.

    pass

if __name__ == "__main__":
    main()
