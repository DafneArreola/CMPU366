#!/usr/bin/env python3

# Assignment 3: Who Said It?
# CMPU 366, Fall 2025

import random

import numpy as np
from spacy.lang.en import English

nlp = English(pipeline=[])
nlp.add_pipe("sentencizer")


#
# Part 1
#


print("1. Loading Austen and Melville sentences: ", end="")


def get_sentences(fname: str) -> list[list[str]]:
    """Read the specified files as a list of sentences."""
    with open(fname, "r") as f:
        text = f.read()

    sents = []

    # We process the books in chunks by paragraph, ensuring that a
    # sentence never crosses a paragraph boundary:
    for para in text.split("\n\n"):
        para = para.replace("\n", " ")
        doc = nlp(para)
        for sent in doc.sents:
            sents.append(sent.text.strip())

    return sents


def test_get_sentences():
    assert get_sentences("/dev/null") == []
    emma_sents = get_sentences("austen-emma.txt")
    assert "FINIS" in emma_sents
    assert "CHAPTER II" in emma_sents


a_sents_all = get_sentences("austen-emma.txt")
m_sents_all = get_sentences("melville-moby_dick.txt")

print("done.\n")


print("2. Discarding short sentences and labeling: ", end="")

a_sents = [(s, "austen") for s in a_sents_all if len(s.split()) > 2]
m_sents = []  # EDIT
sents = []  # EDIT

print("done.\n")


print("3. Number of sentences:")
print(f"   Austen:   {len(a_sents):>5}")
print(f"   Melville: {len(m_sents):>5}")
print(f"   Total:    {len(sents):>5}")
print()


print("4. Shuffling and partitioning:")
# EDIT -- shuffle sents here
test_sents = []  # EDIT
devtest_sents = []  # EDIT
train_sents = []  # EDIT

print(f"   Training: {len(train_sents):>5}")
print(f"   Devtest:  {len(devtest_sents):>5}")
print(f"   Test:     {len(test_sents):>5}")
print()


print("5. Generating feature sets: ", end="")

# You can ignore this list until Part 2.
main_characters = [
    "emma",
    "harriet",
    "ahab",
    "weston",
    "knightley",
    "elton",
    "woodhouse",
    "jane",
    "stubb",
    "queequeg",
    "fairfax",
    "churchill",
    "frank",
    "starbuck",
    "pequod",
    "hartfield",
    "bates",
    "highbury",
    "perry",
    "bildad",
    "peleg",
    "pip",
    "cole",
    "goddard",
    "campbell",
    "donwell",
    "dixon",
    "taylor",
    "tashtego",
]

from sklearn.feature_extraction.text import CountVectorizer

# EDIT: Fit the vectorizer on the sentences

X_train = None  # EDIT
y_train = []  # EDIT

X_devtest = None  # EDIT
y_devtest = []  # EDIT

X_test = None  # EDIT
y_test = []  # EDIT

print("done.\n")


print("6. Training: ", end="")

whosaid = None  # EDIT - Train a LogisticRegression classifier here

print("done.\n")


print("7. Testing:")

# EDIT

print()


print("8. Sub-dividing development testing set: ", end="")


aa = []  # real author is Austen; predicted Austen
mm = []  # real author is Melville; predicted Melville
am = []  # real author is Austen; predicted Melville
ma = []  # real author is Melville; predicted Austen

# It's fine if you need to add code before this loop -- or modify the
# loop itself.
for sent, auth in devtest_sents:
    guess = "austen"  # EDIT to make this the classifier's actual guess
    if auth == "austen" and guess == "austen":
        aa.append((auth, guess, sent))
    # EDIT below to populate mm, am, ma


print("done.\n")


print("9. Sample correct and incorrect predictions from dev-test set:")

for x in (aa):  # EDIT change (aa) to (aa, mm, am, ma)
    print(x)
    auth, guess, sent = random.choice(x)
    print(f"   Author: {auth:10} Prediction: {guess}")
    print(sent)
    print()
print()


print("10. Looking up 40 most informative features:\n")

from sklearn.linear_model import LogisticRegression


def show_most_informative_features(
    classifier: LogisticRegression, vectorizer, n: int = 40
):
    """Print the features with the largest weights for each label.

    Logistic regression uses a weight vector for each class. For the binary
    case scikit-learn stores a single weight vector; the weights for the other
    class are its negation. We sort the weights to display the strongest
    positive indicators per class.
    """

    feature_names = vectorizer.get_feature_names_out()
    class_names = classifier.classes_

    coef = classifier.coef_

    if coef.shape[0] == 1 and len(class_names) == 2:
        class_coefs = np.vstack([-coef[0], coef[0]])
    else:
        class_coefs = coef

    for i, class_name in enumerate(class_names):
        weights = class_coefs[i]
        sorted_indices = np.argsort(weights)[::-1]

        print(f"    {class_name}:")
        for j in range(min(n, len(sorted_indices))):
            index = sorted_indices[j]
            name = feature_names[index]
            value = weights[index]
            print(f"{j + 1:>2}. {name:20} {value:>8.4f}")
        print()


# Modify this line to use the name you gave your vectorizer:
# show_most_informative_features(whosaid, count_vect)


#
# Part 2
#


# EDIT to include any code you need for answering the questions in Part 2.
