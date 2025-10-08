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

# Filter out sentences with 2 or fewer words, and label each remaining sentence
# with its author. This helps ensure the classifier is trained on meaningful text.
a_sents = [(s, "austen") for s in a_sents_all if len(s.split()) > 2]

# Do the same for Melville sentences
m_sents = [(s, "melville") for s in m_sents_all if len(s.split()) > 2]

# Combine both authors' sentences into a single list for further processing
sents = a_sents + m_sents

print("done.\n")


print("3. Number of sentences:")
print(f"   Austen:   {len(a_sents):>5}")
print(f"   Melville: {len(m_sents):>5}")
print(f"   Total:    {len(sents):>5}")
print()


print("4. Shuffling and partitioning:")
random.Random(10).shuffle(sents)
test_sents = sents[:1000]
devtest_sents = sents[1000:2000]
train_sents = sents[2000:]

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

count_vect = CountVectorizer()
# Fit the vectorizer on the training sentences to extract word features
X_train = count_vect.fit_transform([s for s, _ in train_sents])
y_train = [a for _, a in train_sents]

# Transform devtest and test sentences using the fitted vectorizer
X_devtest = count_vect.transform([s for s, _ in devtest_sents])
y_devtest = [a for _, a in devtest_sents]

X_test = count_vect.transform([s for s, _ in test_sents])
y_test = [a for _, a in test_sents]

print("done.\n")


print("6. Training: ", end="")

# Train a logistic regression classifier to distinguish between Austen and Melville
from sklearn.linear_model import LogisticRegression

whosaid = LogisticRegression(solver="lbfgs", max_iter=1000)
whosaid.fit(X_train, y_train)

print("done.\n")


print("7. Testing:")

# Evaluate the classifier on the test set and print a classification report
from sklearn.metrics import classification_report

y_pred = whosaid.predict(X_test)
print(classification_report(y_test, y_pred))

print()


print("8. Sub-dividing development testing set: ", end="")

# Partition devtest predictions into four groups for error analysis:
# aa: Austen sentences predicted as Austen
# mm: Melville sentences predicted as Melville
# am: Austen sentences predicted as Melville
# ma: Melville sentences predicted as Austen
aa = []
mm = []
am = []
ma = []

for sent, auth in devtest_sents:
    guess = whosaid.predict(count_vect.transform([sent]))[0]
    if auth == "austen" and guess == "austen":
        aa.append((auth, guess, sent))
    elif auth == "melville" and guess == "melville":
        mm.append((auth, guess, sent))
    elif auth == "austen" and guess == "melville":
        am.append((auth, guess, sent))
    elif auth == "melville" and guess == "austen":
        ma.append((auth, guess, sent))

print("done.\n")


print("9. Sample correct and incorrect predictions from dev-test set:")

# Print a random example from each group to illustrate classifier performance
for group_name, group in (("AA (Austen→Austen)", aa),
                          ("MM (Melville→Melville)", mm),
                          ("AM (Austen→Melville)", am),
                          ("MA (Melville→Austen)", ma)):
    if group:
        auth, guess, sent = random.choice(group)
        print(f" - {group_name}")
        print(f"   Author: {auth:10} Prediction: {guess}")
        print(f"   {sent}\n")
print()


print("10. Looking up 40 most informative features:\n")

# Display the top 40 most informative word features for each author
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

# Call the feature display function using the trained classifier and vectorizer
show_most_informative_features(whosaid, count_vect)


#
# Part 2
#


# EDIT to include any code you need for answering the questions in Part 2.
# print("Removing character names and re-evaluating:")

# from sklearn.feature_extraction.text import CountVectorizer

# # Use the provided list of main character names as stop words
# count_vect = CountVectorizer(stop_words=main_characters)

# X_train = count_vect.fit_transform([s for s, _ in train_sents])
# y_train = [a for _, a in train_sents]

# X_devtest = count_vect.transform([s for s, _ in devtest_sents])
# y_devtest = [a for _, a in devtest_sents]

# X_test = count_vect.transform([s for s, _ in test_sents])
# y_test = [a for _, a in test_sents]

print("11. Trying out sentences not from Emma or Moby Dick:\n")

sent1 = "Anne was to leave them on the morrow, an event which they all dreaded."
sent2 = "So Alice began telling them her adventures from the time when she first saw the White Rabbit."

# Transform them using the same CountVectorizer
X_new = count_vect.transform([sent1, sent2])

# Predict with your trained classifier
preds = whosaid.predict(X_new)

print(f"Sent1 prediction: {preds[0]}  |  {sent1}")
print(f"Sent2 prediction: {preds[1]}  |  {sent2}")

print()

print("12. Label probabilities for Sent1 and Sent2:\n")

for i, sent in enumerate([sent1, sent2], start=1):
    feats = count_vect.transform([sent])
    probs = whosaid.predict_proba(feats)[0]
    print(f"Sent{i}: {sent}")
    for label, prob in zip(whosaid.classes_, probs):
        print(f"   P({label} | Sent{i}) = {prob:.4f}")
    print()