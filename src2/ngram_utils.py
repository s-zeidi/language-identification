from collections import Counter


def extract_ngrams(text, n_min=1, n_max=3):

    text = text.lower()

    ngrams = []

    for n in range(n_min, n_max + 1):
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i+n])

    return ngrams


def count_ngrams(texts):

    counts = Counter()

    for text in texts:
        ngrams = extract_ngrams(text)
        counts.update(ngrams)

    return counts