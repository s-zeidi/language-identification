from collections import Counter


class NgramTokenizer:

    def __init__(self, n_min=1, n_max=3, max_length=512):
        self.n_min = n_min
        self.n_max = n_max
        self.max_length = max_length

        self.ngram2id = {}
        self.id2ngram = {}

    def extract_ngrams(self, text):

        text = text.lower()

        ngrams = []

        for n in range(self.n_min, self.n_max + 1):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i+n])

        return ngrams

    def build_vocab(self, texts):

        counter = Counter()

        for text in texts:
            ngrams = self.extract_ngrams(text)
            counter.update(ngrams)

        vocab = list(counter.keys())

        self.ngram2id = {ng:i+1 for i,ng in enumerate(vocab)}
        self.id2ngram = {i+1:ng for i,ng in enumerate(vocab)}

    def encode(self, text):

        ngrams = self.extract_ngrams(text)

        ids = [self.ngram2id.get(ng,0) for ng in ngrams]

        ids = ids[:self.max_length]

        if len(ids) < self.max_length:
            ids += [0]*(self.max_length-len(ids))

        return ids