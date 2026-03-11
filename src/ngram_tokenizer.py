class NgramTokenizer:

    def __init__(self, n=3, max_length=512):

        self.n = n
        self.max_length = max_length
        self.ngram2id = {"<pad>": 0}

    def extract_ngrams(self, text):

        text = text.lower()

        ngrams = []

        for i in range(len(text) - self.n + 1):
            ngrams.append(text[i:i+self.n])

        return ngrams

    def build_vocab(self, texts):

        idx = 1

        for text in texts:

            ngrams = self.extract_ngrams(text)

            for ng in ngrams:

                if ng not in self.ngram2id:
                    self.ngram2id[ng] = idx
                    idx += 1

    def encode(self, text):

        ngrams = self.extract_ngrams(text)

        ids = [self.ngram2id.get(ng, 0) for ng in ngrams]

        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        else:
            ids += [0] * (self.max_length - len(ids))

        return ids