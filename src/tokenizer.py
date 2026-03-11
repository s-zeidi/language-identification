import numpy as np


class CharTokenizer:

    def __init__(self, max_length=256):
        self.max_length = max_length

        self.char2id = {}
        self.id2char = {}

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

    def build_vocab(self, texts):
        """
        Build character vocabulary from training texts.
        """

        chars = set()

        for text in texts:
            chars.update(list(text))

        chars = sorted(chars)

        self.char2id = {
            self.pad_token: 0,
            self.unk_token: 1
        }

        for i, c in enumerate(chars, start=2):
            self.char2id[c] = i

        self.id2char = {v: k for k, v in self.char2id.items()}

        print("Vocabulary size:", len(self.char2id))

    def encode(self, text):
        """
        Convert text into list of character IDs.
        """

        ids = []

        for c in text:
            if c in self.char2id:
                ids.append(self.char2id[c])
            else:
                ids.append(self.char2id[self.unk_token])

        # padding / truncation
        if len(ids) < self.max_length:
            ids += [0] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]

        return np.array(ids)

    def decode(self, ids):
        """
        Convert IDs back to text.
        """

        chars = []

        for i in ids:
            if i == 0:
                continue
            chars.append(self.id2char.get(i, ""))

        return "".join(chars)