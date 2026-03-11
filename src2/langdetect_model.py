import math
from collections import defaultdict

from src2.ngram_utils import extract_ngrams, count_ngrams


class LangDetectModel:

    def __init__(self):

        self.language_profiles = {}
        self.languages = []


    def train(self, texts, labels):

        lang_texts = defaultdict(list)

        for text, label in zip(texts, labels):
            lang_texts[label].append(text)

        for lang, lang_data in lang_texts.items():

            counts = count_ngrams(lang_data)

            total = sum(counts.values())

            profile = {k: v / total for k, v in counts.items()}

            self.language_profiles[lang] = profile

        self.languages = list(self.language_profiles.keys())


    def predict(self, text):

        ngrams = extract_ngrams(text)

        scores = {}

        for lang in self.languages:

            profile = self.language_profiles[lang]

            score = 0

            for ng in ngrams:

                prob = profile.get(ng, 1e-7)

                score += math.log(prob)

            scores[lang] = score

        best_lang = max(scores, key=scores.get)

        return best_lang