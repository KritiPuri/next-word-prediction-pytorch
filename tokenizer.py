import re
from collections import Counter

# Similar to the original Keras Tokenizer's default punctuation filtering.
FILTERS = r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n"""

def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans({ch: " " for ch in FILTERS}))
    return text.split()

class WordTokenizer:
    def __init__(self):
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts):
        counts = Counter()
        for text in texts:
            counts.update(tokenize(text))
        # Stable ordering: frequency descending, first-seen order for ties.
        self.word_index = {word: i + 1 for i, (word, _) in enumerate(counts.most_common())}
        self.index_word = {idx: word for word, idx in self.word_index.items()}

    def texts_to_sequences(self, texts):
        return [[self.word_index[w] for w in tokenize(text) if w in self.word_index] for text in texts]

    def to_dict(self):
        return {"word_index": self.word_index}

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.word_index = {str(w): int(i) for w, i in data["word_index"].items()}
        obj.index_word = {i: w for w, i in obj.word_index.items()}
        return obj
