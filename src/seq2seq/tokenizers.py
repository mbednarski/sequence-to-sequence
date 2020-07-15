import torch
import itertools
from collections import Counter


class NumberTokenizer:
    def __init__(self):
        self.idx2tok = ["<pad>", "<s>", "</s>"] + list("0123456789") + list("+-*/")
        self.tok2idx = {t: i for i, t in enumerate(self.idx2tok)}

    def encode(self, numbers, add_special_tokens=False):
        encoded = [self.tok2idx[x] for x in numbers if x != " "]

        if add_special_tokens:
            encoded = [self.tok2idx["<s>"]] + encoded + [self.tok2idx["</s>"]]

        return torch.LongTensor(encoded)

    def decode(self, encoded: torch.LongTensor):
        decoded = [self.idx2tok[x] for x in encoded]
        return decoded

    def decode_clean(self, encoded: torch.LongTensor):
        assert len(encoded.shape) == 1
        decoded = [self.idx2tok[x] for x in encoded]
        decoded_filtered = []
        for d in decoded:
            decoded_filtered.append(d)
            if d == "</s>":
                break
        return "".join(decoded_filtered)


def tokenize(text):
    tokens = []
    split_by_space = text.split(" ")

    for s in split_by_space:
        if "-" in s:
            ss = s.split("-")
            tokens.append(ss[0])
            tokens.append("-")
            tokens.append(ss[1])
        else:
            tokens.append(s)

    return tokens


class TextTokenizer:
    def __init__(self, texts_to_fit):
        self.idx2tok = ["<pad>", "<start>", "<eos>"]

        tokenized = [tokenize(x) for x in texts_to_fit]
        words = Counter(itertools.chain.from_iterable(tokenized))
        self.idx2tok += list(words.keys())
        self.tok2idx = {t: i for i, t in enumerate(self.idx2tok)}

    def encode(self, numbers, add_special_tokens=False):
        numbers = tokenize(numbers)
        encoded = [self.tok2idx[x] for x in numbers if x != " "]

        if add_special_tokens:
            encoded = encoded + [self.tok2idx["<eos>"]]

        return torch.LongTensor(encoded)

    def decode(self, encoded: torch.LongTensor):
        decoded = [self.idx2tok[x] for x in encoded]
        return decoded

    def decode_clean(self, encoded: torch.LongTensor):
        assert len(encoded.shape) == 1
        decoded = [self.idx2tok[x] for x in encoded]
        decoded_filtered = []
        for d in decoded:
            decoded_filtered.append(d)
            if d == "<eos>":
                break
        return " ".join(decoded_filtered)


if __name__ == "__main__":
    tok = NumberTokenizer()
    enc = tok.encode("9 + 711 - 752 / 17", add_special_tokens=True)
    print(enc)
    dec = tok.decode(enc)
    print(dec)

    import pandas as pd

    df = pd.read_csv("data/raw/math.csv")

    tt = TextTokenizer(df["text"])

    t = df["text"].iloc[0]
    print(t)
    enc = tt.encode(t, add_special_tokens=True)
    print(enc)
    print(tt.decode(enc))
