import itertools
from collections import Counter

import torch


class NumberTokenizer:
    def __init__(self):
        self.idx2tok = ["<pad>", "<s>", "</s>"] + list("0123456789") + list("+-*/")
        self.tok2idx = {t: i for i, t in enumerate(self.idx2tok)}

    def get_padding_idx(self):
        return self.tok2idx["<pad>"]

    def get_vocab_size(self):
        return len(self.tok2idx)

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


class TextTokenizer:
    def _tokenize(self, text):
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

    def __init__(self, texts_to_fit):
        self.idx2tok = ["<pad>", "<start>", "<eos>"]

        tokenized = [self._tokenize(x) for x in texts_to_fit]
        words = Counter(itertools.chain.from_iterable(tokenized))
        self.idx2tok += list(words.keys())
        self.tok2idx = {t: i for i, t in enumerate(self.idx2tok)}

    def get_start_idx(self):
        return self.tok2idx["<start>"]

    def get_vocab_size(self):
        return len(self.tok2idx)

    def get_pad_idx(self):
        return self.tok2idx["<pad>"]

    def encode(self, numbers, add_special_tokens=False):
        numbers = self._tokenize(numbers)
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
