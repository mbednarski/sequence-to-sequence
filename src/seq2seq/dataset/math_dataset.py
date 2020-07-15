from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from seq2seq.tokenizers import NumberTokenizer, TextTokenizer

import pandas as pd


class MathDataset(Dataset):
    def __init__(self, fname: str):
        super().__init__()
        self.df_full = pd.read_csv("data/raw/math.csv")
        self.df = pd.read_csv(fname)

        self.number_tokenizer = NumberTokenizer()
        self.text_tokenizer = TextTokenizer(self.df_full["text"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df["numbers"][idx]
        y = self.df["text"][idx]

        x = self.number_tokenizer.encode(x, add_special_tokens=True)
        y = self.text_tokenizer.encode(y, add_special_tokens=True)

        return x, y

    def collate(self, batch):
        x, y = zip(*batch)

        x = list(x)
        y = list(y)

        x = pad_sequence(x, batch_first=True, padding_value=0)
        y = pad_sequence(y, batch_first=True, padding_value=0)

        return x, y


if __name__ == "__main__":
    md = MathDataset()
    print(md[0])
    print(md.number_tokenizer.tok2idx)
    print(md.text_tokenizer.tok2idx)
