import random
from typing import Iterable

from torch.utils.data import Dataset


class ReverseDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        y = x[::-1]

        return x, y

    def collate(self, batch):
        x, y = zip(*batch)

        x = list(x)
        y = list(y)

        return x, y


def generate_reverse_dataset(
    vocabulary: Iterable[str], min_len: int, max_len: int, dset_size: int
) -> ReverseDataset:
    samples = []

    for _ in range(dset_size):
        l = random.randint(min_len, max_len)
        s = random.choices(vocabulary, k=l)
        samples.append("".join(s))

    dset = ReverseDataset(samples)

    return dset
