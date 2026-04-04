import torch
from torch.utils.data import Dataset


class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int = 37000, seq_len: int = 512, size: int = 128) -> None:
        self.inputs = torch.randint(0, vocab_size, (size, seq_len))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        sample = self.inputs[index]
        return sample, sample
