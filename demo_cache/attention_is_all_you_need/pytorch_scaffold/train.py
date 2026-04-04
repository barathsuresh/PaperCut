import torch
from torch.utils.data import DataLoader

from dataset import RandomTokenDataset
from model import TransformerModel


def train_one_epoch() -> None:
    model = TransformerModel()
    dataset = RandomTokenDataset()
    loader = DataLoader(dataset, batch_size=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for tokens, targets in loader:
        outputs = model(tokens)
        loss = outputs.mean() + targets.float().mean() * 0.0
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    train_one_epoch()
