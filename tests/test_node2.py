"""
Tests for Node 2 — PyTorch Architect.

Covers:
  (a) Correct PyTorch generation from a valid contract
  (b) Self-correction triggering on invalid contract (d_model % n_heads != 0)
  (c) Correct handling of 3-retry exhaustion
"""

import json
import tempfile
from pathlib import Path

import pytest

from nodes.node2_pytorch_architect import (
    DimensionError,
    check_dimensions,
    parse_generated_files,
    run_node2,
    validate_contract,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_BLUEPRINT = {
    "model_type": "transformer",
    "architecture": {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048,
        "vocab_size": 37000,
        "max_seq_len": 512,
    },
    "objective": "cross_entropy_with_label_smoothing",
    "key_operations": [
        "scaled_dot_product_attention",
        "layer_norm",
        "positional_encoding",
    ],
    "math_notes": "Q,K,V dims: (batch, seq, d_model/n_heads) — must divide evenly",
}

INVALID_BLUEPRINT = {
    **VALID_BLUEPRINT,
    "architecture": {
        **VALID_BLUEPRINT["architecture"],
        "d_model": 513,  # Not divisible by 8
        "n_heads": 8,
    },
}

# A fake NAT response that contains all four expected files
FAKE_NAT_RESPONSE = """## model.py
```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B, T, C)
        return self.out(out)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=37000, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MultiHeadAttention(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x
```

## dataset.py
```python
import torch
from torch.utils.data import Dataset

class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size=37000, seq_len=512, num_samples=1000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]
```

## train.py
```python
import torch
from model import TransformerModel
from dataset import RandomTokenDataset
from torch.utils.data import DataLoader

model = TransformerModel()
dataset = RandomTokenDataset()
loader = DataLoader(dataset, batch_size=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(10):
    for batch, target in loader:
        out = model(batch)
        loss = criterion(out.view(-1, 37000), target.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## config.yaml
```yaml
model:
  d_model: 512
  n_heads: 8
  n_layers: 6
  d_ff: 2048
  vocab_size: 37000
  max_seq_len: 512
training:
  batch_size: 32
  learning_rate: 0.0001
  epochs: 10
  optimizer: AdamW
  loss: cross_entropy_with_label_smoothing
```
"""


@pytest.fixture
def valid_blueprint_path(tmp_path: Path) -> Path:
    p = tmp_path / "blueprint.json"
    p.write_text(json.dumps(VALID_BLUEPRINT))
    return p


@pytest.fixture
def invalid_blueprint_path(tmp_path: Path) -> Path:
    p = tmp_path / "blueprint_invalid.json"
    p.write_text(json.dumps(INVALID_BLUEPRINT))
    return p


# ---------------------------------------------------------------------------
# (a) Correct generation from a valid contract
# ---------------------------------------------------------------------------

class TestValidGeneration:
    def test_check_dimensions_valid(self):
        errors = check_dimensions(VALID_BLUEPRINT["architecture"])
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_validate_contract_valid(self):
        validate_contract(VALID_BLUEPRINT)  # should not raise

    def test_parse_generated_files(self):
        files = parse_generated_files(FAKE_NAT_RESPONSE)
        assert "model.py" in files
        assert "dataset.py" in files
        assert "train.py" in files
        assert "config.yaml" in files

    def test_run_node2_valid_contract(self, valid_blueprint_path: Path, tmp_path: Path):
        output_dir = tmp_path / "output"

        def fake_nat(prompt: str) -> str:
            return FAKE_NAT_RESPONSE

        result = run_node2(valid_blueprint_path, output_dir, nat_caller=fake_nat)

        assert result == output_dir
        assert (output_dir / "model.py").exists()
        assert (output_dir / "dataset.py").exists()
        assert (output_dir / "train.py").exists()
        assert (output_dir / "config.yaml").exists()
        assert (output_dir / "generation_meta.json").exists()

        meta = json.loads((output_dir / "generation_meta.json").read_text())
        assert meta["attempts"] == 1
        assert "model.py" in meta["files_generated"]

    def test_model_py_contains_nn_module(self, valid_blueprint_path: Path, tmp_path: Path):
        output_dir = tmp_path / "output"

        def fake_nat(prompt: str) -> str:
            return FAKE_NAT_RESPONSE

        run_node2(valid_blueprint_path, output_dir, nat_caller=fake_nat)
        model_code = (output_dir / "model.py").read_text()
        assert "nn.Module" in model_code


# ---------------------------------------------------------------------------
# (b) Self-correction on invalid contract
# ---------------------------------------------------------------------------

class TestSelfCorrection:
    def test_check_dimensions_invalid(self):
        arch = INVALID_BLUEPRINT["architecture"]
        errors = check_dimensions(arch)
        assert len(errors) > 0
        assert any("not divisible" in e for e in errors)

    def test_correction_prompt_includes_errors(self, invalid_blueprint_path: Path, tmp_path: Path):
        """When d_model % n_heads != 0, the first prompt includes a WARNING about
        the dimension issue, and subsequent retries use the correction prompt."""
        output_dir = tmp_path / "output"
        prompts_seen: list[str] = []

        def fake_nat(prompt: str) -> str:
            prompts_seen.append(prompt)
            return FAKE_NAT_RESPONSE

        # d_model=513 fails dimension check on every attempt, so this will
        # exhaust retries — that's expected. We just verify the prompts.
        with pytest.raises(DimensionError):
            run_node2(invalid_blueprint_path, output_dir, nat_caller=fake_nat)

        # Must have been called MAX_RETRIES times
        assert len(prompts_seen) == 3
        # First prompt should contain the WARNING about dimension issues
        assert "WARNING" in prompts_seen[0]
        assert "not divisible" in prompts_seen[0]
        # Subsequent prompts should be correction prompts
        assert "Fix the code" in prompts_seen[1] or "tensor dimension errors" in prompts_seen[1]

    def test_invalid_contract_retries_all_three(self, invalid_blueprint_path: Path, tmp_path: Path):
        """An invalid contract should trigger all 3 retry attempts before failing."""
        output_dir = tmp_path / "output"
        call_count = 0

        def counting_nat(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return FAKE_NAT_RESPONSE

        with pytest.raises(DimensionError, match="3 retries"):
            run_node2(invalid_blueprint_path, output_dir, nat_caller=counting_nat)

        assert call_count == 3


# ---------------------------------------------------------------------------
# (c) 3-retry exhaustion
# ---------------------------------------------------------------------------

class TestRetryExhaustion:
    def test_max_retries_on_missing_files(self, valid_blueprint_path: Path, tmp_path: Path):
        """If NAT consistently fails to produce valid files, raise after 3 retries."""
        output_dir = tmp_path / "output"
        call_count = 0

        def bad_nat(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return "This is not valid output with ## headers"

        with pytest.raises(DimensionError, match="failed after 3 retries"):
            run_node2(valid_blueprint_path, output_dir, nat_caller=bad_nat)

        assert call_count == 3

    def test_max_retries_returns_dimension_errors(self, valid_blueprint_path: Path, tmp_path: Path):
        output_dir = tmp_path / "output"

        def bad_nat(prompt: str) -> str:
            return "no valid output"

        with pytest.raises(DimensionError) as exc_info:
            run_node2(valid_blueprint_path, output_dir, nat_caller=bad_nat)

        assert "3 retries" in str(exc_info.value)
