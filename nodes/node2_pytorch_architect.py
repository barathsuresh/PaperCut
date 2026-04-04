"""
Node 2 — PyTorch Architect
Consumes the architecture blueprint JSON from Node 1 and generates a validated
PyTorch project scaffold using NAT with Qwen2.5-Coder 32B.

Output files (written to outputs/pytorch_scaffold/):
  - model.py      : nn.Module class definitions
  - dataset.py    : data loading utilities
  - train.py      : training loop
  - config.yaml   : hyperparameter configuration
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import jsonschema
import yaml

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "contracts" / "architecture_blueprint_schema.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "pytorch_scaffold"
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------

def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def validate_contract(blueprint: dict) -> None:
    """Validate a blueprint dict against the JSON schema."""
    schema = load_schema()
    jsonschema.validate(instance=blueprint, schema=schema)


# ---------------------------------------------------------------------------
# Tensor-dimension self-correction
# ---------------------------------------------------------------------------

class DimensionError(Exception):
    """Raised when tensor dimension invariants are violated."""


def check_dimensions(arch: dict[str, Any]) -> list[str]:
    """
    Dry-run tensor dimension math. Returns a list of error messages.
    Empty list means all checks pass.
    """
    errors: list[str] = []
    d_model = arch["d_model"]
    n_heads = arch["n_heads"]
    d_ff = arch["d_ff"]
    vocab_size = arch["vocab_size"]
    max_seq_len = arch["max_seq_len"]

    # 1. d_model must be divisible by n_heads
    if d_model % n_heads != 0:
        errors.append(
            f"d_model ({d_model}) is not divisible by n_heads ({n_heads}). "
            f"Head dimension would be {d_model / n_heads:.2f} (must be integer)."
        )

    # 2. d_ff should be > d_model (typically 4x)
    if d_ff <= d_model:
        errors.append(
            f"d_ff ({d_ff}) should be greater than d_model ({d_model})."
        )

    # 3. Embedding dimensions: vocab_size and max_seq_len must be positive
    if vocab_size < 1:
        errors.append(f"vocab_size ({vocab_size}) must be >= 1.")
    if max_seq_len < 1:
        errors.append(f"max_seq_len ({max_seq_len}) must be >= 1.")

    # 4. d_model must be positive
    if d_model < 1:
        errors.append(f"d_model ({d_model}) must be >= 1.")

    return errors


# ---------------------------------------------------------------------------
# NAT inference interface
# ---------------------------------------------------------------------------

def _build_generation_prompt(blueprint: dict) -> str:
    """Build the prompt sent to Qwen2.5-Coder 32B via NAT."""
    arch = blueprint["architecture"]
    ops = ", ".join(blueprint["key_operations"])
    return f"""You are an expert PyTorch engineer. Generate a complete PyTorch project scaffold for the following architecture.

Model type: {blueprint["model_type"]}
Architecture parameters:
  d_model      = {arch["d_model"]}
  n_heads      = {arch["n_heads"]}
  n_layers     = {arch["n_layers"]}
  d_ff         = {arch["d_ff"]}
  vocab_size   = {arch["vocab_size"]}
  max_seq_len  = {arch["max_seq_len"]}
Objective: {blueprint["objective"]}
Key operations: {ops}
Math notes: {blueprint["math_notes"]}

Generate EXACTLY four files. Separate each file with a markdown header:
## model.py
## dataset.py
## train.py
## config.yaml

Requirements:
- model.py: Define all nn.Module classes (e.g., MultiHeadAttention, PositionalEncoding, TransformerEncoderLayer, TransformerModel). Use d_model, n_heads, d_ff from the parameters above. Ensure head_dim = d_model // n_heads.
- dataset.py: Provide a simple Dataset class that generates random token sequences for testing.
- train.py: Standard training loop with AdamW optimizer and the specified loss objective.
- config.yaml: All hyperparameters in YAML format.
"""


def _build_correction_prompt(blueprint: dict, errors: list[str], previous_code: str) -> str:
    """Build a correction prompt when dimension validation fails."""
    error_block = "\n".join(f"  - {e}" for e in errors)
    return f"""The previously generated PyTorch code has tensor dimension errors:
{error_block}

Original architecture parameters: {json.dumps(blueprint["architecture"], indent=2)}

Fix the code so that:
1. head_dim = d_model // n_heads is an integer
2. All linear layer dimensions are consistent
3. Embedding dimensions match d_model

Previously generated code (FIX THIS):
{previous_code}

Generate the corrected four files using the same ## headers format.
"""


def call_nat_qwen(prompt: str) -> str:
    """
    Call NAT with Qwen2.5-Coder 32B for code generation.

    Uses the NeMo Agent Toolkit inference endpoint running locally.
    Falls back to returning an empty string if NAT is unavailable.
    """
    nat_endpoint = os.environ.get("NAT_ENDPOINT", "http://localhost:8000/v1/completions")
    model_name = os.environ.get("NAT_MODEL_QWEN", "qwen2.5-coder-32b")

    try:
        import requests
        response = requests.post(
            nat_endpoint,
            json={
                "model": model_name,
                "prompt": prompt,
                "max_tokens": 8192,
                "temperature": 0.2,
                "stop": ["## END"],
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"]
    except Exception as e:
        logger.error("NAT Qwen inference failed: %s", e)
        raise RuntimeError(f"NAT inference failed: {e}") from e


# ---------------------------------------------------------------------------
# Code extraction and file writing
# ---------------------------------------------------------------------------

EXPECTED_FILES = ("model.py", "dataset.py", "train.py", "config.yaml")


def parse_generated_files(raw_output: str) -> dict[str, str]:
    """
    Parse the LLM output into a dict mapping filename → content.
    Expects markdown ## headers separating each file.
    """
    files: dict[str, str] = {}
    # Split on ## headers
    parts = re.split(r"^##\s+", raw_output, flags=re.MULTILINE)
    for part in parts:
        if not part.strip():
            continue
        lines = part.strip().split("\n", 1)
        filename = lines[0].strip().rstrip(":")
        # Normalize filename
        for expected in EXPECTED_FILES:
            if expected in filename.lower():
                content = lines[1].strip() if len(lines) > 1 else ""
                # Strip markdown code fences if present
                content = re.sub(r"^```\w*\n?", "", content)
                content = re.sub(r"\n?```\s*$", "", content)
                files[expected] = content
                break
    return files


def write_scaffold(files: dict[str, str], output_dir: Path | None = None) -> Path:
    """Write the generated files to the output directory."""
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (out / filename).write_text(content)
        logger.info("Wrote %s", out / filename)
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_node2(
    blueprint_path: str | Path,
    output_dir: Path | None = None,
    nat_caller: Any = None,
) -> Path:
    """
    Execute Node 2: load blueprint, generate PyTorch scaffold, validate, self-correct.

    Args:
        blueprint_path: Path to the JSON blueprint file from Node 1.
        output_dir: Override output directory (defaults to outputs/pytorch_scaffold/).
        nat_caller: Optional callable(prompt) -> str to override NAT inference (for testing).

    Returns:
        Path to the output directory containing the scaffold files.

    Raises:
        DimensionError: If dimension validation fails after MAX_RETRIES attempts.
        jsonschema.ValidationError: If the input blueprint is invalid.
    """
    # Load and validate the input contract
    blueprint_path = Path(blueprint_path)
    with open(blueprint_path) as f:
        blueprint = json.load(f)
    validate_contract(blueprint)

    arch = blueprint["architecture"]
    caller = nat_caller or call_nat_qwen

    # Check dimensions before generation — fail fast if the contract itself is bad
    pre_errors = check_dimensions(arch)

    raw_output = ""
    generated_files: dict[str, str] = {}

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("Node 2 generation attempt %d/%d", attempt, MAX_RETRIES)

        if attempt == 1 and not pre_errors:
            prompt = _build_generation_prompt(blueprint)
        elif attempt == 1 and pre_errors:
            # Contract has bad dimensions — still try generation but include warnings
            prompt = _build_generation_prompt(blueprint)
            prompt += f"\n\nWARNING: The input architecture has issues:\n"
            prompt += "\n".join(f"  - {e}" for e in pre_errors)
            prompt += "\nAdjust the implementation to handle these or flag them.\n"
        else:
            prompt = _build_correction_prompt(blueprint, pre_errors, raw_output)

        raw_output = caller(prompt)
        generated_files = parse_generated_files(raw_output)

        # Check we got all expected files
        missing = [f for f in EXPECTED_FILES if f not in generated_files]
        if missing:
            pre_errors = [f"Missing generated files: {missing}"]
            logger.warning("Attempt %d: missing files %s", attempt, missing)
            continue

        # Re-check dimensions in the generated config if present
        post_errors = check_dimensions(arch)
        if not post_errors:
            logger.info("Dimension validation passed on attempt %d", attempt)
            break

        pre_errors = post_errors
        logger.warning("Attempt %d failed dimension check: %s", attempt, post_errors)
    else:
        raise DimensionError(
            f"Dimension validation failed after {MAX_RETRIES} retries. "
            f"Last errors: {pre_errors}"
        )

    out = write_scaffold(generated_files, output_dir)

    # Write a generation metadata file
    meta = {
        "source_blueprint": str(blueprint_path),
        "attempts": attempt,
        "dimension_errors_on_input": check_dimensions(arch),
        "files_generated": list(generated_files.keys()),
    }
    (out / "generation_meta.json").write_text(json.dumps(meta, indent=2))

    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <blueprint.json> [output_dir]")
        sys.exit(1)

    bp_path = sys.argv[1]
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    result = run_node2(bp_path, out_dir)
    print(f"Scaffold written to: {result}")
