"""
Node 3 — Hardware Blueprint Generator
Consumes the PyTorch scaffold from Node 2, identifies the top 3 computational
bottlenecks, and generates annotated CUDA C++ stubs using NAT with Nemotron
Super 49B.

IMPORTANT: Output is ANNOTATED STUBS — not executable CUDA kernels.
Every line includes mathematical rationale, memory access patterns, and
expected speedup commentary.

Output files (written to outputs/hardware_blueprint/):
  - bottleneck_analysis.json     : top 3 bottlenecks with reasoning
  - <operation>_stub.cu          : annotated CUDA C++ stub per bottleneck
  - hardware_blueprint_meta.json : generation metadata
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCAFFOLD_DIR = Path(__file__).resolve().parent.parent / "outputs" / "pytorch_scaffold"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "hardware_blueprint"

# Required annotation sections in every CUDA stub
REQUIRED_ANNOTATIONS = [
    "Bottleneck Description",
    "Memory Access Pattern",
    "Expected Speedup",
]


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_scaffold(scaffold_dir: Path | None = None) -> dict[str, str]:
    """Load the PyTorch scaffold files from Node 2 output."""
    src = scaffold_dir or SCAFFOLD_DIR
    files: dict[str, str] = {}
    for f in src.iterdir():
        if f.suffix in (".py", ".yaml", ".yml"):
            files[f.name] = f.read_text()
    if not files:
        raise FileNotFoundError(f"No scaffold files found in {src}")
    return files


# ---------------------------------------------------------------------------
# NAT inference interface
# ---------------------------------------------------------------------------

def _build_bottleneck_prompt(scaffold_files: dict[str, str]) -> str:
    """Build the prompt for bottleneck analysis."""
    code_blocks = "\n\n".join(
        f"### {name}\n```python\n{content}\n```"
        for name, content in scaffold_files.items()
    )
    return f"""You are a senior GPU performance engineer analyzing a PyTorch model for hardware optimization.

Given the following PyTorch scaffold, identify the TOP 3 computational bottlenecks that would benefit most from custom CUDA kernel optimization.

{code_blocks}

For each bottleneck, provide:
1. Operation name (e.g., "scaled_dot_product_attention", "layer_norm", "linear_projection")
2. Why it is a bottleneck (FLOPs, memory bandwidth, or both)
3. Computational complexity (Big-O)
4. Current PyTorch implementation limitation
5. Potential speedup from a custom CUDA kernel (conservative estimate)

Return your analysis as a JSON array of exactly 3 objects with keys:
"operation", "reason", "complexity", "pytorch_limitation", "estimated_speedup"

Return ONLY the JSON array, no other text.
"""


def _build_cuda_stub_prompt(
    bottleneck: dict[str, str],
    scaffold_files: dict[str, str],
) -> str:
    """Build the prompt for generating an annotated CUDA C++ stub."""
    model_code = scaffold_files.get("model.py", "")
    return f"""You are a senior CUDA engineer writing an ANNOTATED hardware blueprint stub.

Target operation: {bottleneck["operation"]}
Bottleneck reason: {bottleneck["reason"]}
Complexity: {bottleneck["complexity"]}
PyTorch limitation: {bottleneck["pytorch_limitation"]}

Reference PyTorch code:
```python
{model_code}
```

Generate a CUDA C++ stub file (.cu) that is HEAVILY ANNOTATED. This is a hardware blueprint — a plan that a CUDA engineer would write BEFORE implementing the full kernel.

REQUIRED SECTIONS (use these exact comment headers):
// === Bottleneck Description ===
// === Memory Access Pattern ===
// === Expected Speedup ===

Requirements:
1. Under "Bottleneck Description": explain WHY this operation is compute/memory bound, include Big-O complexity
2. Under "Memory Access Pattern": describe coalesced vs strided access, shared memory usage, bank conflicts
3. Under "Expected Speedup": quantify the expected improvement with reasoning (e.g., "2-3x from eliminating global memory round-trip")
4. Write the CUDA kernel signature and stub body with inline comments on EVERY significant line explaining the mathematical rationale
5. Include grid/block dimension suggestions with reasoning
6. Include shared memory allocation reasoning

IMPORTANT: This is an ANNOTATED STUB, not an executable kernel. Every line must teach the reader WHY, not just WHAT.

Return ONLY the .cu file content, starting with the includes.
"""


def call_nat_nemotron(prompt: str) -> str:
    """
    Call NAT with Nemotron Super 49B for hardware reasoning.

    Uses the NeMo Agent Toolkit inference endpoint running locally.
    """
    nat_endpoint = os.environ.get("NAT_ENDPOINT", "http://localhost:8000/v1/completions")
    model_name = os.environ.get("NAT_MODEL_NEMOTRON", "nemotron-super-49b")

    try:
        import requests
        response = requests.post(
            nat_endpoint,
            json={
                "model": model_name,
                "prompt": prompt,
                "max_tokens": 4096,
                "temperature": 0.3,
                "stop": ["## END"],
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"]
    except Exception as e:
        logger.error("NAT Nemotron inference failed: %s", e)
        raise RuntimeError(f"NAT inference failed: {e}") from e


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------

def parse_bottlenecks(raw: str) -> list[dict[str, str]]:
    """Parse the JSON array of bottleneck analyses from LLM output."""
    if raw is None:
        raise ValueError("NAT returned None for bottleneck analysis (possible content filter or null response)")
    # Try to extract JSON from the response
    raw = raw.strip()
    # Handle markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if match:
        raw = match.group(1)
    bottlenecks = json.loads(raw)
    if not isinstance(bottlenecks, list) or len(bottlenecks) < 1:
        raise ValueError("Expected a non-empty JSON array of bottlenecks")
    return bottlenecks[:3]


def clean_cuda_stub(raw: str) -> str:
    """Clean up the CUDA stub output, removing markdown fences."""
    if raw is None:
        raise ValueError("NAT returned None for CUDA stub (possible content filter or null response)")
    raw = raw.strip()
    raw = re.sub(r"^```(?:cuda|cpp|c\+\+)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)
    return raw


def validate_stub_annotations(stub_content: str) -> list[str]:
    """
    Verify that the stub contains all required annotation sections.
    Returns list of missing sections.
    """
    missing = []
    for section in REQUIRED_ANNOTATIONS:
        if section not in stub_content:
            missing.append(section)
    return missing


def sanitize_filename(name: str) -> str:
    """Convert an operation name to a safe filename."""
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_node3(
    scaffold_dir: Path | None = None,
    output_dir: Path | None = None,
    nat_caller: Any = None,
) -> Path:
    """
    Execute Node 3: analyze bottlenecks and generate annotated CUDA stubs.

    Args:
        scaffold_dir: Path to Node 2 output (defaults to outputs/pytorch_scaffold/).
        output_dir: Override output directory (defaults to outputs/hardware_blueprint/).
        nat_caller: Optional callable(prompt) -> str to override NAT inference (for testing).

    Returns:
        Path to the output directory containing the hardware blueprint.
    """
    caller = nat_caller or call_nat_nemotron
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Load the PyTorch scaffold
    scaffold_files = load_scaffold(scaffold_dir)
    logger.info("Loaded scaffold: %s", list(scaffold_files.keys()))

    # Step 1: Identify top 3 bottlenecks
    logger.info("Analyzing computational bottlenecks...")
    bottleneck_prompt = _build_bottleneck_prompt(scaffold_files)
    raw_bottlenecks = caller(bottleneck_prompt)
    bottlenecks = parse_bottlenecks(raw_bottlenecks)

    # Write bottleneck analysis
    (out / "bottleneck_analysis.json").write_text(
        json.dumps(bottlenecks, indent=2)
    )
    logger.info("Identified %d bottlenecks", len(bottlenecks))

    # Step 2: Generate annotated CUDA stubs for each bottleneck
    stub_files: list[str] = []
    for i, bn in enumerate(bottlenecks):
        op_name = bn.get("operation", f"bottleneck_{i}")
        filename = f"{sanitize_filename(op_name)}_stub.cu"
        logger.info("Generating CUDA stub for: %s", op_name)

        stub_prompt = _build_cuda_stub_prompt(bn, scaffold_files)
        raw_stub = caller(stub_prompt)
        stub_content = clean_cuda_stub(raw_stub)

        # Validate annotations
        missing = validate_stub_annotations(stub_content)
        if missing:
            logger.warning(
                "Stub for %s missing annotations: %s. "
                "Adding placeholder sections.",
                op_name, missing,
            )
            for section in missing:
                stub_content += f"\n\n// === {section} ===\n// [TO BE COMPLETED]\n"

        (out / filename).write_text(stub_content)
        stub_files.append(filename)
        logger.info("Wrote %s", filename)

    # Write metadata
    meta = {
        "source_scaffold": str(scaffold_dir or SCAFFOLD_DIR),
        "bottleneck_count": len(bottlenecks),
        "stub_files": stub_files,
        "note": "ANNOTATED STUBS — not executable CUDA kernels. "
                "These are hardware blueprints showing optimization rationale.",
    }
    (out / "hardware_blueprint_meta.json").write_text(json.dumps(meta, indent=2))

    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    scaffold = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    result = run_node3(scaffold, out_dir)
    print(f"Hardware blueprint written to: {result}")
