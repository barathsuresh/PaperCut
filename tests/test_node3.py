"""
Tests for Node 3 — Hardware Blueprint Generator.

Covers:
  (a) Output contains at least one CUDA stub file
  (b) Each stub contains the required annotation sections:
      - Bottleneck Description
      - Memory Access Pattern
      - Expected Speedup
"""

import json
from pathlib import Path

import pytest

from nodes.node3_hardware_blueprint import (
    REQUIRED_ANNOTATIONS,
    clean_cuda_stub,
    parse_bottlenecks,
    run_node3,
    sanitize_filename,
    validate_stub_annotations,
)

# ---------------------------------------------------------------------------
# Fake NAT responses
# ---------------------------------------------------------------------------

FAKE_BOTTLENECK_RESPONSE = json.dumps([
    {
        "operation": "scaled_dot_product_attention",
        "reason": "O(n^2 * d) compute with large memory footprint for attention matrix",
        "complexity": "O(n^2 * d)",
        "pytorch_limitation": "Separate matmul and softmax operations cause extra global memory round-trips",
        "estimated_speedup": "2-3x with fused kernel",
    },
    {
        "operation": "layer_norm",
        "reason": "Memory-bound operation requiring two passes (mean then variance)",
        "complexity": "O(n * d)",
        "pytorch_limitation": "Default PyTorch impl uses two separate reduction passes",
        "estimated_speedup": "1.5-2x with single-pass fused kernel",
    },
    {
        "operation": "linear_projection",
        "reason": "Large GEMM operations dominate total FLOPs",
        "complexity": "O(n * d^2)",
        "pytorch_limitation": "Generic cuBLAS call may not optimize for specific shapes",
        "estimated_speedup": "1.2-1.5x with shape-specialized kernel",
    },
])

FAKE_CUDA_STUB = """```cuda
#include <cuda_runtime.h>
#include <math.h>

// === Bottleneck Description ===
// Scaled dot-product attention is the dominant compute bottleneck in transformer
// architectures. For sequence length n and head dimension d, the attention matrix
// computation requires O(n^2 * d) FLOPs. The softmax normalization adds an
// additional O(n^2) pass. In PyTorch, these are separate kernel launches:
// matmul(Q, K^T) -> softmax -> matmul(attn, V), each requiring a global memory
// round-trip for the intermediate n×n attention matrix.

// === Memory Access Pattern ===
// Without fusion: Q,K are read from global memory -> attention matrix written to
// global memory -> read back for softmax -> written again -> read for V matmul.
// Total global memory traffic: 3 * n^2 * sizeof(float) bytes for intermediates.
// With fusion: Q,K tiles loaded into shared memory -> attention computed in
// registers -> softmax applied in-register -> V multiplication in same pass.
// Shared memory per block: tile_size^2 * sizeof(float) bytes.
// Coalesced access: threads in a warp read consecutive elements of K^T columns.

// === Expected Speedup ===
// Eliminating 2 global memory round-trips for the n×n attention matrix saves
// ~2 * n^2 * sizeof(float) bytes of bandwidth per head per layer.
// For n=512, d=64: this is 2 * 512^2 * 4 = 2MB per head.
// With 8 heads and 6 layers: 96MB total saved per forward pass.
// Conservative expected speedup: 2-3x for attention computation.

// Kernel signature: fused scaled dot-product attention
// Grid: (batch_size * n_heads, ceil(seq_len / TILE_M), 1)
// Block: (TILE_N, TILE_M, 1) — each block computes one tile of the output
__global__ void fused_scaled_dot_product_attention(
    const float* __restrict__ Q,    // [batch, heads, seq_len, head_dim]
    const float* __restrict__ K,    // [batch, heads, seq_len, head_dim]
    const float* __restrict__ V,    // [batch, heads, seq_len, head_dim]
    float* __restrict__ output,     // [batch, heads, seq_len, head_dim]
    const int seq_len,
    const int head_dim,
    const float scale               // 1/sqrt(head_dim)
) {
    // Shared memory for Q and K tiles — avoids repeated global memory reads
    // Size: 2 * TILE_SIZE * head_dim * sizeof(float)
    extern __shared__ float smem[];

    // Thread index maps to output position
    // Each thread computes one element of the output for one query position
    const int query_pos = blockIdx.y * blockDim.y + threadIdx.y;
    const int head_idx = blockIdx.x;

    // STUB: tile-based attention computation would go here
    // Step 1: Load Q[query_pos] tile into shared memory (coalesced read)
    // Step 2: Iterate over K tiles, compute partial dot products in registers
    // Step 3: Apply online softmax (numerically stable, single pass)
    // Step 4: Accumulate weighted V in registers
    // Step 5: Write final output to global memory (coalesced write)
}
```"""


@pytest.fixture
def scaffold_dir(tmp_path: Path) -> Path:
    """Create a minimal scaffold directory with a model.py file."""
    d = tmp_path / "scaffold"
    d.mkdir()
    (d / "model.py").write_text(
        "import torch.nn as nn\n"
        "class MultiHeadAttention(nn.Module):\n"
        "    def __init__(self, d_model=512, n_heads=8):\n"
        "        super().__init__()\n"
        "        self.head_dim = d_model // n_heads\n"
    )
    (d / "config.yaml").write_text("d_model: 512\nn_heads: 8\n")
    return d


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_parse_bottlenecks(self):
        bottlenecks = parse_bottlenecks(FAKE_BOTTLENECK_RESPONSE)
        assert len(bottlenecks) == 3
        assert bottlenecks[0]["operation"] == "scaled_dot_product_attention"

    def test_parse_bottlenecks_with_code_fence(self):
        fenced = f"```json\n{FAKE_BOTTLENECK_RESPONSE}\n```"
        bottlenecks = parse_bottlenecks(fenced)
        assert len(bottlenecks) == 3

    def test_clean_cuda_stub(self):
        cleaned = clean_cuda_stub(FAKE_CUDA_STUB)
        assert not cleaned.startswith("```")
        assert not cleaned.endswith("```")
        assert "#include" in cleaned

    def test_sanitize_filename(self):
        assert sanitize_filename("scaled_dot_product_attention") == "scaled_dot_product_attention"
        assert sanitize_filename("Layer Norm!") == "layer_norm_"

    def test_validate_stub_annotations_pass(self):
        cleaned = clean_cuda_stub(FAKE_CUDA_STUB)
        missing = validate_stub_annotations(cleaned)
        assert missing == [], f"Missing sections: {missing}"

    def test_validate_stub_annotations_fail(self):
        missing = validate_stub_annotations("// just a comment\nint x = 0;")
        assert len(missing) == len(REQUIRED_ANNOTATIONS)


# ---------------------------------------------------------------------------
# Integration test: run_node3
# ---------------------------------------------------------------------------

class TestNode3Pipeline:
    def test_produces_cuda_stubs(self, scaffold_dir: Path, tmp_path: Path):
        """Node 3 output must contain at least one .cu stub file."""
        output_dir = tmp_path / "hw_blueprint"
        call_count = 0

        def fake_nat(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            # First call is bottleneck analysis, subsequent are CUDA stubs
            if call_count == 1:
                return FAKE_BOTTLENECK_RESPONSE
            return FAKE_CUDA_STUB

        result = run_node3(scaffold_dir, output_dir, nat_caller=fake_nat)

        # Must have at least one .cu file
        cu_files = list(result.glob("*.cu"))
        assert len(cu_files) >= 1, f"Expected at least 1 .cu stub, found {len(cu_files)}"

    def test_all_stubs_have_required_annotations(self, scaffold_dir: Path, tmp_path: Path):
        """Every generated .cu stub must contain all required annotation sections."""
        output_dir = tmp_path / "hw_blueprint"
        call_count = 0

        def fake_nat(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FAKE_BOTTLENECK_RESPONSE
            return FAKE_CUDA_STUB

        result = run_node3(scaffold_dir, output_dir, nat_caller=fake_nat)

        cu_files = list(result.glob("*.cu"))
        assert len(cu_files) >= 1

        for cu_file in cu_files:
            content = cu_file.read_text()
            missing = validate_stub_annotations(content)
            assert missing == [], (
                f"{cu_file.name} missing required sections: {missing}"
            )

    def test_bottleneck_analysis_written(self, scaffold_dir: Path, tmp_path: Path):
        """The bottleneck analysis JSON should be written."""
        output_dir = tmp_path / "hw_blueprint"
        call_count = 0

        def fake_nat(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FAKE_BOTTLENECK_RESPONSE
            return FAKE_CUDA_STUB

        run_node3(scaffold_dir, output_dir, nat_caller=fake_nat)

        analysis_path = output_dir / "bottleneck_analysis.json"
        assert analysis_path.exists()
        bottlenecks = json.loads(analysis_path.read_text())
        assert len(bottlenecks) == 3

    def test_metadata_written(self, scaffold_dir: Path, tmp_path: Path):
        """Generation metadata file must exist and describe output as annotated stubs."""
        output_dir = tmp_path / "hw_blueprint"
        call_count = 0

        def fake_nat(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FAKE_BOTTLENECK_RESPONSE
            return FAKE_CUDA_STUB

        run_node3(scaffold_dir, output_dir, nat_caller=fake_nat)

        meta_path = output_dir / "hardware_blueprint_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        # Must never describe output as executable
        assert "not executable" in meta["note"].lower() or "annotated stubs" in meta["note"].lower()
        assert len(meta["stub_files"]) >= 1

    def test_stub_placeholder_added_for_missing_annotations(self, scaffold_dir: Path, tmp_path: Path):
        """If a stub is missing annotations, placeholders should be injected."""
        output_dir = tmp_path / "hw_blueprint"
        call_count = 0

        incomplete_stub = "// A minimal CUDA stub\n__global__ void kernel() {}\n"

        def fake_nat(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FAKE_BOTTLENECK_RESPONSE
            return incomplete_stub

        run_node3(scaffold_dir, output_dir, nat_caller=fake_nat)

        # Even with incomplete NAT output, all stubs should end up with all sections
        for cu_file in output_dir.glob("*.cu"):
            content = cu_file.read_text()
            for section in REQUIRED_ANNOTATIONS:
                assert section in content, (
                    f"{cu_file.name} should have placeholder for '{section}'"
                )


# ---------------------------------------------------------------------------
# Integration tests — hit real NVIDIA API
# ---------------------------------------------------------------------------

SCAFFOLD_CACHE = (
    Path(__file__).resolve().parent.parent
    / "demo_cache" / "attention_is_all_you_need" / "pytorch_scaffold"
)


@pytest.mark.integration
class TestNode3Integration:
    """Requires network access and a valid NVIDIA_API_KEY in .env."""

    def test_end_to_end_with_real_api(self, tmp_path: Path):
        from nat.nat_client import make_nat_caller_reason

        output_dir = tmp_path / "hw_blueprint"
        caller = make_nat_caller_reason()

        # Use demo_cache scaffold as input (known-good Node 2 output)
        result = run_node3(SCAFFOLD_CACHE, output_dir, nat_caller=caller)

        assert result == output_dir

        # Must produce at least one .cu stub
        cu_files = list(output_dir.glob("*.cu"))
        assert len(cu_files) >= 1, f"Expected >=1 .cu stub, found {len(cu_files)}"

        # Every stub must have all required annotation sections
        for cu_file in cu_files:
            content = cu_file.read_text()
            missing = validate_stub_annotations(content)
            assert missing == [], (
                f"{cu_file.name} missing: {missing}"
            )

        # Bottleneck analysis JSON should exist with at least 1 entry
        analysis = json.loads((output_dir / "bottleneck_analysis.json").read_text())
        assert len(analysis) >= 1

        # Metadata must describe output as annotated stubs
        meta = json.loads((output_dir / "hardware_blueprint_meta.json").read_text())
        assert "not executable" in meta["note"].lower() or "annotated stubs" in meta["note"].lower()
