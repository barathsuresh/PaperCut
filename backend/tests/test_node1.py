import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.nodes.node1_ingestor import ScopeRejectedError, run_node1
from backend.schemas.contract import ArchitectureBlueprint

SAMPLE_BLUEPRINT = {
    "model_type": "transformer",
    "architecture": {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048,
        "vocab_size": 37000,
        "max_seq_len": 512,
    },
    "objective": "cross-entropy language modeling",
    "key_operations": ["scaled dot-product attention", "multi-head attention", "position-wise FFN"],
    "math_notes": "d_model must be divisible by n_heads (512/8=64 per head). d_ff=4*d_model.",
}


@patch("backend.nodes.node1_ingestor.pdf_part_from_gcs")
@patch("backend.nodes.node1_ingestor.get_pro_model")
@patch("backend.nodes.node1_ingestor.run_node0", new_callable=AsyncMock)
async def test_node1_extracts_blueprint(mock_node0, mock_get_model, mock_pdf_part):
    mock_node0.return_value = MagicMock(result="PASS", reason="ML paper")
    mock_response = MagicMock()
    mock_response.text = json.dumps(SAMPLE_BLUEPRINT)
    mock_get_model.return_value.generate_content_async = AsyncMock(return_value=mock_response)
    mock_pdf_part.return_value = MagicMock()

    result = await run_node1("gs://test-bucket/papers/abc/paper.pdf")

    assert isinstance(result, ArchitectureBlueprint)
    assert result.model_type == "transformer"
    assert result.architecture.d_model == 512
    assert result.architecture.n_heads == 8
    assert len(result.key_operations) >= 1


@patch("backend.nodes.node1_ingestor.pdf_part_from_gcs")
@patch("backend.nodes.node1_ingestor.get_pro_model")
@patch("backend.nodes.node1_ingestor.run_node0", new_callable=AsyncMock)
async def test_node1_strips_markdown_fences(mock_node0, mock_get_model, mock_pdf_part):
    mock_node0.return_value = MagicMock(result="PASS", reason="ML paper")
    mock_response = MagicMock()
    mock_response.text = f"```json\n{json.dumps(SAMPLE_BLUEPRINT)}\n```"
    mock_get_model.return_value.generate_content_async = AsyncMock(return_value=mock_response)
    mock_pdf_part.return_value = MagicMock()

    result = await run_node1("gs://test-bucket/papers/abc/paper.pdf")

    assert result.architecture.n_layers == 6


@patch("backend.nodes.node1_ingestor.run_node0", new_callable=AsyncMock)
async def test_node1_raises_on_scope_fail(mock_node0):
    mock_node0.return_value = MagicMock(result="FAIL", reason="Not an ML paper.")

    with pytest.raises(ScopeRejectedError) as exc_info:
        await run_node1("gs://test-bucket/papers/abc/paper.pdf")

    assert "Not an ML paper." in str(exc_info.value)
