from unittest.mock import MagicMock, patch

from backend.nodes.node0_validator import run_node0
from backend.schemas.validator import ScopeValidationResult


@patch("backend.nodes.node0_validator.pdf_part_from_gcs")
@patch("backend.nodes.node0_validator.get_flash_model")
def test_node0_pass(mock_get_model, mock_pdf_part):
    mock_response = MagicMock()
    mock_response.text = '{"result": "PASS", "reason": "This is an ML paper."}'
    mock_get_model.return_value.generate_content.return_value = mock_response
    mock_pdf_part.return_value = MagicMock()

    result = run_node0("gs://test-bucket/papers/abc/paper.pdf")

    assert isinstance(result, ScopeValidationResult)
    assert result.result == "PASS"
    assert result.reason == "This is an ML paper."


@patch("backend.nodes.node0_validator.pdf_part_from_gcs")
@patch("backend.nodes.node0_validator.get_flash_model")
def test_node0_fail(mock_get_model, mock_pdf_part):
    mock_response = MagicMock()
    mock_response.text = '{"result": "FAIL", "reason": "Not an ML paper."}'
    mock_get_model.return_value.generate_content.return_value = mock_response
    mock_pdf_part.return_value = MagicMock()

    result = run_node0("gs://test-bucket/papers/abc/paper.pdf")

    assert result.result == "FAIL"
    assert result.reason == "Not an ML paper."


@patch("backend.nodes.node0_validator.pdf_part_from_gcs")
@patch("backend.nodes.node0_validator.get_flash_model")
def test_node0_strips_markdown_fences(mock_get_model, mock_pdf_part):
    mock_response = MagicMock()
    mock_response.text = '```json\n{"result": "PASS", "reason": "ML paper"}\n```'
    mock_get_model.return_value.generate_content.return_value = mock_response
    mock_pdf_part.return_value = MagicMock()

    result = run_node0("gs://test-bucket/papers/abc/paper.pdf")

    assert result.result == "PASS"
    assert result.reason == "ML paper"
