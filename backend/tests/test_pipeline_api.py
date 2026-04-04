import time
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def wait_for_completion(run_id: str, timeout: float = 3.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/runs/{run_id}")
        response.raise_for_status()
        payload = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Run {run_id} did not finish within {timeout} seconds")


def test_run_pipeline_from_demo_cache():
    response = client.post(
        "/run",
        json={"use_demo_cache": True, "demo_cache_key": "attention_is_all_you_need"},
    )

    assert response.status_code == 200
    accepted = response.json()
    assert accepted["poll_url"].endswith(accepted["run_id"])

    final_state = wait_for_completion(accepted["run_id"])
    assert final_state["status"] == "completed"
    assert final_state["outputs"]["node0"]["result"] == "PASS"
    assert "node3" in final_state["outputs"]


@patch("backend.pipeline.run_node3")
@patch("backend.pipeline.run_node2")
@patch("backend.pipeline.run_node1")
@patch("backend.pipeline.run_node0")
@patch("backend.pipeline.upload_pdf_from_url")
def test_run_pipeline_live_path(
    mock_upload,
    mock_node0,
    mock_node1,
    mock_node2,
    mock_node3,
):
    mock_upload.return_value = "gs://test-bucket/papers/run-1/paper.pdf"
    mock_node0_response = MagicMock()
    mock_node0_response.model_dump.return_value = {
        "result": "PASS",
        "reason": "Transformer paper.",
    }
    mock_node0_response.result = "PASS"
    mock_node0_response.reason = "Transformer paper."
    mock_node0.return_value = mock_node0_response

    mock_blueprint = MagicMock()
    mock_blueprint.model_dump.return_value = {
        "model_type": "transformer",
        "architecture": {
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 2048,
            "vocab_size": 37000,
            "max_seq_len": 512,
        },
        "objective": "cross_entropy",
        "key_operations": ["attention"],
        "math_notes": "d_model must be divisible by n_heads",
    }
    mock_node1.return_value = mock_blueprint
    mock_node2.return_value = {"status": "ok", "output_dir": "outputs/scaffold"}
    mock_node3.return_value = {"status": "ok", "output_dir": "outputs/hw"}

    response = client.post(
        "/run",
        json={"pdf_url": "https://example.com/paper.pdf"},
    )

    assert response.status_code == 200
    accepted = response.json()
    final_state = wait_for_completion(accepted["run_id"])

    assert final_state["status"] == "completed"
    assert final_state["outputs"]["gcs_uri"].startswith("gs://")
    assert final_state["outputs"]["node2"]["status"] == "ok"
    assert final_state["outputs"]["node3"]["status"] == "ok"


def test_run_requires_input():
    response = client.post("/run", json={})
    assert response.status_code == 400
