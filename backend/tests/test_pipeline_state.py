from fastapi import HTTPException

from backend.schemas.contract import ArchitectureBlueprint
from backend.services.pipeline_state import (
    apply_blueprint_result,
    apply_node0_result,
    apply_node_output,
    node_error_status_code,
    pipeline_response,
    require_pipeline_prerequisite,
)


class DummyNode0Result:
    result = "PASS"
    reason = "In scope"


SAMPLE_BLUEPRINT = ArchitectureBlueprint(
    model_type="transformer",
    architecture={
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048,
        "vocab_size": 32000,
        "max_seq_len": 1024,
    },
    objective="cross-entropy",
    key_operations=["attention"],
    math_notes="d_model divisible by n_heads",
)


def test_apply_node0_result_updates_session():
    session = {}

    apply_node0_result(session, DummyNode0Result())

    assert session["node0_result"] == {"result": "PASS", "reason": "In scope"}


def test_apply_blueprint_result_updates_session():
    session = {}

    apply_blueprint_result(session, SAMPLE_BLUEPRINT)

    assert session["node1_result"]["model_type"] == "transformer"


def test_apply_node_output_updates_session():
    session = {}

    apply_node_output(session, "node2_result", {"status": "completed"})

    assert session["node2_result"]["status"] == "completed"


def test_require_pipeline_prerequisite_raises_on_missing_key():
    try:
        require_pipeline_prerequisite({}, "node1_result", "missing")
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail == "missing"
    else:
        raise AssertionError("Expected HTTPException for missing prerequisite")


def test_node_error_status_code_maps_known_error_types():
    assert node_error_status_code({"error_type": "timeout"}) == 504
    assert node_error_status_code({"error_type": "auth"}) == 502
    assert node_error_status_code({"error_type": "other"}) == 500


def test_pipeline_response_combines_node_outputs():
    response = pipeline_response(
        session_id="session-1",
        scope_valid=True,
        scope_reason="In scope",
        blueprint=SAMPLE_BLUEPRINT,
        scaffold={"status": "completed", "files": {"model.py": 123}},
        cuda={"status": "error", "error": "boom", "stub_files": ["a.cu"], "bottlenecks": []},
    )

    assert response.session_id == "session-1"
    assert response.node2_status == "completed"
    assert response.node3_status == "error"
    assert response.node2_files == {"model.py": 123}
    assert response.error == "boom"
