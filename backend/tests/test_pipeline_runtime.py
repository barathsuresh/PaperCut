from backend.services.pipeline_runtime import new_session_payload, persist_graph_results, sse
from backend.schemas.contract import ArchitectureBlueprint


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


def test_sse_formats_event_payload():
    event = sse("done", {"ok": True})

    assert event == 'data: {"type": "done", "ok": true}\n\n'


def test_new_session_payload_has_expected_shape():
    payload = new_session_payload("gs://bucket/paper.pdf")

    assert payload["gcs_uri"] == "gs://bucket/paper.pdf"
    assert payload["name"] == "Untitled Paper"
    assert payload["history"] == []
    assert "uploaded_at" in payload


def test_persist_graph_results_applies_all_state():
    session = {}
    final_state = {
        "scope_valid": True,
        "scope_reason": "In scope",
        "blueprint": SAMPLE_BLUEPRINT,
        "scaffold_code": {"status": "completed"},
        "cuda_blueprint": {"status": "completed"},
    }

    scope_valid, scope_reason, blueprint, scaffold, cuda = persist_graph_results(session, final_state)

    assert scope_valid is True
    assert scope_reason == "In scope"
    assert blueprint == SAMPLE_BLUEPRINT
    assert scaffold == {"status": "completed"}
    assert cuda == {"status": "completed"}
    assert session["node0_result"] == {"result": "PASS", "reason": "In scope"}
    assert session["node1_result"]["model_type"] == "transformer"
    assert session["node2_result"]["status"] == "completed"
    assert session["node3_result"]["status"] == "completed"
