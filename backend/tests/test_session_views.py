from fastapi import HTTPException

from backend.services.session_views import (
    build_session_detail,
    build_session_summary,
    clean_history_entries,
    get_session_or_404,
    infer_language,
)


SAMPLE_SESSION = {
    "name": "Attention Is All You Need",
    "gcs_uri": "gs://bucket/papers/session/paper.pdf",
    "uploaded_at": "2026-04-04T00:00:00+00:00",
    "history": [
        {"role": "user", "text": "Explain the model"},
        {"role": "assistant", "text": "It is a transformer."},
        {"role": "system", "text": "ignored"},
        {"role": "user", "text": 123},
    ],
    "node0_result": {"result": "PASS", "reason": "In scope"},
    "node1_result": {"model_type": "transformer"},
    "node2_result": {"status": "completed"},
    "node3_result": {"status": "completed"},
}


def test_get_session_or_404_returns_session():
    sessions = {"abc": SAMPLE_SESSION}

    assert get_session_or_404(sessions, "abc") is SAMPLE_SESSION


def test_get_session_or_404_raises_for_missing_session():
    try:
        get_session_or_404({}, "missing")
    except HTTPException as exc:
        assert exc.status_code == 404
        assert exc.detail == "Session not found."
    else:
        raise AssertionError("Expected HTTPException for missing session")


def test_build_session_summary_uses_expected_fields():
    summary = build_session_summary("abc", SAMPLE_SESSION)

    assert summary.session_id == "abc"
    assert summary.name == "Attention Is All You Need"
    assert summary.scope_valid is True
    assert summary.model_type == "transformer"
    assert summary.node2_status == "completed"
    assert summary.node3_status == "completed"


def test_build_session_detail_uses_expected_fields():
    detail = build_session_detail("abc", SAMPLE_SESSION)

    assert detail.session_id == "abc"
    assert detail.gcs_uri == "gs://bucket/papers/session/paper.pdf"
    assert detail.scope_valid is True
    assert detail.scope_reason == "In scope"
    assert detail.has_chat_history is True


def test_clean_history_entries_filters_invalid_entries():
    history = clean_history_entries(SAMPLE_SESSION["history"])

    assert [entry.role for entry in history] == ["user", "assistant"]
    assert [entry.text for entry in history] == ["Explain the model", "It is a transformer."]


def test_infer_language_maps_expected_extensions():
    assert infer_language("model.py") == "python"
    assert infer_language("config.yaml") == "yaml"
    assert infer_language("kernel_stub.cu") == "cpp"
    assert infer_language("notes.txt") == "text"
