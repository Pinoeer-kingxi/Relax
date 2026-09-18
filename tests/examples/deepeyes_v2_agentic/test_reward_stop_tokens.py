from __future__ import annotations

import pytest

from examples.deepeyes_v2_agentic import reward_deepeyes_v2 as reward


GPU_STOPPED_SEARCH_RESPONSE = (
    '<think>search first</think><tool_call>{"name":"search","arguments":{"query":"SGLang"}}'
    "<think>use the evidence</think><answer>https://github.com/sgl-project/sglang</answer>"
)


def test_gpu_omitted_tool_close_still_counts_as_valid_search_call():
    assert reward._has_valid_tool_call(GPU_STOPPED_SEARCH_RESPONSE)
    assert reward._count_search_tool_calls(GPU_STOPPED_SEARCH_RESPONSE) == 1


def test_search_reward_accepts_gpu_omitted_stop_tag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(reward, "_judge_chat", lambda **_kwargs: r"\boxed{Yes}")

    result = reward.compute_score_search(
        GPU_STOPPED_SEARCH_RESPONSE,
        "https://github.com/sgl-project/sglang",
        {"question": "What is the official SGLang repository?"},
    )

    assert result["acc"] == 1.0
    assert result["format"] == 1.0
    assert result["search_penalty"] == 0.1
    assert result["format_error_reason"] == ""


def test_search_reward_rejects_truncated_tool_json(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(reward, "_judge_chat", lambda **_kwargs: r"\boxed{Yes}")
    response = '<think>x</think><tool_call>{"name":"search"<answer>x</answer>'

    result = reward.compute_score_search(response, "x", {"question": "q"})

    assert result["format"] == 0.0
    assert "tool_call_tag_mismatch" in result["format_error_reason"]


def test_search_reward_rejects_arbitrary_text_after_omitted_stop_tag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(reward, "_judge_chat", lambda **_kwargs: r"\boxed{Yes}")
    response = (
        '<think>x</think><tool_call>{"name":"search","arguments":{"query":"x"}} trailing garbage <answer>x</answer>'
    )

    result = reward.compute_score_search(response, "x", {"question": "q"})

    assert result["format"] == 0.0
    assert result["search_penalty"] == 0.0
    assert "tool_call_tag_mismatch" in result["format_error_reason"]
