# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU integration for metadata -> committed export -> weighted metrics."""

from types import SimpleNamespace

from relax.agentic.session.service import AgenticSessionShard
from relax.agentic.session.state import SessionForest
from relax.utils.metrics.speculative_metrics import compute_speculative_metrics
from relax.utils.speculative import SpeculativeCounts


class _Tokenizer:
    def decode(self, tokens, **kwargs):
        del kwargs
        return "".join(chr(token) for token in tokens)


def test_cpu_metadata_accumulation_reaches_committed_generation() -> None:
    request = SimpleNamespace(
        request_id="metadata-request",
        pending_weight_version_delta=[],
        pending_spec_counts=None,
        pending_spec_delta={
            "spec_accept_token_num": 0,
            "spec_draft_token_num": 0,
            "spec_verify_ct": 0,
            "completion_token_num": 0,
        },
        pending_prefix_cache_delta={"cached_tokens": 0, "total_prompt_tokens": 0},
    )
    AgenticSessionShard._accumulate_request_meta(
        request,
        meta_info={
            "spec_num_correct_drafts": 1,
            "spec_num_proposed_drafts": 2,
            "spec_verify_ct": 1,
            "completion_tokens": 2,
        },
    )
    assert request.pending_spec_counts == SpeculativeCounts(1, 2, 1, 2)
    assert request.pending_spec_delta["spec_accept_token_num"] == 1

    forest = SessionForest.create_empty(session_id="metadata-session")
    prompt = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "p"}],
        train_token_delta=[112],
        rollout_token_delta=[112],
    )
    leaf = forest.append_resp(
        parent_state_hash=prompt.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": "a"}],
        token_delta=[97],
        logprob_delta=[-0.1],
        spec_delta=request.pending_spec_delta,
        spec_counts=request.pending_spec_counts,
        export_metadata_patch={"request_id": request.request_id},
    )
    sample = forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=_Tokenizer())
    report = compute_speculative_metrics([sample])
    assert report["spec/accepted_total"] == 1
    assert report["spec/proposed_total"] == 2
    assert report["spec/accept_rate"] == 0.5
