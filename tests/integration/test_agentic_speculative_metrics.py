# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU integration for metadata -> committed export -> weighted metrics.

Hand-checkable report for the branch case in ``test_cpu_agentic_export_report``:
``A=(1,2,1,2)``, ``B=(9,10,2,11)``, and ``C=(2,4,1,3)`` occur in exported
trajectories ``A->B`` and ``A->C``.  There are four record occurrences but
three unique generations, so totals are accepted/proposed/verify/completion
``(12,16,4,16)``; the weighted results are ``12/16 = 75%`` and ``16/4 = 4``.
The separate-session case verifies the acceptance example ``(1+9)/(2+10)``.
"""

from types import SimpleNamespace

from relax.agentic.session.service import AgenticSessionShard
from relax.agentic.session.state import SessionForest
from relax.utils.metrics.speculative_metrics import compute_speculative_metrics
from relax.utils.speculative import SpeculativeCounts


class _Tokenizer:
    def decode(self, tokens, **kwargs):
        del kwargs
        return "".join(chr(token) for token in tokens)


def test_cpu_agentic_export_report() -> None:
    forest = SessionForest.create_empty(session_id="cpu-session")
    prompt = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "p"}],
        train_token_delta=[112],
        rollout_token_delta=[112],
    )
    a = forest.append_resp(
        parent_state_hash=prompt.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": "a"}],
        token_delta=[97],
        logprob_delta=[-0.1],
        spec_counts=SpeculativeCounts(1, 2, 1, 2),
        export_metadata_patch={"request_id": "A"},
    )
    b = forest.append_resp(
        parent_state_hash=a.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": "b"}],
        token_delta=[98],
        logprob_delta=[-0.1],
        spec_counts=SpeculativeCounts(9, 10, 2, 11),
        export_metadata_patch={"request_id": "B"},
    )
    c = forest.append_resp(
        parent_state_hash=a.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": "c"}],
        token_delta=[99],
        logprob_delta=[-0.1],
        spec_counts=SpeculativeCounts(2, 4, 1, 3),
        export_metadata_patch={"request_id": "C"},
    )
    samples = [forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=_Tokenizer()) for leaf in (b, c)]
    report = compute_speculative_metrics(samples)
    assert report["spec/unique_generation_count"] == 3
    assert report["spec/record_occurrence_count"] == 4
    assert report["spec/accept_rate"] == 12 / 16
    assert report["spec/tokens_per_verify"] == 16 / 4


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
