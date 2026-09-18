# Task 10 validation results

Validation was run on 2026-09-18 against the Task 10 implementation in this
repository. Credentials, local absolute paths, and model weights are excluded
from this document. The commands below produce machine-readable JSON reports
that include the Git revision, dirty-state flag, configuration fingerprint,
and result fingerprint.

## Acceptance matrix

| Requirement                         | Validation                                                                           | Result                                                                                            |
| ----------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Default example runs offline        | Scripted model plus the default mock backend through the real `app.agent` subprocess | Passed; search was exercised and a final answer was produced                                      |
| Search-R1-compatible backend        | Repository E5/FAISS server, 512-document fixture, four golden queries                | Passed; 5/5 HTTP requests completed, `Hit@5=1.0`, `MRR=1.0`                                       |
| External search API                 | Tavily basic search with direct networking (`trust_env: false`)                      | Passed; five normalized, non-placeholder results                                                  |
| Unified result contract             | Smoke checks require `elapsed_time` plus normalized `data[]` rows                    | Passed for mock, retriever, and external backends                                                 |
| Timeout, retry, and service failure | Unit and subprocess integration tests                                                | Passed; failures return the existing `"Error"` observation and do not terminate the agent process |
| Real training closure               | Six RTX A6000 GPUs, Qwen3.6-35B-A3B actor/rollout, Qwen2.5-1.5B judge, LoRA          | Passed one full rollout and optimizer step                                                        |

## Search-R1-compatible retrieval

The bounded native fixture uses:

- `intfloat/e5-small-v2@ffb93f3bd4047442299a41ebb6fa998a38507c52`
- `Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3`
- 512 documents in a normalized 384-dimensional inner-product FAISS index
- the same `query: ` / `passage: ` prefixes and attention-mask mean pooling as
  the repository retriever

The service ran on one explicitly selected RTX A6000. Its final metrics were
five requests, five completions, zero failures, zero cancellations, and zero
overload rejections. All four golden answers ranked first.

```bash
python examples/deepeyes_v2_agentic/scripts/smoke_search.py \
  --mode tool \
  --config examples/deepeyes_v2_agentic/configs/search/retriever.yaml \
  --query "Who wrote Hamlet?" \
  --require-live \
  --report /tmp/task10-retriever-smoke.json

python examples/deepeyes_v2_agentic/scripts/evaluate_search_quality.py \
  --input /path/to/golden-search-fixture.jsonl \
  --config examples/deepeyes_v2_agentic/configs/search/retriever.yaml \
  --query-field prompt \
  --answers-field extra_info.answers \
  --top-k 5 \
  --require-all-success \
  --min-answer-hit-rate 1.0 \
  --report /tmp/task10-retriever-quality.json
```

## Tavily external search

The external adapter used the committed Tavily configuration with direct
networking and loaded the credential from an explicitly selected dotenv key.
The smoke request returned five real results; schema, successful-search, and
non-placeholder checks all passed.

```bash
python examples/deepeyes_v2_agentic/scripts/smoke_search.py \
  --mode tool \
  --config examples/deepeyes_v2_agentic/configs/search/external-tavily.yaml \
  --query "What organization released the Qwen3 model family?" \
  --require-live \
  --credentials-env-file /path/to/private.env \
  --credentials-key tvly_api_key \
  --report /tmp/task10-tavily-smoke.json
```

## Six-GPU training closure

The real training smoke used the pinned resource bundle documented in
`search-e2e-resources.md`:

- Qwen3.6-35B-A3B actor/reference/rollout with actor TP1/PP3/EP2
- three SGLang rollout engines at TP2
- LoRA rank 16, alpha 32
- Qwen2.5-1.5B judge on CPU
- four sessions in two rollout groups

Ray job `raysubmit_LGvUJbc9bJARw8fR` completed successfully. The run reached
rollout cleanup, log-probability computation, backward, optimizer step, and
post-step LoRA synchronization (`v=2`) without leaked sessions. The optimizer
step recorded:

| Metric                    |                   Value |
| ------------------------- | ----------------------: |
| `train/loss`              | `0.0006746798753738403` |
| `train/pg_loss`           | `0.0006746798753738403` |
| `train/grad_norm`         |   `0.36290308833122253` |
| `train/global_batch_size` |                     `4` |
| `train/lr`                |                  `1e-6` |
| Actor training tokens     |                 `14751` |
| Actor training time       |              `273.98 s` |
| Total train-step time     |              `352.85 s` |

This is an end-to-end integration and optimizer-closure test. Retrieval and
answer quality are evaluated separately above so a deliberately small random
training batch is not presented as a benchmark score.

## Automated tests

The focused CPU suite covers all three backends, normalization, configuration,
timeouts, retry policy, circuit breaking, malformed and oversized responses,
agent subprocess wiring, evaluator behavior, prompt-injection handling, and
Search-R1 request batching and overload behavior. Separate Megatron tests cover
the LoRA weight-conversion and synchronization paths exercised by the GPU run.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/examples/deepeyes_v2_agentic \
  tests/examples/search_r1 \
  tests/utils/test_http_utils.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/backends/megatron/weight_update/test_broadcast_converted.py \
  tests/backends/megatron/weight_update/test_lora_weight_sync.py
```
