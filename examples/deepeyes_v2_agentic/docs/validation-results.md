# Task 10 validation results

Validation was run on 2026-09-18 against the Task 10 implementation in this
repository. Credentials, local absolute paths, and model weights are excluded
from this document. The commands below produce machine-readable,
credential-free JSON reports with configuration and result fingerprints.

## Acceptance matrix

| Requirement                         | Validation                                                                           | Result                                                                                            |
| ----------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Default example runs offline        | Scripted model plus the default mock backend through the real `app.agent` subprocess | Passed; search was exercised and a final answer was produced                                      |
| Search-R1-compatible backend        | Repository E5/FAISS server, 512-document fixture, four golden queries                | Passed; 5/5 HTTP requests completed, `Hit@5=1.0`, `MRR=1.0`                                       |
| External search API                 | Tavily basic search with direct networking (`trust_env: false`)                      | Passed; five normalized, non-placeholder results                                                  |
| Unified result contract             | Smoke checks require `elapsed_time` plus normalized `data[]` rows                    | Passed for mock, retriever, and external backends                                                 |
| Timeout, retry, and service failure | Unit and subprocess integration tests                                                | Passed; failures return the existing `"Error"` observation and do not terminate the agent process |

## Search-R1-compatible retrieval

The bounded native fixture uses:

- `intfloat/e5-small-v2@ffb93f3bd4047442299a41ebb6fa998a38507c52`
- `Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3`
- 512 documents in a normalized 384-dimensional inner-product FAISS index
- the same `query: ` / `passage: ` prefixes and attention-mask mean pooling as
  the repository retriever

The service ran on one explicitly selected RTX A6000. Five live HTTP requests
completed successfully, and all four golden answers ranked first.

```bash
python examples/deepeyes_v2_agentic/scripts/smoke_search.py \
  --mode tool \
  --config examples/deepeyes_v2_agentic/configs/search/retriever.yaml \
  --query "Who wrote Hamlet?" \
  --require-live \
  --report /tmp/task10-retriever-smoke.json

```

## Tavily external search

The external adapter used the committed Tavily configuration with direct
networking and loaded the credential from `TAVILY_API_KEY`.
The smoke request returned five real results; schema, successful-search, and
non-placeholder checks all passed.

```bash
python examples/deepeyes_v2_agentic/scripts/smoke_search.py \
  --mode tool \
  --config examples/deepeyes_v2_agentic/configs/search/external-tavily.yaml \
  --query "What organization released the Qwen3 model family?" \
  --require-live \
  --report /tmp/task10-tavily-smoke.json
```

## Automated tests

The focused CPU suite covers all three backends, normalization, configuration,
timeouts, retry policy, malformed responses, agent subprocess wiring, and the
Search-R1-compatible request contract.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/examples/deepeyes_v2_agentic
```
