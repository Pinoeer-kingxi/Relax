# Search end-to-end resources

The text-search adapter itself is CPU/network code. GPU requirements begin at
the model and native-retriever validation layers, not at the mock or HTTP
contract layers.

| Level     | Components                                                                   | What a pass proves                                                   |
| --------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| L1        | CPU unit tests and HTTP mock transport                                       | Configuration, normalization, retry, limits, and error boundaries    |
| L2        | Scripted chat server, local retriever fixture, actual `app.agent` subprocess | Tool call → search → observation → follow-up answer wiring           |
| L3        | Real external service and a real local LLM/VLM on an assigned GPU            | Live authentication/network plus model consumption of search results |
| L3-native | E5 encoder, matching corpus/index, FAISS GPU server                          | Repository-native Search-R1 retrieval quality and protocol           |
| L4        | Actor/reference/rollout, judge, Ray/Megatron/SGLang, data and sandbox image  | Training rollout → reward → optimizer update closure                 |

Do not report L1/L2 as native Search-R1 deployment or training validation.
The example's original L4 recipe requests eight GPUs and cannot be inferred to
work from a single-model smoke run.

## Exact Qwen3.6 training bundle

The committed DeepEyes-V2 launcher is tied to the following reproducible
resource set. Pin revisions/digests; do not substitute a similarly named model
and report it as validation of this recipe.

| Role                     | Exact resource                                                                                         | Expected content size                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------ | ------------------------------------------- |
| Actor/reference/rollout  | `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`                                        | 71,926,865,825 bytes; 26 safetensors shards |
| Reward judge             | `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`                                  | 3,098,973,447 bytes                         |
| DeepEyes-V2 RL data      | `honglyhly/DeepEyesV2_RL@53b38b4b3bc3feb31706b0793745161588e42cba`                                     | 10,728,587,940 bytes; eight parquet files   |
| Relax CUDA environment   | `ghcr.io/redai-studio/relaxrl@sha256:cd431e1094646c347aa97bc9cb8c5ac8315d4e72b0b6321f1cafd1ae12e680c6` | 22,207,091,234 compressed layer bytes       |
| Per-session code sandbox | `apptainer_env/deepeyes_v2_kernel.def` built as `deepeyes_v2_kernel.sif`                               | about 115 MiB; verify imports after build   |

When the pinned Relax OCI image is materialized as a local SIF, write its
source digest to an adjacent `<image>.oci-digest` file. The training preflight
requires that marker to equal the digest above, inspects the SIF, and runs its
training-stack import and CUDA probe through `apptainer exec --nv`.

The Qwen3.6 checkpoint currently declares the
`Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe` architecture. That is
intentional: `scripts/models/qwen36-35B-A3B.sh` records that Qwen3.6 uses the
same architecture and fixes the 40-layer, 256-expert, top-8 MoE dimensions.
The directory name still must be exactly `Qwen3.6-35B-A3B`, because the launch
script resolves that path directly.

The original colocated recipe requests eight GPUs for both actor and rollout,
with tensor parallel 4, expert parallel 8, two GPUs per rollout engine,
bfloat16 checkpoint weights, optimizer CPU offload, and 16,384 max tokens per
GPU. These are recipe constraints, not a claim that any eight GPUs have enough
memory or interconnect bandwidth. Check all devices against the preflight
threshold immediately before launch.

After downloading with Git LFS, run `git lfs fsck` in each checkout before
discarding its object cache. The preflight also rejects unmaterialized LFS
pointer files and verifies every shard's exact size and Git LFS SHA-256, both
model directories, all eight converted parquets, the sandbox SIF, app Python,
Ray, PyTorch, Apptainer, and eight actually idle GPUs.

## Safe local model validation

1. Inspect GPU ownership and free memory; allocate explicit device IDs. Never
   stop unrelated processes or let a retriever discover every visible GPU.
2. Start the selected OpenAI-compatible model service with an explicit
   `CUDA_VISIBLE_DEVICES`, port, model path/revision, precision, context limit,
   and concurrency.
3. Export `OPENAI_BASE_URL`, a non-secret local `OPENAI_API_KEY`, and
   `OPENAI_MODEL`. If the tokenizer family differs from the committed Qwen
   config, copy `app/deepeyes_v2_config.yaml`, update its stop-token IDs, and
   point `DEEPEYES_V2_AGENT_CONFIG` at the copy.
4. Run `smoke_search.py --mode live-agent ... --require-live`. A final answer
   without a recorded search attempt is a failure.
5. Record the GPU UUID, port, model path, dependency versions, peak memory,
   report JSON, and cleanup status. Stop only processes started by the test.

Use pure text first, then at least one real image input when claiming VLM
coverage. The text-only search smoke does not prove visual-input handling.

## Native Search-R1 validation

The upstream retriever runs E5 inference and GPU FAISS. Its encoder revision,
query/document prefixes, pooling, normalization, corpus, and index must match.
For a bounded smoke fixture, build a matching corpus and index first:

```bash
python scripts/build_search_fixture.py \
  --output-dir /path/to/task10-retrieval-fixture \
  --model-path /path/to/e5-small-v2 \
  --model-revision ffb93f3bd4047442299a41ebb6fa998a38507c52 \
  --dataset-revision b08601e04326c79dfdd32d625aee71d232d685c3 \
  --dataset-cache-dir /path/to/huggingface-cache \
  --max-documents 512 --device cuda
```

This fixture includes four fixed golden documents and fills the remaining
rows from Wikitext. Its manifest records hashes, the E5 contract, and expected
golden-query document IDs. It validates deployment wiring and retrieval
quality on a small corpus; it is not the official Wiki18 benchmark.

Bind the server to only the allocated devices, for example:

```bash
CUDA_VISIBLE_DEVICES="$TASK10_RETRIEVER_GPU_IDS" python retrieval_server.py \
  --index_path /path/to/index --corpus_path /path/to/corpus \
  --retriever_model /path/to/e5 \
  --port 17389
```

Then copy `configs/search/retriever.yaml`, set a worker-reachable URL, and run
the tool and live-agent smoke modes. A loopback URL refers to each worker
itself in a multi-node launch.

## Credential boundary

The training launcher propagates only search configuration paths and
non-secret overrides. Provision API credentials in the worker environment or
use `external.auth.file` with a protected shared token file. Do not place a
token in YAML, Ray runtime JSON, `--agent-env`, shell xtrace, reports, or Git.
The smoke CLI's dotenv option reads only the explicitly named key and never
sources, interpolates, or evaluates the file.
