# Coding-Agent RL

This directory provides an example of running end-to-end **SWE (Software-Engineering) coding-agent RL** with slime: a real coding agent (claude-code CLI) drives `Read/Edit/Grep/Bash/Agent` tools inside a fresh sandbox per sample, the model produces a `git diff`, and the diff is graded against the dataset's test harness in a second clean sandbox (no test-cheating).

**Qwen3.5-4B + public e2b.dev (1 node):** see [README_qwen35_4b_public_e2b.md](./README_qwen35_4b_public_e2b.md) for the local diff summary and runbook (template build, Cloudflare tunnel, smoke data).

Two example files, the shared harness package, and one shared adapter implement the loop:

- `generate.py` — per-sample `generate()` registered via `--custom-generate-function-path`. Boots the sandbox, prepares the SWE workspace, runs the coding harness (claude-code), captures the diff, scores it, and emits one or more `Sample`s back to slime.
- `slime.agent.adapters.AnthropicAdapter` — the shared Anthropic Messages adapter. claude-code talks to it as if it were Anthropic; the adapter tokenizes the current message history each turn, records prompt/output token snapshots, preserves model-generated tokens (`loss_mask=1`) only while later prompts stitch onto them, and masks template/observation tokens (`0`). Each turn is routed into a per-session message tree inside `slime.agent.trajectory.TrajectoryManager`; any divergence in the prompt prefix forks a new branch, so sub-agent dispatches and auto-compaction are handled as separate root-to-leaf chains. `get_trajectory` linearizes each leaf chain into one `Sample`.
- `slime.agent.harness` — harness-agnostic coding-agent lifecycle (install CLI, write config, spawn detached, poll done-marker). `BaseHarness` defines the contract; `CLAUDE_CODE` / `CODEX` are the shipped implementations. Adding a harness is one new file. The shared sandbox contract lives in `slime.agent.sandbox.Sandbox`.
- `swe.py` — harness-agnostic SWE task layer built on `slime.agent.sandbox`: `prepare_workspace` (pre_commands + PROBLEM_STATEMENT.md), `git_diff` (patch capture), and `evaluate` (fresh-sandbox grading). `SWE_PROMPT` is the task instruction handed to whichever harness runs.

`generate.py` owns one `AnthropicAdapter` instance. For each sample it calls
`adapter.open_session(...)` before starting claude-code, serves `adapter.app` as
the Anthropic-compatible endpoint, and drains trainable `TokenSegment`s with
`await adapter.finish_session(...)` when the trajectory ends.

## Environment Setup

The slime training stack itself follows the standard setup. On top of that you need:

1. **An E2B-compatible sandbox cluster** (or any provider that speaks the E2B SDK). Configure via `E2B_API_KEY` (e.g. the standard `e2b_xxx` key from https://e2b.dev, or any internal endpoint that accepts the same SDK). The official SDK validates this value locally, so internal gateways that ignore auth still need a syntactically valid `e2b_` + 40 hex-character placeholder.
2. **Host-side tarballs** that get uploaded into each sandbox at boot:
   - Node 22 (`node-v22.x-linux-x64.tar.xz`) — exported as `SLIME_AGENT_NODE_TARBALL`.
   - Claude Code CLI npm tarball (`anthropic-ai-claude-code-local-linux-x64.tgz`) — exported as `SLIME_AGENT_CC_TARBALL`.
3. **An image routing key** (`SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY`, legacy `SWE_SANDBOX_IMAGE_METADATA_KEY` still accepted) — the metadata key your E2B gateway uses to route a boot to a specific image (e.g. `image`). Each sample's `metadata.image` is passed under this key when booting the sandbox.
4. **Network reachability**: each sandbox dials back to the host's Anthropic adapter over `http://${ADAPTER_PUBLIC_HOST}:${ADAPTER_PORT}`. The adapter host must be reachable from inside the sandboxes (set `ADAPTER_PUBLIC_HOST` to a routable IP, not `127.0.0.1`).

## Dataset Format

Standard slime JSONL with three keys. Rows may mix ScaleSWE and Tmax via
``metadata.protocol`` (``scaleswe`` default when omitted; ``tmax`` for
terminal-task images). Per-sample harness selection uses ``metadata.agent``
(``claude_code`` / ``codex`` / ``pi`` / ``opencode`` / ``miniswe``; aliases
``cc``, ``mini-swe-agent``). Missing ``agent`` falls back to ``SWE_AGENT``
(default ``claude_code``). All agents except ``codex`` dial the Anthropic (CC)
adapter; ``codex`` uses the OpenAI chat-completions route on the same adapter URL.

```jsonc
{
  "prompt": [{"role": "user", "content": "<problem>"}],
  "label": "<instance_id or grader label>",
  "metadata": {
    "protocol": "scaleswe",  // or "tmax"; omit => scaleswe
    "agent": "claude_code",  // or codex | pi | opencode | miniswe
    "image": "your-registry/swe-image:<tag>",  // sandbox image reference
    "workdir": "/workspace/<repo>",            // tmax: usually /home/user
    "problem_statement": "<issue body>",
    // scaleswe graders (exactly one):
    "swepro": { /* SWE-bench Pro test harness — preferred */ },
    "eval_cmd": "pytest -x tests/...",
    // sweb-style: metadata.remote_env_info.f2p_script
    // tmax grader (deferred until after Claude Code exits):
    "test_sh": "#!/bin/bash\n..."
  }
}
```

Wire it up with `--input-key prompt --label-key label --metadata-key metadata`.

Multi-agent smoke (5 agents × 2 rows each):

```bash
python examples/coding_agent_rl/build_agents_smoke_jsonl.py
# -> data/scaleswe_agents_smoke.jsonl
# -> data/tmax_agents_smoke.jsonl
```

Host tarballs / wheels (set the ones you need for the agents in the jsonl):

- `SLIME_AGENT_NODE_TARBALL` — Node 22 (npm CLIs)
- `SLIME_AGENT_CC_TARBALL` — Claude Code
- `SLIME_AGENT_CODEX_TARBALL` — Codex
- `SLIME_AGENT_PI_TARBALL` — Pi
- `SLIME_AGENT_OPENCODE_TARBALL` — OpenCode
- `SLIME_AGENT_MINISWE_WHEEL` — mini-swe-agent

### Mixing ScaleSWE + Tmax

```bash
# Convert each source (tmax pulls HF task-data for test_sh):
python examples/coding_agent_rl/convert_scaleswe_to_slime.py \
  --src /path/to/scaleswe.jsonl --dst data/scaleswe.jsonl
python examples/coding_agent_rl/convert_tmax_to_slime.py \
  --dst data/tmax.jsonl --limit 200

# Offline mix; SWE_TRAIN_PROTOCOL is only a fallback when protocol is missing:
cat data/scaleswe.jsonl data/tmax.jsonl | shuf > data/mixed.jsonl
PROMPT_DATA=$PWD/examples/coding_agent_rl/data/mixed.jsonl \
  bash examples/coding_agent_rl/run_qwen35_4b_swe_1node_docker_async.sh
```

Tmax grading runs **in the same agent sandbox** after the harness exits (tests
are written only then — not present during the CC episode). ScaleSWE still uses
git_diff + a fresh eval sandbox.

## Mid-turn LLM offload (PyroDash)

Offload is implemented **on this black-box agent path**, not as post-hoc math custom-rm.
**Every agent round** is: agent → SLM → (if offload span) GLM → **complete** reply → agent.

1. Offload usage is appended by the coding **adapter** to the request `system` **after** Claude Code's full system (including `gitStatus`). Override text with `SLIME_AGENT_OFFLOAD_SYSTEM_APPEND`; `SWE_PROMPT` is unchanged.
2. Actor may emit `<|llm_offload|>N<|/llm_offload|>` **inside thinking** mid-turn (`N`: `0`=no think, `1–5`=`high`, `6–9`=`max`) — before `</think>` (Qwen often omits the opening `<think>` from `output_ids`). Outside think: no GLM call + think-format penalty on solved reward. PyroDash: open **248077**, close **248078** (stop on close).
3. Adapter waits for GLM (when needed), merges SLM prefix + GLM into one assistant message, then responds to Claude Code / Codex.
4. Only local-model `output_ids` are trained; GLM text lands in history with `loss_mask=0` on later turns.
5. Train reward: if solved, `1 - λ * cost_ratio`, then subtract `OFFLOAD_THINK_FORMAT_PENALTY` (default `0.25`) once if any offload span was outside thinking (after `</think>`); else `0`. Empty patches are never solved.

```bash
# CPU plumbing smoke (mock GLM, 2 adapter turns; no GPU/Docker)
python examples/coding_agent_rl/smoke_offload_adapter.py

# 1-sample docker train smoke
export DASHSCOPE_API_KEY=...
export DASHSCOPE_BASE_URL=http://host:8000/v1
bash examples/coding_agent_rl/run_pyrodash4b_swe_offload_smoke.sh

# full async docker train (after convert_pyrodash4b_to_torch_dist.sh)
bash examples/coding_agent_rl/run_pyrodash4b_swe_offload_1node_docker_async.sh
```

Key env: `SLIME_AGENT_OFFLOAD=1`, `OFFLOAD_EFFICIENCY_LAMBDA`, `OFFLOAD_THINK_FORMAT_PENALTY`, `ROLLOUT_STOP_TOKEN_IDS` (includes offload id), `DASHSCOPE_*`.

> `examples/llm_offload/` is a separate **math** GRPO sketch; do not use it for coding-agent offload.

## Running the Script

Override the paths at the top of the launcher, then run from a long-lived shell on the Ray head node (do **not** wrap in `nohup` — Ray child processes get cleaned up with it):

```bash
cd slime/

export HF_CHECKPOINT=/path/to/Qwen3.6-35B-A3B
export REF_MODEL_PATH=/path/to/Qwen3.6-35B-A3B_torch_dist
export PROMPT_DATA=/path/to/swe_train.jsonl
export SLIME_AGENT_NODE_TARBALL=/path/to/node-v22.20.0-linux-x64.tar.xz
export SLIME_AGENT_CC_TARBALL=/path/to/anthropic-ai-claude-code-local-linux-x64.tgz

# Sandbox provider:
export E2B_API_KEY=e2b_xxx                       # real key for e2b.dev; a syntactically
                                                 # valid placeholder if your gateway ignores auth
export SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY=image   # metadata key your gateway routes images by

bash examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh
```

The launcher fans Ray out to every worker listed in `$HOSTFILE` (default
`/root/mpi_rack_hostfile`, one worker IP per line, reachable over passwordless
SSH as `root`) — create that file (or point `HOSTFILE` at your own) before
launching. It then dumps every rollout to `runs/${EXP_TAG}_${STAMP}/rollout_dumps/`
and tees stdout into `runs/${EXP_TAG}_${STAMP}/run.log`.

## New Arguments

`generate.py` is wired in through slime's standard custom-generate hook:

```bash
ROLLOUT_ARGS=(
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-context-len 96000
   --rollout-max-response-len 32768
   --rollout-stop-token-ids 248046 248044
   --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)
```

The SGLang server must expose Qwen3.6's tool-call and reasoning parsers so claude-code's tool invocations are parsed correctly:

```bash
SGLANG_ARGS=(
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
   ...
)
```

## SWE-specific Environment Knobs

All set in the launcher; tune per cluster.

Env vars split by layer. `SLIME_AGENT_*` are the reusable agent library's
contract (read inside `slime/agent/`); `SWE_*` are this SWE example's task knobs;
`ADAPTER_*` are host-side deployment/reply-path addresses read only by
`generate.py`. Keep new vars on the prefix that matches the layer that reads them.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ADAPTER_PUBLIC_HOST` | `${MASTER_ADDR}` | Public IP the sandbox uses to reach the Anthropic adapter. **Must be routable from inside the sandbox.** |
| `ADAPTER_BIND_HOST` / `ADAPTER_PORT` | `0.0.0.0` / `18001` | Bind address of the Anthropic adapter on the host. |
| `E2B_API_KEY` | — | E2B (or compatible) API key. |
| `SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY` | — | **Required.** Which metadata key the E2B gateway routes images by (e.g. `image`); each sample's `metadata.image` is passed under it. (Legacy `SWE_SANDBOX_IMAGE_METADATA_KEY` still accepted.) |
| `SLIME_AGENT_NODE_TARBALL` | — | Host path to Node 22 tarball uploaded into each sandbox. |
| `SLIME_AGENT_CC_TARBALL` | — | Host path to the Claude Code CLI npm tarball. |
| `SLIME_AGENT_CC_EXTRA_ARGS` | (see launcher) | Extra flags appended to the `claude` CLI invocation — registers the read-only `investigator` sub-agent, disables `WebFetch`/`WebSearch`, disables slash commands. |
| `SLIME_AGENT_OFFLOAD_SYSTEM_APPEND` | `OFFLOAD_SYSTEM_PROMPT_APPEND` | SLM-only offload instructions; adapter appends after full CC system (incl. `gitStatus`). |
| `SLIME_AGENT_CC_EXTRA_ENVS` | unset | JSON object of extra env vars exported into the `claude` process — escape hatch for env-only knobs (`MAX_THINKING_TOKENS`, `BASH_MAX_TIMEOUT_MS`, ...). Merged last, so it can also override the built-in defaults. |
| `SWE_AGENT_TIME_BUDGET_SEC` | `1800` | Wallclock budget for the in-sandbox agent CLI itself (think/edit/run). |
| `SWE_EVAL_TIMEOUT_SEC` | `600` | Wallclock cap on the evaluator sandbox. |
| `SWE_ROLLOUT_GUARD_SEC` | `agent+eval+180` | Outer safety net wrapping the whole rollout (boot + workspace + agent + diff + eval). Auto-derived if unset. |
| `SWE_BOOT_CONCURRENCY` | `16` | Cap on simultaneous sandbox boots (eases h2/SSL long-tail). |
| `SWE_CC_PROMPT` | unset | Optional override for the user-turn prompt. Setting this to require sub-agent dispatch is the most reliable way to maximize fan-out. |
| `SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS` | `8192` | Trajectory fork/merge threshold (tokens). Higher → fewer `TOKEN_FORK` segments (more REALIGN / assistant-rewrite merge), longer Samples, and more wipe risk (`loss_mask=0`). Does not collapse subagent `TREE_LEAF` branches. Unset falls back to manager default `1024`. |

`--rollout-max-response-len` is the per-turn generation cap passed to each
SGLang `/generate` call as `max_new_tokens`. `--rollout-max-context-len` is the
multi-turn prompt+response budget enforced only during generation: each turn
clamps `max_new_tokens` to the remaining context. Trajectory merge/export keeps
the emitted segments and does not drop them for length.
The Anthropic adapter reuses `--sglang-tool-call-parser` and
`--sglang-reasoning-parser` for output parsing, so those flags must match the
served model.

## String-in, Token-out Trajectories

The coding-agent environment is string/message based: claude-code sends
Anthropic Messages requests, receives streamed text/thinking/tool-use blocks,
and later sends back rendered tool observations. Training, however, must stay
token based. A trajectory is only a valid RL target when the optimized tokens
are the same tokens the rollout model actually sampled.

The Anthropic adapter therefore follows a **string in, token out** contract:

- Each incoming message history is rendered with the served model's chat
  template and sent to SGLang as `input_ids`.
- SGLang is called with `return_logprob=True`; the adapter records the exact
  `prompt_ids`, sampled `output_ids`, and per-token rollout logprobs for that
  turn.
- At training export time, samples are assembled from those saved token ids.
  The decoded `response` field is only a readable sidecar; it is not
  re-tokenized to recover the training sequence.

Multi-turn agents still force the adapter to tokenize later message
histories, because tool observations and claude-code's own compacted messages
arrive as strings. `slime.agent.trajectory.TrajectoryManager` routes
those later prompts against the saved token stream:

- New prompt suffixes that are tool/user/environment context are appended with
  `loss_mask=0`.
- Fresh model outputs from SGLang are appended with `loss_mask=1`.
- If a later prompt no longer token-matches an earlier sampled output, the
  unmatched suffix is dropped. If the drift cuts through the middle of a
  previous model output, the retained prefix of that whole output turn is also
  assigned `loss_mask=0`.

That last case is the important correctness guard. A re-tokenization mismatch
can make a string-level conversation look continuous while token-level
provenance is broken. slime keeps the context needed to continue the agent, but
does not backprop through tokens whose sampled origin can no longer be proven.
The unit tests in `tests/test_agent/test_trajectory_manager_branching.py` cover matched
prefixes, skipped turns, split-output drift, changed token counts, and
prompt-base restarts.

## Fan-out Semantics

- `generate()` returns `list[Sample]` — one Sample per root-to-leaf chain in the per-session message tree.
- Per-trajectory reward is split as `reward / K` across chains; `rollout_id` is shared so the per-rollout-mean loss reducer still counts the trajectory once.
- Sub-agent dispatch and auto-compaction increase `K` (each prompt-prefix divergence forks a new branch), so the effective batch after flatten can be much larger than `rollout_batch_size * n_samples_per_prompt`.

## Porting to a New Sandbox Backend

`slime.agent.sandbox.Sandbox` exposes the shared sandbox contract.
Shipped backends:

- `E2BSandbox` — remote E2B / E2B-compatible gateway (default)
- `DockerSandbox` — local Docker engine (`SLIME_AGENT_SANDBOX_BACKEND=docker`)

Prefer `make_sandbox(image)` so examples switch backends via env:

```python
from slime.agent.sandbox import make_sandbox

async with make_sandbox(image) as sb:
    await sb.exec(cmd, user=..., check=..., timeout=...)
    await sb.write_file(sandbox_path, content_or_host_path, user=...)
    await sb.read_file(sandbox_path, user=...)
```

Local Docker smoke (no E2B)::

```bash
python examples/coding_agent_rl/smoke_docker_sandbox.py
# ScaleSWE image (must be pulled locally first):
python examples/coding_agent_rl/smoke_docker_sandbox.py \
  --image aweaiteam/scaleswe:arviz-devs_preliz_pr249 \
  --workdir /workspace/preliz --pull
```

Training with local Docker (adapter must be reachable from containers;
on Linux bridge networks prefer `ADAPTER_PUBLIC_HOST=host.docker.internal`)::

```bash
# Dedicated launcher (recommended):
bash examples/coding_agent_rl/run_qwen35_4b_swe_1node_docker.sh

# Or set env yourself then call the generic 1-node script:
export SLIME_AGENT_SANDBOX_BACKEND=docker
export ADAPTER_PUBLIC_HOST=host.docker.internal
# metadata.image is a real Docker image name/tag
bash examples/coding_agent_rl/run_qwen35_4b_swe_1node.sh
```

Reimplement the same methods on Modal / a local VM and everything in `generate.py` keeps working unchanged.
