# LLM Offload GRPO (math-only sketch)

> **Coding-agent offload lives in `examples/coding_agent_rl/`** (mid-turn Adapter
> handoff + cost-aware SWE reward). This directory is a standalone **math** GRPO
> port of `pyroDash-training/train_test/phase2` and does **not** drive the
> black-box coding agent.

Slime port of phase2 math offload GRPO:

- Small model (PyroDash-4B) may emit `<|llm_offload|>` mid-reasoning.
- Rollout **stops** on that token; reward calls a remote large model (GLM) to finish.
- Score = accuracy − λ · cost_ratio + think-format − think-length penalty.

## Files

| Path | Role |
|---|---|
| `reward.py` | `--custom-rm-path examples.llm_offload.reward.reward_func` (batch / `--group-rm`) |
| `convert_offload_dataset.py` | phase2 jsonl → slime prompt jsonl |
| `run_pyrodash4b_offload_1node.sh` | 1-node 8-GPU colocate GRPO launcher |
| `data/offload_grpo_train.jsonl` | converted train set (optional; regenerate anytime) |

## Setup

```bash
# 1) torch_dist
bash examples/coding_agent_rl/convert_pyrodash4b_to_torch_dist.sh

# 2) dataset
python examples/llm_offload/convert_offload_dataset.py \
  --src /workspace/work/spt/pyroDash-training/data/phase2/glm52_hint_8b_answers.jsonl \
  --dst examples/llm_offload/data/offload_grpo_train.jsonl

# 3) remote GLM / OpenAI-compatible API
export DASHSCOPE_API_KEY=...
export DASHSCOPE_BASE_URL=http://host:8000/v1
export DASHSCOPE_MODEL=glm-5.2-fp8
export OFFLOAD_EFFICIENCY_LAMBDA=0.6
```

## Train

```bash
bash examples/llm_offload/run_pyrodash4b_offload_1node.sh
```

Stop token id for `<|llm_offload|>` on PyroDash-4B-SFT-0723 is **248077** (`OFFLOAD_STOP_TOKEN_ID`).
