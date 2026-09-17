# Reward=1 轨迹熵标注

筛 `reward=1` → 读每个样本 **最后一个** `requests/req_*.json`（只读）→
对每条 assistant 用前缀打 vLLM → 在 **新目录** 写出标注文件（不改原 runs）。

## 数据源

| 集合 | 路径 |
|------|------|
| SWE (glm) | `runs/infer_cc_glm_20260728_141118` |
| SWE (dsv4flash) | `runs/infer_cc_dsv4flash_20260806_141118` |
| TMax | `runs/infer_cc_tmax_20260807_162106` |

约 5.4k 条 reward=1，vLLM 调用数 ≈ 各样本 last-req 里的 assistant 数之和（约 15 万）。

## 步骤

```bash
# 1) 起 vLLM
bash examples/coding_agent_rl/sft/launch_vllm_pyrodash4b_sft0902.sh

# 2) dry-run
DRY_RUN=1 bash examples/coding_agent_rl/sft/run_annotate_reward1_entropy.sh

# 3) smoke
RUN_DIRS=/workspace/work/spt/slime/runs/infer_cc_dsv4flash_20260806_141118 \
  LIMIT_SAMPLES=1 SAMPLE_CONCURRENCY=1 ASSISTANT_CONCURRENCY=4 \
  bash examples/coding_agent_rl/sft/run_annotate_reward1_entropy.sh

# 4) 全量（已有 req_*_entropy.json 则 skip）
bash examples/coding_agent_rl/sft/run_annotate_reward1_entropy.sh
```

## 逻辑

对 last req 的完整对话（`messages` + 最终 `response` 当作最后一条 assistant）：

- assistant 在 index `i` → 用前缀请求 vLLM（logprobs + thinking），**生成完整 response**
- 熵只算 **reasoning 段**：用 `message.reasoning` 对齐 `logprobs.content` 前缀
- `vllm_response` 保存完整 Qwen 回复：`content` / `reasoning` / `tool_calls` / `finish_reason` / `usage`
- 标注写到 **新文件**；不修改原 `runs/.../requests/req_*.json`
- 默认 **不用** `</think>` stop（依赖 `--reasoning-parser qwen3`）；若要手动加：`STOP_STR='...'`

## 输出

```
trajectories_entropy/                          # --out-dir，与 runs/ 分离
  inventory.json / summary.json / sft_messages.jsonl
  <run>/<sample>/
    req_<N>_entropy.json       # 新文件：与原 req_<N>.json 同结构 + 标注
    sft_messages.json          # 展开后的 messages（含最终 response）便于 SFT
```
