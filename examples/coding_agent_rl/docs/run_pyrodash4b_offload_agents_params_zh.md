# `run_pyrodash4b_swe_offload_1node_docker_async_agents.sh` 参数说明

PyroDash-4B + mid-turn LLM offload 的 **mixed-agent async Docker** 启动器。脚本设置环境变量后 `exec` 到 `run_qwen35_4b_swe_1node_docker_async.sh`。下列变量均可在命令前 `export` 覆盖默认值。

开训示例：

```bash
export NCCL_P2P_DISABLE=1
# OFFLOAD_SEEK_ALPHA=0.15 GIGPO_W=0.5 PROMPT_DATA=.../xxx.jsonl \
bash examples/coding_agent_rl/run_pyrodash4b_swe_offload_1node_docker_async_agents.sh
```

日志：`runs/<EXP_TAG>_*/run.log`。

---

## 流程概览

Actor 发出 `<|llm_offload|>N<|/llm_offload|>` → Adapter 调用远程 LLM（N 控制 thinking：`0`=off，`1–3`=low，`4–6`=high，`7–9`=max）→ 续写写回 trajectory → 默认用 **help_seeking** 奖励塑形。

空 patch 不算解出。

---

## 1. Agent / 轨迹预算

| 变量 | 默认 | 含义 |
|------|------|------|
| `ADAPTER_MAX_TURNS_PER_SID` | `150` | 单 session 最大 turn 数 |
| `SWE_AGENT_TIME_BUDGET_SEC` | `600` | 单 agent 墙钟时间上限（秒） |
| `SAVE_INTERVAL` | `20` | Megatron 每隔多少 step 存 ckpt |

---

## 2. Offload 开关与远程 LLM

| 变量 | 默认 | 含义 |
|------|------|------|
| `SLIME_AGENT_OFFLOAD` | `1` | 开启 mid-turn offload |
| `DASHSCOPE_BASE_URL` | 脚本内配置 | OpenAI 兼容端点（需含 `/v1`） |
| `DASHSCOPE_API_KEY` | 脚本内配置 | 鉴权；也可用 `OPENAI_API_KEY` |
| `DASHSCOPE_MODEL` | `deepseek-v4-flash-0731` | 远程模型名 |
| `OFFLOAD_MAX_TOKENS` | `32768` | 单次 offload 最大生成 token |
| `OFFLOAD_STOP_TOKEN_ID` | `248078` | `<\|/llm_offload\|>` close token（PyroDash vocab） |
| `ROLLOUT_STOP_TOKEN_IDS` | `248046 248044 248078` | rollout 停止 token 列表 |
| `SLIME_OFFLOAD_EMBED_IN_TRAJECTORY` | `1` | 远程续写 embed 进 `Sample.tokens`（`loss_mask=0`） |
| `SLIME_OFFLOAD_EMBED_MAX_TOKENS` | （未设） | 可选：embed 进轨迹的 token 上限 |
| `SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS` | `160000` | fork / rewrite-merge 响应 token 上限 |

未设置 `DASHSCOPE_API_KEY` / `OPENAI_API_KEY` 时脚本会警告；运行期 offload 调用会失败。

---

## 3. Reward（help_seeking）

| 变量 | 默认 | 含义 |
|------|------|------|
| `OFFLOAD_REWARD_MODE` | `help_seeking` | 奖励模式；`cost_aware` 恢复「失败 → 0」 |
| `OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG` | `1` | 组 shaping 发 α；组内有人「不 offload 就解出」时用 `SEEK_SOLO_SCALE` 打折，不是清零 |
| `OFFLOAD_SEEK_SOLO_SCALE` | `0.3` | 有裸解兄弟时：失败求助 α × 此系数（`0`=旧逻辑整组清零，`1`=不打折） |
| `OFFLOAD_EFFICIENCY_LAMBDA` | `0.05` | 解出时扣 `λ * cost_ratio` |
| `OFFLOAD_SEEK_ALPHA` | `0.1` | 未解出但合法 in-think offload 的部分分（整条固定给，不按轮数平均） |
| `OFFLOAD_SEEK_EMPTY_SCALE` | `0.5` | empty patch 时 α 缩放 |
| `OFFLOAD_UNIQUE_SOLVER_BONUS` | `0.15` | 组内唯一解出者的 bonus |
| `OFFLOAD_TURN_PENALTY_COEF` | `0.15` | 解出后对超长轨迹的惩罚系数 |
| `OFFLOAD_TURN_PENALTY_REF` | `50` | turn ≤ REF 不罚；超出按 `(n−ref)/ref` |
| `OFFLOAD_SOLVED_REWARD_FLOOR` | `0.3` | 解出奖励下限 |
| `OFFLOAD_SEEK_BUDGET_TURN_K` | `0` | `>0` 时 budget=`n_turns//K`（默认关，避免鼓励拖长） |
| `OFFLOAD_SEEK_BUDGET_DECAY` | `0.5` | 超 budget 时 α 衰减 |
| `OFFLOAD_SEEK_OVERAGE_PENALTY` | `0` | 超 budget 额外惩罚 |
| `OFFLOAD_SEEK_BUDGET` | （未设） | 可选固定 seek 次数上限；与 turn budget 同时设时取 `min` |

**解出奖励（简化）：**

```text
r = 1 - λ·cost_ratio - coef·max(0, n_turns − ref) / ref
r = max(floor, r)
```

实现细节见 `examples/coding_agent_rl/offload.py`（`shape_group_help_seeking_rewards`）。

---

## 4. GiGPO advantage

| 变量 | 默认 | 含义 |
|------|------|------|
| `GIGPO_W` | `1.0` | step advantage 权重：`A_t = A_E + w·A_S + …` |
| `GIGPO_GAMMA` | `0.95` | 折扣因子 |
| `GIGPO_TURN_RESIDUAL_W` | `1.0` | turn residual `(r_t − mean r)` 权重；`0` 关掉 |
| `CUSTOM_ADVANTAGE_FUNCTION_PATH` | （下游默认 gigpo） | 自定义 advantage 入口 |

更多方案说明见 [gigpo_scheme_readme.md](./gigpo_scheme_readme.md)。

---

## 5. 模型 / 数据 / 混合 Agent

| 变量 | 默认 | 含义 |
|------|------|------|
| `HF_CHECKPOINT` | 脚本内路径 | SGLang 加载的 HF 权重（BF16） |
| `REF_MODEL_PATH` | 脚本内路径 | Megatron 训练 / 参考权重（torch_dist，词表 pad 到 248320） |
| `EXP_TAG` | `agent_offload_pyrodash4b_sft_entropy_docker_async_turn` | 实验名 / `runs/` 目录前缀 |
| `SGLANG_KV_CACHE_DTYPE` | `fp8_e4m3` | rollout KV 缓存精度（权重仍 BF16；可加长 agent context） |
| `PROMPT_DATA` | `data/release/mixed_reward1_agents_baked.jsonl` | 训练 jsonl（baked 多 agent 镜像） |
| `SLIME_AGENT_CODEX_TARBALL` | `tarballs/openai-codex-local.tgz` | Codex agent 包 |
| `SLIME_AGENT_PI_TARBALL` | `tarballs/pi-coding-agent-local.tgz` | pi agent 包 |
| `SLIME_AGENT_OPENCODE_TARBALL` | `tarballs/opencode-ai-local-linux-x64.tgz` | opencode 包 |
| `SLIME_AGENT_MINISWE_WHEEL` | `tarballs/miniswe-wheels` | miniswe wheel 目录 |

`PROMPT_DATA` 为 mixed `*_agents.jsonl` 时，上述四个 agent 包必须存在，否则脚本直接 `exit 1`。

单 agent / 原始 base 镜像可改回例如：

```bash
PROMPT_DATA=examples/coding_agent_rl/data/swe_train_scaleswe_200_baked.jsonl \
  bash examples/coding_agent_rl/run_pyrodash4b_swe_offload_1node_docker_async_agents.sh
```

---

## 6. Debug：只训内存快照（默认关）

| 变量 | 默认 | 含义 |
|------|------|------|
| `DEBUG_TRAIN_MEM` | `0` | `1`：只训、不启 SGLang/Docker，用 dump 做 OOM 分析 |
| `LOAD_DEBUG_ROLLOUT_DATA` | mem 模式默认 `runs/debug_rollout_32/.../rollout_{rollout_id}.pt` | dump 路径模板 |
| `RECORD_MEMORY_HISTORY` | `1`（mem） | 记录 CUDA memory history |
| `MEMORY_SNAPSHOT_NUM_STEPS` | `4` | 训多少 step 后 dump |
| `NUM_ROLLOUT` / `NUM_STEPS_PER_ROLLOUT` | `4` / `1` | mem 模式步数 |
| `MICRO_BATCH_SIZE` | `4` | micro batch |
| `ROLLOUT_BATCH_SIZE` / `N_SAMPLES_PER_PROMPT` / `GLOBAL_BATCH_SIZE` | `4` / `8` / `32` | 匹配 32-sample dump |
| `NUM_GPUS` / `ACTOR_GPUS` / `ROLLOUT_GPUS` | `8` / `6` / `0` | 无 rollout engine |
| `TP_SIZE` / `CP_SIZE` | `1` / `6` | 训练并行 |

用法：

```bash
DEBUG_TRAIN_MEM=1 bash examples/coding_agent_rl/run_pyrodash4b_swe_offload_1node_docker_async_agents.sh
DEBUG_TRAIN_MEM=0 bash ...   # 全量 async RL（当前默认）
```

---

## 7. 下游 async Docker 继承项

本脚本未全部显式 export，但 `run_qwen35_4b_swe_1node_docker_async.sh` 提供默认：

| 变量 | 典型默认 | 含义 |
|------|----------|------|
| `ROLLOUT_BATCH_SIZE` | `4` | 每步 prompt 批大小 |
| `N_SAMPLES_PER_PROMPT` | `8` | 每 prompt 采样数 |
| `GLOBAL_BATCH_SIZE` | `RBS × N` | 须能被 DP 整除 |
| `NUM_ROLLOUT` | `100` | rollout 轮数 |
| `SWE_BOOT_CONCURRENCY` | `32` | sandbox 启动并发 |
| `NUM_GPUS` / `ACTOR_GPUS` / `ROLLOUT_GPUS` | `8` / `6` / `2` | GPU 划分 |
| `TP_SIZE` / `CP_SIZE` | `1` / `6` | 并行度 |
| `QWEN_GDN_BACKEND` | `flashqla` | GDN 后端 |
| `SWE_EVAL_TIMEOUT_SEC` | `300` | 评测超时 |
| `ADAPTER_PUBLIC_HOST` | pod IP | sandbox 可达的 adapter 地址（不可为 `127.0.0.1`） |
| `ADAPTER_BIND_HOST` / `ADAPTER_PORT` | `0.0.0.0` / `18001` | adapter 监听 |
| `SLIME_AGENT_SANDBOX_BACKEND` | `docker` | sandbox 后端 |
| `SLIME_AGENT_DOCKER_RUN_TIMEOUT_SEC` | `300` | `docker run` 超时 |

全量 RL 启动前会清理残留：`docker ps -aq --filter name=slime-sb- | xargs -r docker rm -f`。

---

## 8. 运行时杂项

| 变量 | 默认 | 含义 |
|------|------|------|
| `NCCL_DEBUG` | `INFO` | NCCL 日志级别 |
| `NCCL_CUMEM_ENABLE` | `0` | 关闭 NCCL cuMem |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 缓解显存碎片 |
| `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` | `1` | 允许 TF32 cuBLAS |
| `SLIME_DIR` | 仓库根 | slime 根目录 |

---

## 前置条件

1. PyroDash HF → torch_dist：`bash examples/coding_agent_rl/scripts/convert_pyrodash4b_to_torch_dist.sh`
2. `DASHSCOPE_*` 指向可用的 OpenAI 兼容 deepseek 服务
3. Docker daemon + 可路由 pod IP（`ADAPTER_PUBLIC_HOST`）
4. mixed agent 时 `tarballs/` 下四个包齐全

总览与排障见 [TRAINING_README_zh.md](./TRAINING_README_zh.md)。
