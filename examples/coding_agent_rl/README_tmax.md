# Tmax 数据与 slime 接入说明

本文说明如何在 slime `coding_agent_rl` 管线中使用 [allenai/tmax-15k-open-instruct](https://huggingface.co/datasets/allenai/tmax-15k-open-instruct)（及同源的 [allenai/TMax-15K](https://huggingface.co/datasets/allenai/TMax-15K)），并与现有 ScaleSWE 数据兼容混训。

相关上游：

- 代码 / 训练配方：[hamishivi/tmax](https://github.com/hamishivi/tmax)
- 论文：[Tmax: A simple recipe for terminal agents](https://arxiv.org/abs/2606.23321)

---

## 1. 数据是什么

Tmax 是一组 **终端 Agent 任务**（约 14.6k）：每题对应一个 Docker 镜像 + 题面 + 程序化 verifier。任务多为合成终端场景（调试、数据分析、运维等），成功标准是 **环境终态**（文件、二进制、答案等），不是「git patch + pytest」。

| HF 数据集 | 用途 |
|-----------|------|
| `allenai/tmax-15k-open-instruct` | **训练用**：`messages` + `env_config.image` + `task-data.tar.gz`（含 `setup.sh` / `tests/test.sh`） |
| `allenai/TMax-15K` | **源数据 / 重建环境**：parquet + `tasks.zip`（Apptainer def 等）；一般不直接当 slime `--prompt-data` |

slime 侧使用 **open-instruct 版** 转换后的 jsonl。

---

## 2. 与 ScaleSWE 对比

ScaleSWE 是 slime `coding_agent_rl` 原有数据：真实/类 SWE 的「修 issue → 交 patch → 跑单测」。Tmax 是「进终端改环境 → 终态过 verifier」。两边用 `metadata.protocol` 区分 grader；用 `metadata.agent` 选择 harness（`claude_code` / `codex` / `pi` / `opencode` / `miniswe`）。除 `codex` 外共用 Anthropic（CC）adapter。

| 维度 | ScaleSWE | Tmax |
|------|----------|------|
| 任务形态 | 修开源仓库 bug/PR | 合成终端任务（调试、ETL、运维…） |
| 典型镜像 | `…/scaleswe-agent:<instance>` 或 baked 镜像 | raw `hamishi740/swerl-tmax-v3:<hash>`，或 baked `…/tmax-agent:<hash>` |
| `workdir` | `/workspace/<repo>` | `/home/user` |
| `protocol` | `scaleswe`（可省略，默认） | `tmax`（必须写出） |
| `agent` | 可选；缺省 `SWE_AGENT` / `claude_code` | 同左 |
| 题面 | issue / problem_statement | 终端任务说明（从 HF `messages` user 抽出） |
| 验题材料 | `remote_env_info.f2p_script`（或 swepro / eval_cmd） | `test_sh`（来自 task-data `tests/test.sh`） |
| 验题时机 | agent 结束后：`git_diff` → **新干净沙箱** apply → 跑测 | agent 结束后：**同一沙箱** 才写入并跑 `test_sh` |
| 成功含义 | 测试退出码 0 / f2p 通过 | 终态断言通过（或 `reward.txt`） |
| 空 patch | 强制 `solved=0` | 不看 git_diff |
| 转换脚本 | [`convert_scaleswe_to_slime.py`](./convert_scaleswe_to_slime.py) | [`convert_tmax_to_slime.py`](./convert_tmax_to_slime.py) |
| 现成 jsonl | `data/swe_train_scaleswe_200_baked.jsonl`、`data/scaleswe_agents_smoke.jsonl`、`data/mixed_agents_bake_smoke_scaleswe*.jsonl` | `data/tmax_smoke_3.jsonl`、`data/tmax_agents_smoke.jsonl`、`data/tmax_train_200.jsonl`、`data/mixed_agents_bake_smoke_tmax*.jsonl` |

ScaleSWE 样例（截断，见 `data/swe_train_scaleswe_200_baked.jsonl`）：

```jsonc
{
  "prompt": [{"role": "user", "content": "# Enforce URI schemes in CORS_ORIGIN_WHITELIST ..."}],
  "label": "adamchainz_django-cors-headers_pr397",
  "metadata": {
    "instance_id": "adamchainz_django-cors-headers_pr397",
    "image": "pyrominddynamics/scaleswe-agent:adamchainz_django-cors-headers_pr397",
    "workdir": "/workspace/django-cors-headers",
    "problem_statement": "...",
    "remote_env_info": { "f2p_script": "import pytest\\n..." }
    // 新转换还会带 "protocol": "scaleswe"
  }
}
```

---

## 3. 原始数据 vs 转换后数据

### 3.1 HF 原始一行（`tmax-15k-open-instruct`）

字段：`messages` / `ground_truth` / `dataset` / `env_config` / `source`。另有附属包 `task-data.tar.gz`，按 `task_id` 含 `setup.sh`、`tests/test.sh`、`instruction.md`。

```jsonc
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful coding assistant. You have access to a bash terminal. ... submit by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    },
    {
      "role": "user",
      "content": "Please solve this task:\\n\\nYou are tasked with resolving a critical concurrency regression in our C-based multithreaded telemetry parser. ...\\n\\n## Recommended Workflow\\n...\\n## Important Rules\\n1. Every response must contain exactly one tool call to `bash`.\\n..."
    }
  ],
  "ground_truth": "task_000000_c19dda5b",
  "dataset": "passthrough",
  "env_config": {
    "env_name": "swerl_vanillux_sandbox",
    "image": "hamishi740/swerl-tmax-v3:618e344e0172",
    "task_id": "task_000000_c19dda5b"
  },
  "source": "tmax_skill_taxonomy"
}
```

同题在 `task-data` 中（不进 parquet，在 tar 里）：

```text
task_000000_c19dda5b/
  instruction.md
  setup.sh              # 建环境：已 bake 进上面的 image
  tests/test.sh         # 终态 verifier：slime 需要带进 jsonl
```

再举一条不同域的原始字段（同一 schema）：

```text
ground_truth / task_id: task_000000_f8baca82
env_config.image:       hamishi740/swerl-tmax-v3:785c91e6228d
user 题面开头:          You are a data analyst. You have been provided with a CSV file...
```

### 3.2 转换后一行（slime jsonl）

仓库样例：[`data/tmax_smoke_3.jsonl`](./data/tmax_smoke_3.jsonl) 第一行（字段齐全，正文截断）：

```jsonc
{
  "prompt": [{
    "role": "user",
    "content": "You are tasked with resolving a critical concurrency regression in our C-based multithreaded telemetry parser. ..."
  }],
  "label": "task_000000_c19dda5b",
  "metadata": {
    "protocol": "tmax",
    "instance_id": "task_000000_c19dda5b",
    "image": "hamishi740/swerl-tmax-v3:618e344e0172",
    "workdir": "/home/user",
    "problem_statement": "You are tasked with resolving a critical concurrency regression ...",
    "test_sh": "#!/bin/bash\\nset -e\\nmkdir -p /logs/verifier\\ncat << 'TEST_EOF' > /tmp/test_final_state.py\\n..."
  }
}
```

对比原始 → 转换的字段映射：

| 原始 | 转换后 | 说明 |
|------|--------|------|
| `messages[user]` 题干 | `prompt` + `problem_statement` | 去掉 `Please solve this task:` 与 vanillux「Recommended Workflow / 只许一次 bash」尾部 |
| `messages[system]` | **丢弃** | 面向 vanillux bash tool；slime 用 Claude Code + `SWE_PROMPT` |
| `env_config.image` | `metadata.image` | 起哪个 Docker |
| `env_config.task_id` / `ground_truth` | `label` / `instance_id` | 任务 ID |
| （固定） | `workdir=/home/user` | Tmax 任务路径约定 |
| （固定） | `protocol=tmax` | 走同箱 grader |
| `task-data/.../tests/test.sh` | `metadata.test_sh` | deferred 验题脚本全文 |
| `task-data/.../setup.sh` | **默认不写** | 已 bake 进镜像；重跑又慢又易冲突 |

---

## 4. 怎么转、为什么这样转

脚本：[`convert_tmax_to_slime.py`](./convert_tmax_to_slime.py)

### 步骤

1. `load_dataset("allenai/tmax-15k-open-instruct")`
2. 解压同 repo 的 `task-data.tar.gz`（或 `--task-data-dir` 指向已解压目录）
3. 按 `task_id` 读 `tests/test.sh` 全文
4. 从 `messages` 取 user 正文，剥 vanillux harness 尾部
5. 写出 slime 三键：`prompt` / `label` / `metadata`

### 为什么这样转

1. **slime 要的是 Sample 行，不是 open-instruct 的 tool-env 行**  
   训练入口是 `--prompt-data` jsonl + `metadata.protocol`，不是 `grpo_fast --tools swerl_vanillux_sandbox`。
2. **Agent 是 Claude Code，不是 vanillux**  
   原始 system /「每步必须一个 bash tool call」会误导 CC；故丢掉 system，并裁掉 user 里的 Recommended Workflow / Important Rules。
3. **只要 `test_sh`，不要默认 `setup_sh`**  
   `setup.sh` 已在镜像构建时执行；每条 rollout 再 apt/造 200 commit 既慢又可能破坏 bake 状态。`test_sh` 必须延期注入，防偷看。
4. **显式 `protocol=tmax`**  
   与 ScaleSWE 混在同一 jsonl 时，靠该字段分支 grader（缺省仍当 scaleswe，避免旧数据误伤）。

### 命令示例

```bash
cd slime/

python examples/coding_agent_rl/convert_tmax_to_slime.py \
  --dst examples/coding_agent_rl/data/tmax_smoke_3.jsonl \
  --limit 3

python examples/coding_agent_rl/convert_tmax_to_slime.py \
  --dst examples/coding_agent_rl/data/tmax_train_200.jsonl \
  --limit 200

# 全量（首次解压 task-data 较慢）
python examples/coding_agent_rl/convert_tmax_to_slime.py \
  --dst examples/coding_agent_rl/data/tmax_train_full.jsonl

# 复用已解压目录（推荐）
python examples/coding_agent_rl/convert_tmax_to_slime.py \
  --task-data-dir /path/to/task-data.tar.gz.extracted \
  --dst examples/coding_agent_rl/data/tmax_train_200.jsonl \
  --limit 200
```

HF cache 中常见解压路径：

```text
~/.cache/huggingface/hub/datasets--allenai--tmax-15k-open-instruct/snapshots/<rev>/task-data.tar.gz.extracted/
```

仓库现成文件：[`data/tmax_smoke_3.jsonl`](./data/tmax_smoke_3.jsonl)、[`data/tmax_train_200.jsonl`](./data/tmax_train_200.jsonl)。

若 jsonl 手动带上 `metadata.setup_sh`，`prepare_workspace` 仍会执行（兜底）。

---

## 5. 镜像使用

### 镜像是什么

- 名称形如：`hamishi740/swerl-tmax-v3:<content-hash>`（raw）或 `pyrominddynamics/tmax-agent:<content-hash>`（baked）
- **每题一个 tag**（约 1.4 万个不同镜像）；raw 镜像内是任务环境（代码、依赖、fixture）
- **baked**（`docker_build/bake_tmax_agent_images.py`）额外预装 Node 22 + `claude` / `opencode` / `pi` / `mini`；**不会**把 `test_sh` / `/tests` 打进镜像
- 来源字段：HF 行的 `env_config.image` → 转换后的 `metadata.image`；bake 后原镜像写在 `metadata.docker_image`，`cli_prebaked=True`

### slime 如何起容器

```bash
export SLIME_AGENT_SANDBOX_BACKEND=docker   # 或 e2b
export SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY=image
# docker 时还需 ADAPTER_PUBLIC_HOST 等，使容器能回调 Anthropic adapter
```

流程：按 `metadata.image` `docker pull` → boot → harness `install_cli`（raw 会装 Node + CLI；`cli_prebaked` 镜像在二进制可用时跳过）→ 跑 harness。

### 运维注意

1. **Docker Hub 拉取额度**：全量 RL 前需账号 / mirror / 缓存，否则拉镜像会成为瓶颈。
2. **CLI 安装**：raw Tmax 每次 boot 装 CLI；大批量优先用 `tmax-agent` bake（见 [`docker_build/README.md`](./docker_build/README.md)）。
3. **特权 / DinD**：部分 Tmax 题会起容器（如 neo4j）；若失败需检查 Docker 能力（`--privileged` 等，按环境调整）。
4. **workdir**：默认 `/home/user`，须在镜像中存在（一般已有 `user` 家目录）。

### 手动进镜像调试

```bash
docker run --rm -it --entrypoint bash hamishi740/swerl-tmax-v3:618e344e0172
# 按 PROBLEM / 题面操作；验题需自行拷贝 test.sh：
# docker cp .../tests/test.sh <cid>:/tests/ && docker exec <cid> bash /tests/test.sh
```

---

## 6. Reward（验题）

### 时机（deferred tests）

与官方 open-instruct 一致：**agent 解题期间容器内没有 tests**，防止偷看 verifier。

```text
boot 镜像（无 tests）
  → prepare_workspace（写 PROBLEM_STATEMENT.md；可选 setup_sh）
  → Claude Code harness.run
  → grade_tmax_inplace：write_file(test_sh) + bash   ← 此时才拷测试
  → 关沙箱
```

官方用 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 触发验题；slime + CC 黑盒用 **harness 进程退出** 作为触发点。

### 计分规则

实现：[`swe.grade_tmax_inplace`](./swe.py)

1. 将 `metadata.test_sh` 写到容器 `/workspace/__tmax_test__.sh`
2. `bash` 执行（timeout 由 `SWE_EVAL_TIMEOUT_SEC` 等控制）
3. **优先**读 `/logs/verifier/reward.txt`（若 verifier 写入了 0~1 浮点）
4. 否则：`exit 0 → reward 1.0`，非 0 → `0.0`

Train 路径上若开启 offload，还会在 `solved` 之上做 cost / help_seeking shaping（与 ScaleSWE 相同，见主 README）。

### 与 ScaleSWE 的关键差异

- Tmax **不**走 `git_diff` + 新沙箱 `run_evaluation`
- Tmax **不**因 empty patch 强制 `solved=0`（终态往往不在 git 里）

---

## 7. 接入 slime

### 协议选择

[`generate.py`](./generate.py) 按样本选择协议：

```text
metadata.protocol 优先；缺失时用 SWE_TRAIN_PROTOCOL / SWE_EVAL_PROTOCOL（默认 scaleswe）
```

| `protocol` | 路径 |
|------------|------|
| `tmax` | CC → 同箱 `grade_tmax_inplace` |
| `scaleswe` / 缺省 | CC → `git_diff` → 新箱 apply + f2p |
| `swebench` | 官方 swebench grader |

实现文件：

- 任务层：[`swe.py`](./swe.py)（`PROTOCOL_TMAX`）
- 编排：[`generate.py`](./generate.py)
- 转换：[`convert_tmax_to_slime.py`](./convert_tmax_to_slime.py)

### 仅 Tmax 推理 / 验题（不训）

```bash
# 需 DASHSCOPE_*、tarballs、ADAPTER_PUBLIC_HOST
bash examples/coding_agent_rl/run_infer_cc_tmax_traj.sh
# 默认 data/tmax_smoke_3.jsonl，本地 docker + inplace eval
```

脚本：[`run_infer_cc_tmax_traj.sh`](./run_infer_cc_tmax_traj.sh)、[`infer_cc_tmax_traj.py`](./infer_cc_tmax_traj.py)

### 训练（纯 Tmax）

```bash
export PROMPT_DATA=$PWD/examples/coding_agent_rl/data/tmax_train_200.jsonl
export SLIME_AGENT_SANDBOX_BACKEND=docker
export SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY=image
# 其余与现有 docker async launcher 相同（Node/CC tarball、ADAPTER_PUBLIC_HOST…）
bash examples/coding_agent_rl/run_qwen35_4b_swe_1node_docker_async.sh
```

行内已有 `"protocol": "tmax"`，无需改 `SWE_TRAIN_PROTOCOL`（它只是缺省 fallback）。

### 与 ScaleSWE 混训

```bash
cat examples/coding_agent_rl/data/swe_train_scaleswe_200_baked.jsonl \
    examples/coding_agent_rl/data/tmax_train_200.jsonl \
  | shuf > examples/coding_agent_rl/data/mixed_scaleswe_tmax_400.jsonl

PROMPT_DATA=$PWD/examples/coding_agent_rl/data/mixed_scaleswe_tmax_400.jsonl \
  bash examples/coding_agent_rl/run_qwen35_4b_swe_1node_docker_async.sh
```

同一 job、同一 Claude Code agent；仅 grader / workdir / 镜像按行分支。

### 单测

```bash
python -m pytest tests/test_tmax_protocol.py -q
```

---

## 8. 端到端流程（一张图）

```text
HF tmax-15k-open-instruct
        │ convert_tmax_to_slime.py
        ▼
jsonl (protocol=tmax, image, test_sh, workdir=/home/user)
        │ --prompt-data / PROMPT_DATA
        ▼
slime generate()
  boot image → install CC → prepare → Claude Code
        │
        ▼ (agent 退出后)
  write test_sh → bash → reward 0/1
        │
        ▼
  GRPO / 日志 / rollout dump
```

---

## 9. 常见问题

**Q: 为什么不用官方 vanillux（只 bash）？**  
A: 当前 slime 接入选择与 ScaleSWE 共用 Claude Code，降低分叉。官方 vanillux 在 open-instruct `swerl_vanillux_sandbox`；行为与 CC 不完全一致。

**Q: `TMax-15K` 和 `tmax-15k-open-instruct` 用哪个？**  
A: slime 转换 / 训练用 **open-instruct** 版。`TMax-15K` 适合分析技能轴或自己 rebuild 容器。

**Q: reward 一直是 0？**  
A: 检查镜像是否拉对、workdir 是否 `/home/user`、agent 是否真正改到 verifier 检查的路径、`test_sh` 是否在 grade 阶段写入（解题中不应出现 `/tests`）、以及 Docker Hub / 权限问题。

**Q: 能否把 setup 也塞进 jsonl？**  
A: 可以手动加 `metadata.setup_sh`；或扩展转换脚本。默认不建议对已 bake 镜像每步重跑。
