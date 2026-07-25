# Qwen3.5-4B + 公网 E2B Coding-Agent RL（1 机 8 卡）

本文说明本分支相对上游 coding-agent RL 的改动，以及如何用 **公网 e2b.dev** 跑通 Qwen3.5-4B 冒烟 / 训练。

通用设计（adapter / harness / 数据 schema）仍见同目录 [README.md](./README.md)。

---

## 背景：公网 E2B 和内部 gateway 的差别

| | 内部 E2B-compatible gateway | 公网 [e2b.dev](https://e2b.dev) |
|---|---|---|
| 选环境 | `AsyncSandbox.create(metadata={image: "docker/..."})` | `AsyncSandbox.create(template="alias")` |
| Docker 镜像 | gateway 按 metadata 路由 | **不会**读 metadata 拉任意镜像 |
| 前置步骤 | 无 | 先把 Docker **build 成 E2B template** |
| Adapter 回连 | 集群内网 IP 即可 | 必须公网可达（反代 / tunnel） |
| 并发 | 通常很高 | 账号限额常见 **20** 个并发 sandbox |

---

## Diff 总览

### 已修改（相对 `origin/main`）

```
 build_conda.sh                               | 环境 pin（torchvision / scipy / libnuma / libcuda stub）
 examples/coding_agent_rl/generate.py         | ADAPTER_PUBLIC_URL（完整 HTTPS URL）
 slime/agent/sandbox.py                       | SLIME_AGENT_E2B_USE_TEMPLATE → create(template=...)
 slime/backends/sglang_utils/sglang_engine.py | 每 engine 独立 TRITON_CACHE_DIR
 slime/ray/train_actor.py                     | 每 Megatron rank 独立 TRITON_CACHE_DIR
```

### 新增文件

| 路径 | 作用 |
|---|---|
| `run_qwen35_4b_swe_1node.sh` | 1 机 8 卡训练入口；template 模式自动降并发 |
| `smoke_public_e2b.py` | Docker → E2B template build + `/workspace` 探测（**训练不自动调用**） |
| `start_adapter_tunnel.sh` | Cloudflare quick tunnel，暴露本地 `:18001` adapter |
| `convert_scaleswe_to_slime.py` | ScaleSWE → slime JSONL |
| `data/swe_smoke_preliz_e2b.jsonl` | 单样本冒烟（`image` = template 名） |
| `data/swe_train_scaleswe_200.jsonl` | 200 条训练数据（`image` 仍是 Docker 名，全量公网需先批量 build template） |
| `tarballs/` | Node 22 + Claude Code 本地包 |

### 关键逻辑 diff（摘要）

**1. `slime/agent/sandbox.py` — 公网用 template**

```python
use_template = SLIME_AGENT_E2B_USE_TEMPLATE in ("1", "true", "yes")
if use_template:
    AsyncSandbox.create(template=self.image, timeout=...)
else:
    AsyncSandbox.create(timeout=..., metadata={image_key: self.image, ...})
```

`USE_TEMPLATE=1` 时，样本 `metadata.image` 必须是 **E2B template 名**（如 `scaleswe-preliz-pr249`），不是 `aweaiteam/scaleswe:...`。

**2. `generate.py` — adapter 公网 URL**

```python
if ADAPTER_PUBLIC_URL:
    adapter_url = ADAPTER_PUBLIC_URL          # e.g. https://xxx.trycloudflare.com
else:
    adapter_url = f"http://{ADAPTER_PUBLIC_HOST}:{port}"
```

**3. Triton cache 隔离（避免多进程抢写）**

- SGLang：`~/.triton/cache/sglang_engine_{rank}`
- Megatron：`~/.triton/cache/megatron_rank_{rank}`

可用 `SLIME_TRITON_CACHE_BASE` 改根目录。

---

## Template 是怎么创建的

公网 E2B 上的 template = 账号里的一份可启动快照，**不是仓库里的文件**。

```bash
export E2B_API_KEY=e2b_...   # https://e2b.dev 真 key

# 等价于：
#   Template().from_image("aweaiteam/scaleswe:arviz-devs_preliz_pr249")
#   Template.build(..., name="scaleswe-preliz-pr249")
# 再 create(template="scaleswe-preliz-pr249") 并检查 /workspace/preliz
python examples/coding_agent_rl/smoke_public_e2b.py

# 已 build 过可跳过：
python examples/coding_agent_rl/smoke_public_e2b.py --skip-build
```

冒烟数据里的对应关系：

```jsonc
// data/swe_smoke_preliz_e2b.jsonl
{
  "metadata": {
    "image": "scaleswe-preliz-pr249",                          // ← E2B template 名
    "docker_image": "aweaiteam/scaleswe:arviz-devs_preliz_pr249", // 仅记录
    "workdir": "/workspace/preliz"
  }
}
```

---

## 运行（公网 E2B 冒烟）

在 **长寿命 shell / tmux** 里跑（不要用短命 nohup 包一层，Ray 子进程会被带走）。

### 0. 环境

```bash
# 按仓库常规方式准备 slime conda（可用 build_conda.sh）
# 模型路径按本机修改：
#   HF:  /workspace/models/Qwen/Qwen3.5-4B
#   dist:/workspace/models/Qwen/Qwen3.5-4B_torch_dist
```

### 1. Build template（首次）

```bash
cd /path/to/slime
export E2B_API_KEY=e2b_...
python examples/coding_agent_rl/smoke_public_e2b.py
# 期望结尾: WORKDIR_OK / PASS
```

### 2. 起 Cloudflare 反代（沙箱回连 adapter）

Adapter 在训练启动后才监听 `127.0.0.1:18001`；tunnel 可先起。  
`GET /` 常 404，健康检查看 **`/v1/models` → 200**。

```bash
bash examples/coding_agent_rl/start_adapter_tunnel.sh
# 输出例如：
#   ADAPTER_PUBLIC_URL=https://xxxx.trycloudflare.com
#   export ADAPTER_PUBLIC_URL=...

export ADAPTER_PUBLIC_URL=https://xxxx.trycloudflare.com
```

训练开始后可再跑一次 tunnel 脚本确认 `public probe .../v1/models -> HTTP 200`。

### 3. 清 Triton 坏缓存（若曾 FileNotFoundError）

```bash
rm -rf ~/.triton/cache
```

### 4. 启动训练

```bash
export E2B_API_KEY=e2b_...
export SLIME_AGENT_E2B_USE_TEMPLATE=1
export ADAPTER_PUBLIC_URL=https://xxxx.trycloudflare.com
export PROMPT_DATA="$(pwd)/examples/coding_agent_rl/data/swe_smoke_preliz_e2b.jsonl"

bash examples/coding_agent_rl/run_qwen35_4b_swe_1node.sh
```

`USE_TEMPLATE=1` 时脚本默认：`rollout-batch-size=1`、`n-samples-per-prompt=2`、`SWE_BOOT_CONCURRENCY=4`（避开公网约 20 并发上限）。可用环境变量覆盖。

### 5. 成功时日志里应看到

```text
adapter=https://xxxx.trycloudflare.com
[coding_agent_rl] ... reward=... applied=... agent_exit_code=... segments=...
Update weights: 100%|...
Job 'raysubmit_...' succeeded
```

可忽略：SGLang `AttributeError: '_IncludedRouter' object has no attribute 'path'`（查 `/v1/loads` 的已知噪音，不阻断主路径）。

---

## 关键环境变量

| 变量 | 公网冒烟建议 | 说明 |
|---|---|---|
| `E2B_API_KEY` | 真 key | 公网 e2b.dev |
| `SLIME_AGENT_E2B_USE_TEMPLATE` | `1` | `create(template=metadata.image)` |
| `ADAPTER_PUBLIC_URL` | Cloudflare URL | 优先于 HOST:PORT |
| `ADAPTER_PUBLIC_HOST` | （可留默认） | 仅内网 / 自建 gateway |
| `PROMPT_DATA` | `.../swe_smoke_preliz_e2b.jsonl` | 单样本；全量需先改 image→template |
| `SLIME_AGENT_NODE_TARBALL` / `SLIME_AGENT_CC_TARBALL` | 脚本默认 `tarballs/` | 上传进 sandbox |
| `SWE_BOOT_CONCURRENCY` | template 模式默认 4 | 同时 boot 上限 |
| `ROLLOUT_BATCH_SIZE` / `N_SAMPLES_PER_PROMPT` | 默认 1 / 2 | 控制并发 sandbox |

---

## 常见问题

1. **`chown: cannot access '/workspace/...'`**  
   起的是默认 base template，不是 ScaleSWE。检查 `USE_TEMPLATE=1` 且 `metadata.image` 是已 build 的 template 名。

2. **`Rate limit ... concurrent E2B sandboxes (20)`**  
   降并发，或在 dashboard 杀掉残留 sandbox。

3. **Rollout 长时间无日志，adapter 仍是 `http://10.x.x.x:18001`**  
   未设置 `ADAPTER_PUBLIC_URL`；公网沙箱连不上 Pod IP。

4. **Tunnel 探测 `/` 404**  
   正常；看 `/v1/models`。

5. **Triton `*.cubin` / `*.ptx` FileNotFoundError**  
   清 `~/.triton/cache` 后重跑（代码已按 rank/engine 隔离）。

---

## 内部 gateway（对照）

不设 `SLIME_AGENT_E2B_USE_TEMPLATE`，保持：

```bash
export SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY=image
export ADAPTER_PUBLIC_HOST=<沙箱可达的内网 IP>
# PROMPT_DATA 里 metadata.image 用 Docker 名 aweaiteam/scaleswe:...
bash examples/coding_agent_rl/run_qwen35_4b_swe_1node.sh
# 或 8 机脚本 run_qwen36_35b_a3b_swe_8nodes.sh
```
