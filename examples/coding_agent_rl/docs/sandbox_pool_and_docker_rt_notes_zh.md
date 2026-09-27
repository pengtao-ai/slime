# Sandbox Pool 与 Docker-RT / Pyromind-SDK 问题记录

> 日期：2026-09-25  
> 范围：coding-agent RL 本地 Docker（docker-rt）沙箱预热，以及切换/调试 `pyromind-sdk` vs `misc_spt` 过程中踩到的坑。

---

## 1. Slime 侧：Sandbox 改动摘要

### 1.1 目标

Agent rollout 里 `docker run` + `install_cli` 很慢，且异步训练希望 **当前 step 跑样本时，预热下一步的容器**。因此加了 sandbox pool，让 `boot_agent_sandbox` 尽量 `acquire HIT`，少走 cold boot。

### 1.2 新增 / 修改的主要文件

| 路径 | 作用 |
|------|------|
| `examples/coding_agent_rl/sandbox_pool.py` | `SandboxPool`：按 `(image, harness)` 预热；同 image 首次 warm 串行，成功后并行；受 `SWE_BOOT_CONCURRENCY` 限制 |
| `examples/coding_agent_rl/rollout_with_pool.py` | `--rollout-function-path`：`get_samples` 同时 `reserve_samples` 下一批并 `schedule_prefetch`；默认 `SANDBOX_POOL_MAX = 2 × rollout_batch × n_samples` |
| `slime/rollout/data_source.py` | `reserve_samples`：预取下一批 prompt groups，与 `get_samples` 消费顺序一致 |
| `slime/agent/sandbox.py` | `DockerSandbox.from_container` / `_wait_until_running`（`docker exec true` 等到 ready）；`SLIME_AGENT_DOCKER_READY_TIMEOUT_SEC`（默认 180） |
| `examples/coding_agent_rl/generate.py` | `boot_agent_sandbox`：优先 `pool.acquire`，超时再 cold boot |
| `tests/test_sandbox_pool.py` | 池命中、同 image 串行 warm 等单测 |

### 1.3 环境变量

| 变量 | 含义 | 典型值 |
|------|------|--------|
| `SANDBOX_POOL` | 开关（默认开） | `1` |
| `SANDBOX_POOL_MAX` | 在池 + 使用中上限 | `2 × batch × n_samples`（如 16×8 → 256） |
| `SANDBOX_POOL_ACQUIRE_TIMEOUT_SEC` | acquire 等待 | 默认 `600`；`0`=一直等 |
| `SWE_BOOT_CONCURRENCY` | 全局 boot 并发 | 如 `32` |
| `SLIME_AGENT_DOCKER_READY_TIMEOUT_SEC` | run 后等 exec-ready | 默认 `180` |
| `SLIME_AGENT_DOCKER_RUN_TIMEOUT_SEC` | `docker run` 超时 | 默认 `300` |

启动脚本：`run_pyrodash4b_swe_offload_1node_docker_async_agents.sh`（`SANDBOX_POOL=1`，经 `run_qwen35_4b_swe_1node_docker_async.sh` → `train_async.py`）。

### 1.4 行为要点

1. **命中**：`acquire` 取出已 `run` + `install_cli` 的容器 → `boot_sandbox` ≈ 0。  
2. **用完销毁**：`release` 不把脏容器放回池（避免串味）。  
3. **预取失败不杀训练**：warm 失败只打 warning；真正跑 sample 时还会 on-demand warm / cold boot。  
4. **Prefetch key**：`(image, harness)`，靠 `reserve_samples` 提前知道下一批镜像，避免预热错图。

### 1.5 线上观察到的现象（与 pool 相关）

- 冷启动第一轮 rollout 仍很慢（无池）；稳态后大量 `pool_hit`，`boot_sandbox≈0`。  
- 高并发时出现 `warm failed … not exec-ready within 180s (Container is not running)`：docker-rt/k8s 调度或起 Pod 慢，**不是镜像坏了**；同镜像稍后常又能 warm 成功。  
- `prepare_workspace` 仍约 **20s/样本**：baked 数据无 `pre_commands`，耗时几乎全在 **`write_file` → `docker cp`**（见下文 docker-rt archive）。

### 1.6 训练 / 异步说明（易混淆）

- 脚本名里的 async = `train_async.py`（`train(N)` 与 `generate(N+1)` 一层重叠）+ agent 内异步，**不是** fully-async 持续生成。  
- `--update-weights-interval 1` 每步要等 in-flight generate 完再 sync weight。  
- 本场景 **rollout 墙钟（agent/环境）>> train**，推理卡常打不满；加大 `ROLLOUT_BATCH_SIZE` 可拉长 train（更多 microbatch），峰值显存仍由 `max-tokens-per-gpu` 卡住（`micro-batch-size=1` + dynamic batch）。

---

## 2. Docker-RT：`pyromind-sdk` vs `misc_spt`

路径对照：

| 树 | 路径 | 后端 |
|----|------|------|
| **pyromind-sdk**（原先常用） | `/workspace/sh-work/docker-rt/pyromind-sdk/pyromind_sdk/docker_rt/` | **k8s-middleware**（`PyromindSDK` HTTP） |
| **misc_spt**（本次切换） | `/workspace/sh-work/docker-rt/misc_spt/docker_rt/` + `code_sandbox/` | **直连 K8s API**（`pods/exec` websocket） |

两者都是「Docker CLI ↔ unix socket ↔ 假 daemon ↔ K8s Pod」，**不是**本机 containerd。

### 2.1 为何从 pyromind-sdk 切到 misc_spt

在 `SANDBOX_POOL_MAX≈64`、`SWE_BOOT_CONCURRENCY≈32` 时，走 **pyromind-sdk → middleware** 的路径上大量出现：

- `docker exec … true` / agent poll **30s 超时**
- warm / spawn 失败、个别 sample abort

判断：每次 exec 多一跳 **middleware HTTP**，高并发下排队明显。`misc_spt` 直连 apiserver，**exec 延迟与并发通常更好**（仍受 kubelet/apiserver 上限约束）。

代价：`misc_spt` 功能较旧（juicefs / GPU label / 部分 wrapper 等不如 sdk）；需自备 kubeconfig、namespace、镜像权限；与旧 daemon 的 container adopt 不兼容，切换要停 daemon、清 Pod。

### 2.2 切换与运维注意

- Context：`DOCKER_HOST=unix:///tmp/docker-rt.sock`  
- `DOCKER_RT_ORPHAN_POLICY=reap`：重启 daemon 会清掉无主的 managed Pod（避免泄漏，也会弄断旧会话）。  
- kubeconfig 需指向 **可达** 的集群入口（曾把不可达的 `10.200.0.14` 改为集群内网 IP）。  
- Pod 保活：`misc_spt` 实际常用 `sleep 2h`（与 CLI `--entrypoint sleep … infinity` 的 Cmd 元数据可能不一致，以 Pod spec 为准）。

---

## 3. 问题清单（含 pyromind-sdk / misc_spt）

### 3.1 pyromind-sdk（middleware）路径

| 问题 | 现象 | 原因 / 结论 | 状态 |
|------|------|-------------|------|
| 高并发 exec 超时 | `docker exec` 30s timeout，warm/agent 失败 | middleware 多一跳，排队 | **规避**：改用 misc_spt 直连 kube；或降低 `SWE_BOOT_CONCURRENCY` / `SANDBOX_POOL_MAX` |
| 能力绑定 middleware | 部分 API 依赖 terminal websocket / 不支持的 Docker 子命令 | 设计如此（见 sdk `install_wrapper` / `pyromind_sdk_env`） | 已知限制 |
| 与 slime 池叠加 | pool 把并发推高后 middleware 更易顶满 | 池本身正确，后端吞吐不够 | 换后端或降并发 |

### 3.2 misc_spt 直连路径（切换后暴露）

| 问题 | 现象 | 原因 | 修复 |
|------|------|------|------|
| **缺 imagePullSecrets** | `ErrImagePull` / `insufficient_scope`（ACR） | 直连建 Pod 默认不带 secret；middleware 侧同镜像 Pod 带了 `niqi-dev-secret` | `misc_spt/.../backend/runtime.py`：默认 `DEFAULT_IMAGE_PULL_SECRETS=["niqi-dev-secret"]`；可用 `DOCKER_RT_IMAGE_PULL_SECRETS` 或 label `docker-rt.image-pull-secrets` 覆盖 |
| **exec WebSocket 401** | `docker exec` 空失败 / WS `Unauthorized`；REST list 正常 | kubeconfig refresh 把 `BearerToken` 写成 `Bearer <jwt>`，再叠 `api_key_prefix=Bearer` → **`Bearer Bearer …`** | `misc_spt/code_sandbox/kube_environment.py`：Token 规范为单个 `Bearer <jwt>`，并 **去掉** `api_key_prefix["BearerToken"]`（注释写明与 pyromind-sdk 同类修复） |
| **kube API 不可达** | create/list 挂起或失败 | `.kube.yaml` server 指到错误/外网地址 | 改为集群可达内网 endpoint |
| **put_archive / docker cp 极慢** | 小文件 `docker cp` ≈ **15s**；`prepare_workspace` ≈ 20s | `backend/archive.py` 在 stdin EOF 后 **`min(15s)` 死等** WS 关闭 | 已改为：优先等 WS close / `returncode`；否则按体积 **idle 早退**（小文件 ~0.5s）；保留 max timeout。**需重启 docker-rt 生效** |
| **Cmd / entrypoint 语义** | inspect 显示 `Path=infinity`，Pod 实为 `sleep 2h` | create 忽略 Entrypoint，只存 Cmd；keepalive 判定只认 `sleep` | 当前 `-d` 仍可 Running；若将来走 attach 主命令路径需注意 |
| **warm 180s 失败** | `Container is not running` | 高并发下 Pod 未在 ready 窗口内就绪；失败后 `rm -f` | 降并发 / 加大 `SLIME_AGENT_DOCKER_READY_TIMEOUT_SEC`；属容量问题非逻辑必崩 |

### 3.3 Slime ↔ docker-rt 交互层

| 问题 | 说明 |
|------|------|
| `write_file` 用 `docker cp` | 小文件也走 archive，被 3.2 的 15s 坑放大；可选优化：小文件改 `exec`+`cat`/`base64` |
| `ensure_agent_user` 的 `chown -R` | 大仓库上可能秒～十几秒；baked + 小仓时通常不是主因 |
| READY 180 vs kube ready 600 | slime 侧更早放弃；与 docker-rt start 超时不一致时会出现「run 回来了但一直 not running」类日志 |

---

## 4. 建议与后续

1. **重启 docker-rt** 以加载 `put_archive` 早退补丁后，用小文件 `docker cp` 确认从 ~15s 降到亚秒～1s 量级；再看 timeline 里 `prepare_workspace`。  
2. 长期：pyromind-sdk middleware 若仍要用，需评估 middleware 并发与超时；或给 sdk 也带上与 misc_spt 对齐的 pull-secret / auth / archive 策略。  
3. Pool：推理 GPU 未打满时，墙钟瓶颈在 agent/环境；优先稳 warm（并发、ready timeout），而不是加 rollout 卡。  
4. 加大 `ROLLOUT_BATCH_SIZE` 时记得改 **agents 脚本 DEBUG 块外** 的默认（曾误改只在 `DEBUG_TRAIN_MEM=1` 内的行）；`GLOBAL_BATCH` 随 `batch × n_samples` 变大，训练更久、峰值显存一般不涨。

---

## 5. 关键命令速查

```bash
# docker-rt 状态
docker context show
docker ps -a
# managed pods（misc_spt python client）
cd /workspace/sh-work/docker-rt/misc_spt/docker_rt
.venv/bin/python -c "..."  # list label docker-rt.managed=true

# 小文件 cp 耗时（验证 archive 补丁）
echo hi >/tmp/t.md
time docker cp /tmp/t.md <container>:/tmp/t.md
```

Timeline 里看 prepare：`runs/.../timelines/rollout_*.json` → `prepare_workspace` 的 B/E 跨度。
