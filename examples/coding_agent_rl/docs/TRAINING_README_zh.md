# Coding RL 训练

---

## 1. 代码与环境

```bash
cd /workspace/work/spt/slime
git checkout gigpo-entropy(默认是这个分支)

micromamba activate slime

export LD_LIBRARY_PATH="\
$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:\
$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

 自检
python -c "import transformer_engine; print('TE OK')"
strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep GLIBCXX_3.4.32
docker ps
```

## 2. 模型（二选一）

**A. Qwen3.5-4B（先跑通，无 offload），先download模型，然后转为torch版本，默认已有**

```bash
export HF_CHECKPOINT=/workspace/models/Qwen/Qwen3.5-4B
export REF_MODEL_PATH=/workspace/models/Qwen/Qwen3.5-4B_torch_dist

hf download Qwen/Qwen3.5-4B --local-dir "$HF_CHECKPOINT"

source scripts/models/qwen3.5-4B.sh
PYTHONPATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}" \
  python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint "$HF_CHECKPOINT" --save "$REF_MODEL_PATH"
```

**B. PyroDash + offload**

```bash
export HF_CHECKPOINT=/workspace/models/pyromind/PyroDash-4B-SFT-0803
export REF_MODEL_PATH=/workspace/models/pyromind/PyroDash-4B-SFT-0803_torch_dist
HF_CHECKPOINT="$HF_CHECKPOINT" SAVE="$REF_MODEL_PATH" \
  bash examples/coding_agent_rl/scripts/convert_pyrodash4b_to_torch_dist.sh

```

---

## 3. 开训

在脚本中修改数据集、具体的参数，或者 放到命令前面
```bash
export NCCL_P2P_DISABLE=1
bash examples/coding_agent_rl/run_pyrodash4b_swe_offload_1node_docker_async_agents.sh
```

参数明细：[run_pyrodash4b_offload_agents_params_zh.md](./run_pyrodash4b_offload_agents_params_zh.md)。

日志：`runs/<EXP_TAG>_*/run.log`。看到 `reward=` / `Update weights` 即正常。

---

## 4. 出事时

```bash
 清残留
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
sleep 3
pkill -9 ray || true

 清残留 sandbox
docker ps -aq --filter name=slime-sb- | xargs -r docker rm -f

 Triton 缓存坏了
rm -rf ~/.triton/cache
```
