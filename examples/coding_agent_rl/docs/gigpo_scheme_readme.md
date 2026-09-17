# GiGPO 方案解读（SWE / coding-agent）

> 对照仓库内可视化：  
> - `runs/.../docker_async_20260804_035850/gigpo_*`（tool-intent / git-diff）  
> - `runs/.../phase2_sft03_prob_*/gigpo_*_turn_rewards.html`（含 \(A_I\)）  
> - 对比页：`/workspace/work/mjy/traj_compare_group_advantage.html`（粗组广播变体，非标准逐步 \(A_S\)）

---

## 1. 要解决什么

长轨迹 + 稀疏 outcome（解/不解）时，普通 GRPO 只有 **轨迹级** \(A_E\)：成功轨上每一步一起被抬，失败轨上一起被压。

GiGPO（Group-in-Group Policy Optimization）在 **不学 critic** 的前提下，用「同锚点组内相对回报」再分一层 **逐步优势** \(A_S\)。

---

## 2. 公式（标准形态）

对同一 prompt 的 sibling 轨迹（你们常取 8 条；对比页用 3 模型示意）：

\[
A_t = A_E + A_S(t)
\]

| 符号 | 含义 | 怎么算 |
|------|------|--------|
| \(R\) | 轨迹 outcome | 解开≈1，否则≈0（或 dump 里的 episode reward） |
| \(A_E\) | 轨迹间 GRPO | \(R - \mathrm{mean}(R\mid\text{siblings})\) |
| \(G_t\) | 从该步看的折扣回报 | \(G_t = R\cdot\gamma^{n-1-t}\)（\(\gamma\) 常用 0.95） |
| 锚点组 | 「同状态」的工程近似 | 见下节 |
| \(\overline{G}\) | 同组均值 | 组内所有（跨轨）成员的 \(G\) 平均 |
| \(A_S(t)\) | **逐步** 组内优势 | \(G_t - \overline{G}_{\text{同组}}\) |

要点：

1. **同组 ≠ 同优势**。同锚点下每步仍有自己的 \(G_t\)；越靠后 \(G_t\) 越大（折扣），\(A_S\) 通常越高。  
2. **\(A_S\) 是 advantage，不是 step reward**。多数实现里即时 \(r_{\mathrm{imm}}\approx 0\)，信用来自 outcome 的折扣分配 + 组内去均值。  
3. **锚点撞不上** → 该步 \(A_S=0\)，退化成纯 \(A_E\)。

---

## 3. 你们仓库里的三种锚点

| 变体 | 锚点 | 成组 | \(A_S\) 质量 |
|------|------|------|-------------|
| **tool-intent** | 意图 + 工具（如「验证 · pytest」） | 密 | 有信号，但是粗；同意图不等于同状态 |
| **exact git-diff** | 累计 git diff 完全一致 | 稀 | 更接近「同状态」；常 \(A_S\approx 0\) |
| **fuzzy / files-only / action-sig** | 相似 diff / 文件集 / 动作签名 | 很密 | 覆盖高，错绑风险高 |
| **phase2 + \(A_I\)** | 轨内按 diff 切段，段内再折扣 | — | \(A_I\) 额外奖励「靠近段末」；\(A=A_E+w(A_S+A_I)\) |

docker 页常见：\(A = A_E + A_S\)（无 \(A_I\)）。  
phase2 页：\(A = A_E + w(A_S + A_I)\)。

---

## 4. 一步在算什么（示意）

```text
Sibling 轨迹:  T0✓  T1✗  T2✗  ...

某步 t 落在组 g =「探索 · Read」
  G_t = R * γ^{距终点}
  Ā_G = mean( 所有落在 g 的步的 G )   ← 跨轨
  A_S = G_t - Ā_G
  A   = A_E + A_S
```

- 成功轨、靠后的同组步：\(G\) 大 → \(A_S\) 正  
- 失败轨同组步：\(G=0\) → \(A_S\) 常为负  
- **同轨、同组、更早的步**：\(G\) 更小 → \(A_S\) 更低（折扣因子的作用）

---

## 5. Case 演示

### Case A — sybil pr296（docker tool-intent HTML）

**设定**：8 sibling，1 成 7 败；\(\gamma=0.95\)；\(A=A_E+A_S\)。

成功轨 `trajectory 24`：\(R\approx 0.996\)，\(A_E\approx +0.87\)。

**同组 T0「验证 · pytest」在成功轨上随时间升高（折扣）：**

| 位置 | \(G\) | \(A_S\) | \(A\) |
|------|-------|---------|-------|
| t6 较早 pytest | 0.13 | +0.06 | +0.93 |
| t17 | 0.24 | +0.16 | +1.03 |
| t43 靠后 pytest | 0.90 | +0.82 | +1.69 |

同组内不是常数：越晚 \(A_S\) 越大。

失败轨 `trajectory 25` 同锚点 T5「探索 · Read」首步：\(A_S\approx -0.01\)，\(A\approx -0.14\)（主音量仍是负 \(A_E\)）。

**解读**：tool-intent 能把「成功轨走过的同意图工具」抬高，但抬的是 **带位置偏差的相对 \(G\)**，不是「这一次 pytest 语义上是否有用」。

---

### Case B — astropy-14182（对比 8 题里的 ✗✓✗）

**Sibling**：DeepSeek✗ / SFT✓ / Qwen✗ → \(\mathrm{mean}R=1/3\)。

| 模型 | \(R\) | \(A_E\) |
|------|-------|---------|
| DeepSeek | 0 | −0.333 |
| SFT | 1 | +0.667 |
| Qwen | 0 | −0.333 |

对粗组 `探索/定位 · Bash:python`（跨三模型共桶），\(\overline{G}\approx 0.24\)。

**SFT 同组内逐步 \(A_S\)（真 GiGPO，非广播）：**

| turn | \(G_t\) | \(A_S\) | \(A=A_E+A_S\) |
|------|---------|---------|----------------|
| t11 偏早 | 0.09 | −0.15 | +0.52 |
| t15 | 0.11 | −0.13 | +0.54 |
| t52 偏晚 | 0.74 | +0.50 | +1.16 |
| t56 更晚 | 0.90 | +0.66 | +1.33 |

Qwen 同组（失败，\(G=0\)）：每步 \(A_S\approx -0.24\)，\(A\approx -0.57\)。

**解读**：

- \(A_E\)：谁解开了  
- \(A_S\)：同锚点里，成功轨靠后的步更吃香；失败轨同组被压  
- 组内折扣 → **同轨同组仍会「越晚越大」**

---

### Case C — exact git-diff（同一 docker run）

同实例上 exact diff 分组后，大量步 \(A_S=0\)（组内 \(G\) 无差异或不成组）。  
**逐步项几乎熄火**，训练信号≈纯 GRPO \(A_E\)。

---

## 6. 和 LLM turn judge 的边界（方案层）

| | GiGPO \(A_S\) | LLM \(r_i-\bar r\) |
|--|--|--|
| 信号类型 | 组内相对 **advantage** | 过程 **reward** → 再变残差 |
| 是否要 sibling 撞锚点 | 要 | 不要 |
| 同组内折扣 | 有 → 位置效应 | 无（按语义打分） |
| SWE 空转 / 错 offload | 难专门打 | 能打 |
| 成本 | 几乎 0 | API / 延迟 |

推荐叙事（对内对齐）：

> GiGPO = GRPO \(A_E\) + 同锚点逐步 \(A_S\)（折扣回报相对优势）。  
> LLM judge = 过程奖励，接到 \(A_t=A_s+(r_i-\bar r)\)。  
> 二者都不是「对方的替代品」；SWE 上锚点稀/粗时，GiGPO 的逐步项弱或带位置偏置。

---

## 7. 落地建议（短）

1. **训练**：保留 GRPO \(A_E\)；GiGPO 用 **exact / 强约束锚点** 时加小权重 \(\lambda A_S\)；粗意图桶慎当主信号。  
2. **可视化**：展示逐步 \(G_t,A_S\)（同组不同值），避免「组级广播」与真 GiGPO 混淆。  
3. **过程问题**（spin、乱 offload）：优先 LLM judge 或规则过程分，而不是加粗 GiGPO。

---

## 8. 相关产物

| 路径 | 内容 |
|------|------|
| `.../gigpo_tool_intent_turn_rewards_pr296_diff_sequence.html` | Case A 全量 |
| `.../gigpo_gitdiff_turn_rewards_pr296_diff_sequence.html` | Case C |
| `/workspace/work/mjy/traj_compare_group_advantage.html` | 8 题×3 模型粗组页（当前为组级广播变体） |
| `examples/coding_agent_rl/docs/online_llm_turn_judge.md` | LLM 逐步分方案 |
