# Design: Slime Coding-Agent RL 训练演变文档补全

日期：2026-09-22  
状态：待用户审阅后进入 implementation plan

## 目标

在不重写 2026-07-25→08-13 既有叙述的前提下，补全 `a_csv` 中两份演变史 HTML，并新增一份按实验汇总「效果与不足」的 Markdown，覆盖 `slime/runs` 中早于 `agent_offload_pyrodash4b_sft_entropy_docker_async_turn_20260919_135347` 的 `agent_offload_*` 训练（不含该 run 及之后）。

## 交付物

路径均在 `/workspace/work/spt/slime/a_csv/`：

1. **更新** `奖励与损失函数演变史.html`  
   - 保留既有 timeline / 公式 / 模式对照（07-25→08-13 不重写正文）。  
   - 更新页眉统计与「当前」表述至新终点。  
   - **追加** 08-13 之后的**逐实验明细表**（方法 + 奖励/损失相关设置）。

2. **更新** `数据演变史.html`  
   - 保留既有五阶段叙述（不重写）。  
   - 更新统计与「当前」表述。  
   - **追加** 同期**逐实验**数据/基座表（训练集、池规模、bake、基座 checkpoint）。

3. **新建** `训练效果与不足汇总.md`  
   - 阶段速览（A）+ 逐实验效果/不足 + 关键 run 展开（C）+ 同配置重跑附录。

**明确不做：** 重做成总实验看板；重写早期 timeline 段落；默认不重建 `experiments.xlsx`（除非实现时为核对所必需且用户未反对）。

## 实验收录与去重

- **集合：** `runs/agent_offload_*`，目录名时间戳 `ts < 20260919_135347`。  
- **去重规则 B：** 相同配置只保留一次「最完整」代表；其余列入附录，标注「同配置重跑」。  
- **完整度优先序：** 有 `offload_config.json` > rollout/日志更完整 > 目录产物更全。  
- **配置等价键（实现时固定）：** 基座、训练集、训练方法族（GRPO / GRPO+SFT / GiGPO / entropy-turn 等）、奖励模式与主要超参（λ、α、seek_only、format/malformed、SFT λ、tag_prob、advantage/filter、offload_insert 等）。名称时间戳不同但键相同 → 合并。

## 逐实验必填字段

### 奖励/方法表（进奖励 HTML，MD 可简写并回链）

| 字段 | 说明 |
|------|------|
| run 名 / 时间 | 保留代表 run 全名 |
| 训练方法 | 如 async GRPO、GRPO+SFT、GiGPO、turn advantage、entropy、offload_insert、trun 截断等实际启用项 |
| 奖励模式 | cost_aware / help_seeking 等 |
| 奖励超参 | λ、α、format_penalty、empty_scale、seek_only_all_wrong、unique_bonus、malformed 等能还原的全部列出 |
| Loss / advantage | GRPO clip、KL/entropy 系数、`compute_turn_advantages`、GiGPO 分组、SFT λ 与 CE 范围等 |
| 备注 | 过滤规则、未跑完等 |

### 数据表（进数据 HTML）

| 字段 | 说明 |
|------|------|
| run 名 | 与上表对齐 |
| 训练集 | jsonl 名与规模/池来源 |
| 基座 | checkpoint / SFT 版本名 |
| 环境相关 | bake、agent 路由等若相对前一代表有变则写 |

还原不出的字段写 **`未记录`**，禁止臆造。

## 参数还原来源（优先级）

1. `runs/<run>/offload_config.json`  
2. `runs/<run>/run.log`（启动命令与环境变量）  
3. `a_csv/experiments.csv` 与 `imgs/`  
4. 旁路说明：`奖励函数说明.md`、`GRPO与SFT训练说明.html`、`sft00902-offload-insert-train-metrics.html`、`gigpo_grouping_review_*`  
5. 评测文档：`pyroDash-training/train_test/evaluation/All_Test*.md` 等（仅当能对上 checkpoint/run 时写入「效果」）

## 汇总 MD 结构

1. **总览** — 时间范围、收录/合并后实验数、规则 B 说明  
2. **阶段速览（A）** — 按后期自然分期各 3–5 行：在试什么、整体效果、主要不足（分期仅作导航，**不是**用阶段代替逐实验）  
3. **逐实验明细** — 每个保留实验：方法+奖励（简）、数据/基座、效果（训练为主，有则补评测）、不足/为何改下一配置  
4. **关键 run 展开（C）** — 每阶段自动选 1–2 个代表，补充曲线/评测细节  
5. **附录** — 被合并的同配置重跑名单  

效果依据：**训练指标为主，评测有则补（来源 C）**。

## 与旧 HTML 的关系

- 旧文是「设计演变史」叙事；新增部分是「逐实验台账」。  
- 早期「08-13 → 当前」等措辞改为指向新终点或改为「08-13 节点」，避免与后文矛盾，但**不重写**该节技术内容。  
- 仍保留「非严格消融」类免责声明，并延伸到后期多变量同时变化的事实。

## 成功标准

- 两份 HTML 打开后，08-13 之后每个**保留**实验都能查到方法与奖励（或明确 `未记录`）。  
- MD 对同一集合有对应效果/不足，且附录列出被合并重跑。  
- 不含 `20260919_135347` 及更晚 run。  
- 早期 07-25→08-13 叙事段落无内容重写（仅允许统计/「当前」措辞与 foreword 链接式修补）。

## 实现顺序（计划阶段细化）

1. 枚举 cutoff 前 `agent_offload_*`，抽取 config/log/csv，做配置去重。  
2. 生成逐实验方法/奖励/数据表草稿，人工核对歧义项。  
3. 追加进两份 HTML（样式沿用现页）。  
4. 撰写 MD（阶段速览 + 逐实验 + 关键展开 + 附录）。  
5. 抽查若干 run 与 `offload_config.json` / 评测文档交叉验证。
