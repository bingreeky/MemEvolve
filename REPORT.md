# 考核报告：Memory Systems that Evolve（题目一）

> 前沿 AI 方向在线考核 · 题目一 · 论文：[MemEvolve: Meta-Evolution of Agent Memory Systems](https://arxiv.org/abs/2512.18746)（arXiv 2512.18746）
> 本仓库为官方实现 [bingreeky/MemEvolve](https://github.com/bingreeky/MemEvolve) 的 fork，所有新增/修改均以独立 commit 叠加在原代码之上。

## 1. 实验设置

| 项目 | 取值 |
|---|---|
| 执行框架 | Flash-Searcher（仓库自带，DAG 并行深搜 agent） |
| 模型 | `deepseek-v4-flash`（DeepSeek 官方 API，OpenAI 兼容接口） |
| 判分 | LLM-as-a-Judge，裁判模型同为 `deepseek-v4-flash`（需注意裁判与被试同源） |
| Benchmark | xBench-DeepSearch（2505 版加密 CSV），取前 20 条（`data[:20]`，各设置任务集合与顺序一致） |
| Memory 设置 | ① No-Memory（对照）② `lightweight_memory`（**MemEvolve 自动进化产物**，使用官方发布版本，未重新执行 meta-evolution）③ `expel`（semantic 记忆 baseline）④ `voyager`（procedural 记忆 baseline） |
| 运行参数 | `max_steps=40`；memory 组 `concurrency=1`（串行保证 online 记忆逐条积累，与论文 online 模式一致）；No-Memory 组 `concurrency=4` |
| 控制变量 | 每个 memory run 前清空对应 `storage/<provider>/`（从空记忆起步）；跑完归档记忆终态 |

### 复现命令

```bash
# 环境：Python 3.10 + Flash-Searcher-main/requirements.txt，.env 配置见 .env.example
cd Flash-Searcher-main

# Run 0: No-Memory 对照
python run_flash_searcher_mm_xbench.py \
    --infile ./data/xbench/DeepSearch.csv \
    --outfile ./xbench_output/nomem_20.jsonl \
    --sample_num 20 --max_steps 40 --concurrency 4

# Run 1–3: 三个 memory system（memory_provider 依次换成 expel / voyager）
rm -rf storage/lightweight_memory
python run_flash_searcher_mm_xbench.py \
    --infile ./data/xbench/DeepSearch.csv \
    --outfile ./xbench_output/lightweight_20.jsonl \
    --memory_provider lightweight_memory \
    --sample_num 20 --max_steps 40

# 结果汇总（绕过 eval_utils.py 的判分统计 bug，见 §5）
python summarize_results.py xbench_output/*_20.jsonl
```

## 2. 主结果

| 设置 | 准确率 | 平均耗时/任务 | 平均 token/任务 | 平均 API 调用 | memory 实际注入率 |
|---|---|---|---|---|---|
| No-Memory | **18/20 (90%)** | 196.8 s | 189,015 | 15.8 | — |
| **Lightweight**（MemEvolve 进化） | **18/20 (90%)** | 336.9 s (+71%) | 264,103 (+40%) | 27.3 (+73%) | 20/20 |
| ExpeL（semantic） | **17/20 (85%)** | 257.1 s (+31%) | 274,383 (+45%) | 18.2 (+15%) | 19/20 |
| Voyager（procedural） | **18/20 (90%)** | 219.9 s (+12%) | 200,074 (+6%) | 16.7 (+6%) | **0/20** |

按 `task_id` 对齐的逐题正误（o=对，x=错）：

```
No-Memory     ooooooooxooooooxoooo
Lightweight   oxooooooxooooooooooo
ExpeL         oxooooooxoooooooooox
Voyager       oxoooooooooooooxoooo
```

总成本：4 组 × 20 条 ≈ 1,860 万 token（输入占 93%），DeepSeek-V4-Flash 计价合计约 ¥15。

**核心观察：在强基线（90%）下，所有 memory system 的准确率收益为 0 或为负，且全部带来显著成本增加。** 收益与伤害集中在少数难题上并互相抵消（详见 §3）。这与论文 Table 3 的方向一致（xBench 上多数人工 memory baseline 不增益甚至降分），但比论文更极端——原因是我们的 20 条子集 + 更强的执行模型把基线推到了 90%，留给 memory 的提升空间只剩 2 条题。

## 3. 成功 / 失败 Case 分析（题目 iii）

四组结果出现分歧的任务共 4 条，全部做了轨迹级核查（memory 注入内容存于轨迹的 `memory_guidance` 字段）。

### Case A（任务 5）：memory 带来收益 —— 电影台词 → 国标羽绒制品计算题

多跳题：从台词典故定位"羽毛"，再依 GB/T 11881-2006 国标计算最少需要几只家禽。golden=12。

- **Lightweight（✓）**：working memory 在第 1 步即注入正确的问题框架——"GB/T 11881-2006 是羽绒羽毛国标；目标是算出做 10 个合规产品最少需要几只家禽"。后续步骤沿此框架推进，答对。
- **ExpeL（✓）**：检索到的"相似成功案例"内容上完全无关（一道 B 站视频题），但其中**策略层经验**（拆解子目标 → 并行多种检索式 → 交叉验证）成功迁移，同样答对。
- No-Memory / Voyager（✗）均答 8（漏算环节）。

> 结论：事实密集、链路长的题目上，working-memory 的"问题框架固定"与 semantic memory 的"策略迁移"都能产生真实收益，且机制不同。

### Case B（任务 10）：memory 帮倒忙 —— 北京三祠堂等距点

几何多跳题（求三点外心到三点的距离），golden=6~7 km。**唯一答对的是 No-Memory**（6.87 km，16 步、36 万 token）。

- **Lightweight（✗，173 万 token，42 步，答 27 km）**：记忆把**不完整的中间结论固化为"已确认事实"**注入——两个祠堂有坐标、第三个只有地址无坐标。agent 拿残缺坐标硬算外心，且后续步骤不再质疑这些"事实"，越走越偏。
- **ExpeL（✗，195 万 token，答 4.4 km）**：注入的相似案例与几何计算无关，纯噪声；agent 在反复检索中烧掉基线整组一半的 token。
- Voyager（✗，102 万 token）：零注入（见 Case D），失败属于该题的高方差本性。

> 结论：这是 memory 的结构性风险样本——**记忆会把早期未经验证的中间结论"钉死"**，使 agent 丧失自我纠错能力；同时该题暴露了所有设置都缺少"成本止损"机制（单题烧到 195 万 token 无任何告警）。

### Case C（任务 17）：memory 无能为力 —— 历任校长姓氏统计

边界裁量 + 聚合统计题（哪些前身机构的校长算在内），golden=王。No-Memory、Lightweight、ExpeL 全错（典型错误：把存疑人选计入后答"陈王并列"），唯一答对的 Voyager 实为零注入的"运气采样"。

> 结论：瓶颈在裁量与推理而非经验复用的题型上，memory 不构成杠杆。

### Case D（横切发现）：Voyager 全程零注入，等价于第二个无记忆基线

跑完 20 条任务后 `voyager_memory.json` 中 `memories: []`——**一条技能都没存进去**，自然 20 条任务零检索零注入。Voyager 的 encode 设计面向 Minecraft 式"可复用代码技能"，与深搜轨迹（搜索词 + 网页摘要）不匹配，EvolveLab 复现版对此**静默失败**：不报错、不告警，表面上正常跑完。其 18/20 应解读为基线的又一次采样，而非 procedural memory 有效的证据。

## 4. 哪种形态的 memory 更有效？（题目 iv）

基于上述证据，按记忆形态分述：

| 形态 | 本实验载体 | 证据 | 适配判断 |
|---|---|---|---|
| Working/episodic（任务内事实与框架） | Lightweight 的 guidance 注入 | Case A 救场、Case B 闯祸 | **双刃剑**：适合事实密集多跳题，但必须配验证门控，否则固化错误 |
| Semantic（跨任务抽象经验/insight） | ExpeL（积累 80 条 insights） | Case A 策略迁移成功；其余多为噪声且成本最高 | 偶发收益，**检索相关性是命门**——无关案例占据上下文是纯负担 |
| Procedural（技能/工作流库） | Voyager（积累 0 条） | Case D 静默失效 | **与深搜任务族结构性不匹配**：轨迹中无"可封装技能"可提取 |
| Tool-use memory（API/工具用法） | 本次未单独覆盖 | 论文 Figure 7 中 Lightweight 的 tool-use suggestion 属此类 | 推断最适合"工具行为可复用"的场景（如 MediaWiki API 查历史版本） |

总判断：**没有普适最优的记忆形态，任务族决定形态价值**——这正是 MemEvolve"让架构随任务进化"的立论前提，我们的 20 条实验从正反两面支持了它：进化产物 Lightweight 确实比两个人工 baseline 表现更稳（唯一在收益 case 上有清晰机制、且未净降分的系统），但它远非免费（+40% token、+73% 调用），且同样未解决"错误固化"问题。

## 5. 发现的 Limitation 与已实施的 patch（题目 v 之一）

1. **xBench 判分统计 bug（已修，见配套 PR）**：`eval_utils.py` 的 `generate_unified_report` 按 GAIA 的字符串字段 `judgement` 统计正误，而 xBench runner 写入的是数字字段 `score`，导致 xBench 的官方报告恒为 `Accuracy: 0.00%`（资源统计不受影响）。修复：统计时兼容两种 schema。
2. **结果文件以追加模式写入且行序为完成序**：重跑同名 outfile 会混入旧记录；并发时行序≠任务序，跨组对比必须按 `task_id` 对齐。配套的 `summarize_results.py` 做了去重（保留每个 task_id 最后一条）并按 id 对齐。
3. **记忆写入静默失败**（Case D）：provider 存储 0 条记忆不产生任何告警，实验者可能误以为 memory 在工作。建议在 run 结束时输出 memory store/retrieve 命中统计。
4. **无成本止损**：单任务可烧到 195 万 token（基线整组的一半）而不触发任何熔断。
5. **进化产物自带 7 条冷启动记忆**：`lightweight_memory` 初始化即注入 5 条策略 + 2 条操作记忆。这意味着 meta-evolution 把"经验"部分固化进了架构本身，与"从空记忆起步"的 baseline 严格说不同起跑线（对比时应披露）。
6. **裁判与被试同模型**：判分由同一个 `deepseek-v4-flash` 完成，存在自我偏好风险（本次答案多为数值/实体，影响有限）。

## 6. meta-evolution 与 harness 自进化的关系，以及我会改哪里（题目 v）

**关系**：MemEvolve 本质上是 harness 自进化的一个**受限特例**。完整的 harness 自进化（如 Darwin Gödel Machine 一脉）允许改写 prompt、工具、规划器乃至进化逻辑自身；MemEvolve 把可进化面收窄到 (Encode, Store, Retrieve, Manage) 四模块接口之内。收窄换来三样东西：搜索空间可控、坏变异不破坏系统其余部分、fitness 信号可归因到记忆行为。代价是天花板被接口锁死——Case B/C 暴露的问题（中间结论无验证、裁量类推理瓶颈、无成本熔断）都落在接口之外，无论进化多少轮记忆架构都修不到。

**如果让我改 MemEvolve，按优先级**：

1. **给 Retrieve 加"验证门控"（方法级，针对 Case B）**：注入记忆时区分"已验证事实"与"待验证假设"，对坐标/数值类中间结论强制要求来源标注，agent 对未验证项保留质疑权。可在 Lightweight 的 working memory 写入路径上实现，做 20 条 patch 前后对比。
2. **把"资源消耗"提为一等公民的进化信号（方法级）**：论文的 fitness 已含 cost/delay，但只在架构选择层起作用；应下沉到 Manage 模块——运行时单任务 token 超阈值即触发记忆侧的"止损摘要"。
3. **fitness 评估的统计稳健性（方法级）**：外层进化每候选仅 60 条轨迹、K=1 贪心保留，以我们观察到的单题方差（同一题不同设置 36 万~195 万 token、对错翻转），单次排名噪声极大。应引入置信区间或配对检验再淘汰。
4. **记忆健康度自检（代码级，针对 Case D）**：EvolveLab 层面为所有 provider 加 store/retrieve 命中率统计与零存储告警——否则进化过程中产生的"静默死亡"候选会污染 fitness 信号。

### 6.1 已实施：提案 1 的方法级 patch 与 ablation（验证门控）

**实现**（见配套 PR，44 行新增 / 7 行修改，沿 extract → store → inject 三点改造）：提取 prompt 输出 `{fact, source, verified}`，verified 仅当数值直接读自当前步上下文中的来源、且搜索未果必须记录为显式缺口；存储带 `(source: …)` / `[UNVERIFIED]` 溯源标注；注入时"已验证事实"与"待验证假设"分区渲染，附"未验证数值用于计算前先核实"指令。

**Ablation 结果**（同 20 条任务、空记忆起步、同模型，单次运行）：

| 设置 | 准确率 | 平均 token/任务 | 平均 API 调用 |
|---|---|---|---|
| Lightweight 原版 | 18/20 | 264,103 | 27.3 |
| Lightweight 门控版 | **15/20** | 279,597 | 28.2 |

**靶子任务（#10 三祠堂）上机制完全生效**：祠堂别名猜测被正确标为假设；"百科页面未提供坐标"被记录为三条显式缺口；agent 先补齐坐标（带 Wikipedia 来源）再计算；token 173 万 → 105 万（**-39%**）。但答案仍错（26.85 km vs golden 6~7 km）——三点近共线使外心对坐标误差极度敏感，这是**数值病态问题**，超出 memory 层可修范围。

**总分回退的机制分析**（#5/#9/#16 翻错，无任务翻对）：#5 的轨迹最有信息量——门控**正确推翻了原版固化的错误假设**（GB/T 11881-2006 实为羽毛球国标，而非原版记忆中的"羽绒标准"），但 agent 随后收集到多条互相矛盾的"已验证事实"（每球 16 根羽毛 / 每鹅仅 14 根可用 / 单翅 6 根不可混用），在矛盾消解上失败，反而答错。即：**验证压力扩大了搜索面、暴露了更多来源矛盾，而 agent 缺乏矛盾仲裁能力时，更多的"诚实"反而带来更差的聚合**。原版的"过度自信"在部分任务上歪打正着。

**结论与修正方向**：全局无差别门控净收益为负（n=20 单次运行，方差告诫适用：原四组实验中同一任务在不同设置下本就频繁翻转）。门控应当**选择性触发**——仅对将进入下游计算的数值/坐标类事实施加验证要求，对定性事实不加质疑负担；并需配套"来源矛盾仲裁"机制（如多数表决、权威源优先级）才能兑现收益。这个负结果反过来印证了 §6 的判断：这类门控变体完全可以在 MemEvolve 的 (E/U/R/G) 接口内表达，交给 meta-evolution 在更大任务批次上筛选，比人工一次性设计更可靠——这正是该框架存在的意义。

## 7. 产物清单

| 文件 | 说明 |
|---|---|
| `REPORT.md` | 本报告 |
| `Flash-Searcher-main/summarize_results.py` | 结果汇总脚本（task_id 对齐 + 去重 + 对比表/逐题网格） |
| `Flash-Searcher-main/eval_utils.py` 修复 | 判分统计 bug patch（独立 PR） |
| `lightweight_memory_provider.py` 验证门控 patch | 方法级 patch + ablation（独立 PR） |
| `xbench_output/`（本地，不入库） | 5 组原始轨迹 jsonl + 记忆库终态归档（含解密题文，遵循 xBench 不上传明文的要求） |

*实验日期：2026-06-11。执行环境：macOS / Python 3.10 / DeepSeek API。*
