# 题目一实验报告：Memory Systems that Evolve

> 前沿 AI 方向在线考核 · 题目一 · 论文：[MemEvolve: Meta-Evolution of Agent Memory Systems](https://arxiv.org/abs/2512.18746)（arXiv 2512.18746）
> 本仓库 fork 自官方实现 [bingreeky/MemEvolve](https://github.com/bingreeky/MemEvolve)，我的全部改动都以独立 commit/PR 叠加在原代码之上，方便和原作者的代码区分。

## 1. 实验设置

我在 Flash-Searcher（仓库自带的深搜 agent 框架）上用 `deepseek-v4-flash` 跑了 xBench-DeepSearch 的前 20 条任务，对比四个设置：

| 项目 | 取值 |
|---|---|
| 模型 | `deepseek-v4-flash`（DeepSeek 官方 API，OpenAI 兼容接口） |
| 判分 | LLM-as-a-Judge，裁判模型同为 `deepseek-v4-flash`（裁判与被试同源，§5 有讨论） |
| Benchmark | xBench-DeepSearch（2505 版加密 CSV），取前 20 条（`data[:20]`，四个设置看到的任务集合与顺序完全一致） |
| Memory 设置 | ① No-Memory（对照）② `lightweight_memory`（**MemEvolve 自动进化产物**，用的是官方发布版本，我没有重新跑 meta-evolution）③ `expel`（semantic 记忆 baseline）④ `voyager`（procedural 记忆 baseline） |
| 运行参数 | `max_steps=40`；memory 组 `concurrency=1`（串行，保证记忆逐条积累，对应论文的 online 模式）；No-Memory 组 `concurrency=4` |
| 控制变量 | 每个 memory run 开始前我都清空了对应的 `storage/<provider>/`，保证从空记忆起步；跑完立刻把记忆终态归档备份 |

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

总成本：4 组 × 20 条大约 1,860 万 token（输入占 93%），按 DeepSeek-V4-Flash 计价合计 15 元左右。

跑完第一组我就发现基线比预想强很多：无记忆就有 90%，比论文里的 69% 高出一截（应该是前 20 条偏简单，加上执行模型比论文当时更强）。这直接导致**所有 memory system 的准确率收益是 0 或负的，而成本全都显著上升**。收益和伤害集中在少数难题上互相抵消，§3 逐条拆。方向上这和论文 Table 3 一致（xBench 上多数人工 memory baseline 不增益甚至降分），只是在我的设置下更极端。

## 3. 成功 / 失败 Case 分析（题目 iii）

四组结果有分歧的任务一共 4 条，我把每条的轨迹都翻了一遍（memory 注入的内容在轨迹的 `memory_guidance` 字段里能直接看到）。

### Case A（任务 5）：memory 帮了忙——电影台词引出的国标计算题

多跳题：先从台词典故定位到"羽毛"，再按 GB/T 11881-2006 国标算最少需要几只家禽，golden=12。

- **Lightweight（答对）**：working memory 第 1 步就注入了正确的问题框架（"这是羽毛相关国标，目标是算 10 个合规产品最少要几只家禽"），后面的步骤一直沿着这个框架走。
- **ExpeL（答对）**：它检索到的"相似成功案例"内容上完全不相干（一道 B 站视频题），但里面"拆子目标、并行换几种检索式、交叉验证"这套打法迁移过来了，也答对了。
- No-Memory 和 Voyager 都答 8，少算了一个环节。

我的结论：事实密集、链路长的题上，working memory 的"框架固定"和 semantic memory 的"策略迁移"都能产生真收益，而且是两种不同的机制。

### Case B（任务 10）：memory 帮了倒忙——北京三祠堂等距点

求三点外心到三点的距离，golden=6~7 km。**唯一答对的反而是 No-Memory**（6.87 km，16 步、36 万 token）。

- **Lightweight（答错，173 万 token、42 步，答 27 km）**：它的记忆把没验证完的中间结论当成"已确认事实"注入了——两个祠堂有坐标、第三个只有地址没坐标，agent 就拿着残缺坐标硬算外心，而且后面的步骤再也没质疑过这些"事实"，越走越偏。
- **ExpeL（答错，195 万 token）**：注入的相似案例和几何计算毫无关系，纯噪声，agent 在反复检索里烧掉了基线整组一半的 token。
- Voyager（答错，102 万 token）：它是零注入状态（见 Case D），失败属于这道题本身方差大。

这条是整个实验里我觉得最重要的样本：**记忆会把早期没验证过的中间结论"钉死"，让 agent 失去自我纠错能力**。它也暴露了框架没有成本止损——一条任务烧到 195 万 token 没有任何告警。

### Case C（任务 17）：memory 无能为力——历任校长姓氏统计

边界裁量加聚合统计的题（哪些前身机构的校长要算进去），golden=王。No-Memory、Lightweight、ExpeL 全错（都把存疑人选算了进去，答"陈王并列"），唯一答对的 Voyager 其实是零注入状态，纯属又采样了一次的运气。瓶颈在裁量和推理上的题，记忆帮不上忙。

### Case D（横切发现）：Voyager 全程零注入，实际是第二个无记忆基线

跑完 20 条任务我去看 `voyager_memory.json`，里面 `memories: []`——一条技能都没存进去，自然也没有任何检索和注入。回头想原因：Voyager 的 encode 是为 Minecraft 那种"可复用代码技能"设计的，深搜轨迹（搜索词 + 网页摘要）里没有它能提取的东西，而 EvolveLab 的复现版对这种情况**静默失败**——不报错、不告警，表面上正常跑完。所以它的 18/20 只能解读为基线的又一次采样，不能当作 procedural memory 有效的证据。

## 4. 哪种形态的 memory 更有效？（题目 iv）

按我拿到的证据分形态说：

| 形态 | 本实验载体 | 证据 | 我的判断 |
|---|---|---|---|
| Working/episodic（任务内事实与框架） | Lightweight 的 guidance 注入 | Case A 救场、Case B 闯祸 | 双刃剑：适合事实密集的多跳题，但必须配验证门控，否则会固化错误 |
| Semantic（跨任务抽象经验） | ExpeL（积累了 80 条 insights） | Case A 策略迁移成功；其余多为噪声且成本最高 | 偶发收益，命门在检索相关性——无关案例占着上下文就是纯负担 |
| Procedural（技能/工作流库） | Voyager（积累了 0 条） | Case D 静默失效 | 和深搜任务族结构性不匹配：轨迹里没有可封装的"技能" |
| Tool-use memory（API/工具用法） | 这次没有单独覆盖 | 论文 Figure 7 里 Lightweight 的 tool-use suggestion 属此类 | 推断最适合工具行为可复用的场景（比如用 MediaWiki API 查历史版本） |

总的看法：**没有普适最优的记忆形态，任务族决定形态价值**。这正好是 MemEvolve"让架构跟着任务进化"的立论前提，我的 20 条实验从正反两面支持了它：进化产物 Lightweight 确实比两个人工 baseline 稳（唯一在收益 case 上机制清晰、又没有净降分的系统），但它不是免费的（+40% token、+73% 调用），而且同样没解决"错误固化"。

## 5. 我发现的 Limitation 和已实施的修复（题目 v 之一）

1. **xBench 判分统计 bug（我修了，见配套 PR）**：`eval_utils.py` 的 `generate_unified_report` 按 GAIA 的字符串字段 `judgement` 统计正误，但 xBench runner 写入的是数字字段 `score`，所以 xBench 的官方报告永远显示 `Accuracy: 0.00%`（资源统计正常）。我加了一个兼容两种 schema 的判定函数，修复后同一份数据从 0% 变为正确的 90%。
2. **结果文件是追加写入，行序是完成顺序**：重跑同名 outfile 会混进旧记录；开并发时行序和任务序对不上，跨组对比必须按 `task_id` 对齐。我写的 `summarize_results.py` 做了去重（每个 task_id 保留最后一条）和按 id 对齐。
3. **记忆写入静默失败**（Case D）：provider 存了 0 条记忆不会有任何告警，很容易误以为 memory 在工作。建议 run 结束时输出 store/retrieve 命中统计。
4. **没有成本止损**：单任务可以烧到 195 万 token 而不触发任何熔断。
5. **进化产物自带 7 条冷启动记忆**：`lightweight_memory` 初始化就注入 5 条策略 + 2 条操作记忆。也就是说 meta-evolution 把一部分"经验"固化进了架构本身，严格说它和"从空记忆起步"的 baseline 不在同一起跑线，对比时应该披露。
6. **裁判与被试同模型**：判分和作答都是 `deepseek-v4-flash`，有自我偏好的风险（这次答案多是数值和实体，影响应该有限）。

## 6. meta-evolution 与 harness 自进化的关系，以及我会改哪里（题目 v）

**关系**：我认为 MemEvolve 本质上是 harness 自进化的一个受限特例。完整的 harness 自进化（Darwin Gödel Machine 那一脉）什么都能改——prompt、工具、规划器、甚至进化逻辑自身；MemEvolve 把可进化面收窄到 (Encode, Store, Retrieve, Manage) 四个模块的接口之内。收窄换来三样东西：搜索空间可控、坏变异不会破坏系统其余部分、fitness 信号能归因到记忆行为。代价是天花板被接口锁死——Case B/C 暴露的问题（中间结论无验证、裁量类推理瓶颈、无成本熔断）都落在接口之外，记忆架构进化多少轮都修不到。

**如果让我改，按优先级**：

1. **给 Retrieve 加"验证门控"（针对 Case B）**：注入记忆时区分"已验证事实"和"待验证假设"，坐标、数值类中间结论强制要求来源标注，让 agent 对未验证项保留质疑权。这条我实际做了，见 §6.1。
2. **把资源消耗下沉为运行时信号**：论文的 fitness 已经含 cost/delay，但只在架构选择层起作用；应该下沉到 Manage 模块——单任务 token 超阈值就触发记忆侧的止损摘要。
3. **fitness 评估的统计稳健性**：外层进化每个候选只评 60 条轨迹、K=1 贪心保留。以我观察到的单题方差（同一道题在不同设置下 36 万~195 万 token、对错来回翻转），单次排名的噪声非常大，应该引入置信区间或配对检验再做淘汰。
4. **记忆健康度自检（针对 Case D）**：EvolveLab 层面给所有 provider 加 store/retrieve 命中率统计和零存储告警，否则进化过程中"静默死亡"的候选会污染 fitness 信号。

### 6.1 我实施的方法级 patch 与 ablation（验证门控）

**实现**（见配套 PR，44 行新增 / 7 行修改，沿 extract → store → inject 三个环节改造）：提取 prompt 改为输出 `{fact, source, verified}`，verified 只有在数值直接读自当前步上下文里的来源时才为真，搜索未果必须记录成显式缺口；存储时带 `(source: …)` 或 `[UNVERIFIED]` 标注；注入时把"已验证事实"和"待验证假设"分区渲染，并附"未验证数值用于计算前先核实"的指令。

**Ablation 结果**（同 20 条任务、空记忆起步、同模型，单次运行）：

| 设置 | 准确率 | 平均 token/任务 | 平均 API 调用 |
|---|---|---|---|
| Lightweight 原版 | 18/20 | 264,103 | 27.3 |
| Lightweight 门控版 | **15/20** | 279,597 | 28.2 |

**靶子任务（#10 三祠堂）上机制完全生效**：祠堂别名的猜测被正确标成假设；"百科页面未提供坐标"被记录成三条显式缺口；agent 这次是先补齐坐标（带 Wikipedia 来源）再计算；token 从 173 万降到 105 万（-39%）。但答案还是错的（26.85 km）——后来我意识到三个祠堂几乎共线，外心位置对坐标误差极其敏感，这是数值病态问题，不归 memory 层管。

**总分回退的分析**（#5/#9/#16 翻错，没有任务翻对）：#5 的轨迹最有意思——门控其实**抓住了原版固化的一个错误假设**（GB/T 11881-2006 实际是羽毛球国标，不是原版记忆里写的"羽绒标准"），但 agent 接着收集到一堆互相矛盾的"已验证事实"（每球 16 根羽毛 / 每鹅只有 14 根可用 / 单翅 6 根不能混用），在矛盾消解上失败，反而答错了。我的理解是：**验证压力扩大了搜索面、暴露了更多来源矛盾，agent 又没有仲裁矛盾的能力，于是更多的"诚实"换来了更差的聚合**。原版的"过度自信"在一部分题上反而歪打正着。

**结论**：全局无差别的门控净收益为负（n=20 单次运行，方差告诫适用——原来四组实验里同一道题本来就经常翻转）。修正方向：门控应该**选择性触发**，只对将进入下游计算的数值/坐标类事实施加验证要求，定性事实不加质疑负担；同时要配套来源矛盾的仲裁机制（多数表决、权威源优先级之类）才能兑现收益。这个负结果反过来印证了 §6 的判断：这类门控变体完全可以在 MemEvolve 的四模块接口内表达，交给 meta-evolution 在更大的任务批次上筛选，比我这样人工一次性设计更可靠——这恰恰是这个框架存在的意义。

## 7. 产物清单

| 文件 | 说明 |
|---|---|
| `REPORT.md` | 本报告 |
| `Flash-Searcher-main/summarize_results.py` | 结果汇总脚本（task_id 对齐 + 去重 + 对比表/逐题网格） |
| `Flash-Searcher-main/eval_utils.py` 修复 | 判分统计 bug 修复（独立 PR） |
| `lightweight_memory_provider.py` 验证门控 | 方法级 patch + ablation（独立 PR） |
| `xbench_output/`（本地，不入库） | 5 组原始轨迹 jsonl + 记忆库终态归档（含解密题文，遵循 xBench 不上传明文的要求） |

*实验日期：2026-06-11。环境：macOS / Python 3.10 / DeepSeek API。*
