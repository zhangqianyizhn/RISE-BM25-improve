# RISE-BM25 on EnterpriseRAGBench

运行 EnterpriseRAGBench 的 **RISE-BM25（无 TOC）**。支持 10,970 / 511,959 篇文档，两种规模使用相同的 500 道 QA，默认题目并发为 4。

基于 [官方 RISE 固定提交](https://github.com/texttron/RISE/tree/32672b59ead8381decae7d412f05bb53d399946a) 和[论文](https://arxiv.org/abs/2606.06880)，复现其中的 RISE-BM25 消融版本。下面分别说明沿用官方的设置和本次实验适配。

## 沿用官方的配置

| 项目 | 本仓库沿用的设置 |
|---|---|
| 检索与文档形式 | 全文档 BM25，k1=1.5、b=0.75；无 TOC，不切 chunk，不使用 embedding 或 reranker |
| 召回与预览 | 每个子查询 top-1000 召回、top-10 预览；召回文档累积到工作目录 |
| Prompt 与工具 | 官方无 TOC prompt，以及 search / bash / read 工具；bash 输出上限 4000 字符、子进程超时 60 秒，read 默认读取 2000 行 |
| 单次输出预算 | `max_tokens=32000`，来自官方代码的 `per_call_max_tokens` 默认值；这是单次模型调用的输出预算，不是整题 token 总量或上下文窗口大小 |
| 对话上下文管理 | `context_level="level0"`，不主动总结或清空历史；工具自身的输出截断仍然生效 |
| 时间预算 | 每题 3600 秒；沿用官方实现，在开始下一轮模型调用前检查时间预算 |
| 题目并发 | 默认同时进行 4 道 QA，与论文实验的并发数一致 |

一小时预算见[论文第 4 节](https://arxiv.org/html/2606.06880v1#S4)；单次输出预算和上下文管理以[官方配置](https://github.com/texttron/RISE/blob/32672b59ead8381decae7d412f05bb53d399946a/src/rise/protocol.py#L85)及 [Agent 实现](https://github.com/texttron/RISE/blob/32672b59ead8381decae7d412f05bb53d399946a/src/rise/agent.py#L120)为依据，不将这些代码默认值误写成论文逐项明确列出的参数。

一小时预算不会在请求或工具执行中途强行中断。例如第 59 分钟开始的调用在第 61 分钟返回，程序会在准备下一轮时发现超时。这种检查方式沿用官方实现；下述强制回答可能进一步增加整题耗时。

## 本次实验适配

- **模型与数据集**：使用 DeepSeek 和 EnterpriseRAGBench；支持 10,970 / 511,959 篇文档，两种规模使用同一组 500 道 QA。QA 温度设为 **0**。
- **轮次与预算耗尽后的回答**：常规模型调用上限设为 **50 轮**（论文为 100 轮），并开启强制回答。在轮次或时间预算耗尽后，必要时追加一轮不再执行工具的最终回答；论文主实验关闭该选项。因此每题最多 **51 轮：50 轮常规模型调用＋1 轮强制回答**，提前作答则提前结束。此上限按模型响应轮数计算，不按工具调用次数或 HTTP 重试次数计算。
- **HTTP 超时**：HTTP 客户端超时参数显式设为 **900 秒**。官方客户端没有显式设置该参数，其固定 SDK 版本默认读取等超时为 600 秒、连接超时为 5 秒。900 秒属于本次 API 服务适配；请求重试可能使一次模型调用的累计耗时更长。
- **评估与记录**：采用固定的 0–4 分 judge，统一统计 token、问答时间和 iteration，补充完整 trace、失败重试与断点恢复。指标定义见“结果与计时”。
- **运行环境**：使用 Linux `bubblewrap` 沙箱及固定版本依赖，适配服务器执行。
- **代码修复**：修复 `read` 的目录边界校验，保持该工具只能读取已召回工作目录内文件的限制。

## 安装

需要 Linux x86_64、Python 3.10、可用的 bubblewrap（原环境 0.6.1）以及 Bash / GNU 常用命令。以下命令均在本仓库根目录执行，虚拟环境需放在 `.venv`：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r adapters/requirements.txt
mkdir -p tools
curl -fL https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/ripgrep-14.1.1-x86_64-unknown-linux-musl.tar.gz -o tools/rg.tar.gz
echo '4cf9f2741e6c465ffdb7c26f38056a59e2a2544b51f7cc128ef28337eeae4d8e  tools/rg.tar.gz' | sha256sum -c -
tar -xzf tools/rg.tar.gz -C tools
cp tools/ripgrep-14.1.1-x86_64-unknown-linux-musl/rg tools/rg
bwrap --version
```

`bwrap` 需允许创建用户／网络命名空间；缺失时由服务器管理员安装。运行器会用它隔离模型执行的 bash 命令。

## 数据

**已有 prepared 数据时直接使用其目录，跳过下载和导出。** 目录须含 `corpus/`、`documents.jsonl`、`qa.jsonl`。运行前会校验文档正文哈希和问题集合；gold 和证据标注不会进入 Agent 的检索范围。

从公开数据准备：

```bash
.venv/bin/python -B reproduce.py download --out raw
.venv/bin/python -B reproduce.py prepare --raw raw --scale 10970 --out prepared/enterprise_10970
.venv/bin/python -B reproduce.py prepare --raw raw --scale 511959 --out prepared/enterprise_511959
```

来源固定为 [BM25-Wins-at-Scale](https://github.com/BstWPY/BM25-Wins-at-Scale/tree/0626d4f4019c6a04b891b7453dd93a3491999584) 的[公开语料包](https://huggingface.co/datasets/Bstwpy/BM25-Wins-at-Scale-Corpus/tree/a00241078f59719a06770a0d6eea10067690d76a)，下载时检查 SHA-256。输入 Parquet 约 1.38 GB，全量正文约 2.48 GB、511,959 个文件；运行还需保存输入副本、索引和 trace。

`data/` 只保存复现必需的文档 ID、QA 子集 ID 和校验值，不含历史实验结果。导出保留文档顺序、路径与正文，不转换正文中的字面量 `\n`；两篇 scaffold 文档沿用原导出 `.strip()` 约定。失败时保留 `.part` / `.building` 供检查，不自动覆盖。

## API 与运行

复制并编辑凭据文件，填写自己的服务地址和密钥：

```bash
cp credentials.example.json credentials.local.json
chmod 600 credentials.local.json
```

请求模型为 `deepseek-v4-flash`，响应模型检查值为 `deepseek-v4-flash-ga-260731`。API 需支持 OpenAI 兼容 Chat Completions、工具调用和完整 usage。使用不同模型或服务端版本时，应作为不同模型条件记录。

生成两个 500 QA 运行计划，默认并发 4。若使用已有数据，把 `--data` 换成相应绝对路径；该步骤不调用 API：

```bash
.venv/bin/python -B reproduce.py plan --scale 10970 --subset all500 --data prepared/enterprise_10970 --run-root runs/enterprise_10970_500qa --credentials credentials.local.json --rg tools/rg --out plans/enterprise_10970_500qa.json
.venv/bin/python -B reproduce.py plan --scale 511959 --subset all500 --data prepared/enterprise_511959 --run-root runs/enterprise_511959_500qa --credentials credentials.local.json --rg tools/rg --out plans/enterprise_511959_500qa.json
```

启动实验（会真实调用 API）：

```bash
mkdir -p runs/enterprise_10970_500qa runs/enterprise_511959_500qa
nohup .venv/bin/python -u -B adapters/experiment/launch.py --plan plans/enterprise_10970_500qa.json > runs/enterprise_10970_500qa/launcher.log 2>&1 < /dev/null &
nohup .venv/bin/python -u -B adapters/experiment/launch.py --plan plans/enterprise_511959_500qa.json > runs/enterprise_511959_500qa/launcher.log 2>&1 < /dev/null &
```

两个规模独立运行。启动器先执行真实 API / 工具预检，再完成正式问答和评分。查看各运行目录的 `state.json`、`run.log` 和数据集子目录的 `summary.json`。完成时应为 `accuracy_is_final=true` 且 `successful_qa=scored_questions=500`。

其他选项：`--concurrency` 设置题目并发；`--subset original100` 使用原固定 100 题；`remaining400` 使用其补集。更改配置应使用新的计划文件和运行目录。中断后，待原进程退出，用原计划重新启动即可跳过已完成题；评分失败只补评分。

## 结果与计时

```bash
.venv/bin/python -B reproduce.py summarize --runs runs/enterprise_10970_500qa
.venv/bin/python -B reproduce.py summarize --runs runs/enterprise_511959_500qa
```

汇总器逐题核查完整性；同一规模若分子集运行，可在 `--runs` 后传入多个不重叠的目录。

| 指标 | 口径 |
|---|---|
| accuracy | [固定 judge prompt](assets/judge.txt) 给出 0–4 分，取平均后除以 4；不是全部答对的题目比例 |
| 平均 token | 有效 QA attempt 的 API 输入（含缓存）＋输出；不重复添加已包含的推理 token |
| 平均问答时间 | 本题首次实际 HTTP 发送到最终完整回答返回；包含其间工具执行和重试，不含评分和开始前排队 |
| 平均 iteration | 模型响应轮数，包含最终回答轮；不是工具调用次数 |

执行完成但答错的题仍纳入统计；预检、失败尝试、judge 成本不计入主问答均值。完整 trace 保存在 `runs/<run>/enterprise_rag_bench_<scale>/qa/<题目ID哈希>/attempt_*/`，包括 `native_full.json`、`http.jsonl`、`tools.jsonl`、`retrieval.jsonl` 和 `workspace.jsonl`；上层 `result.json` 保存原题目 ID 和有效结果。

原执行器的 `summary.json.build_time` 只计 BM25。需要完整入库时间时，在实验结束、没有运行进程后执行：

```bash
.venv/bin/python -B reproduce.py measure --plan plans/enterprise_10970_500qa.json
.venv/bin/python -B reproduce.py measure --plan plans/enterprise_511959_500qa.json
```

读取 `runs/<run>/ingestion_timing_v2.json` 的 `total_ingestion_seconds`：重新生成 Parquet＋构建 BM25，不覆盖正式索引，并核对新旧产物和检索结果。不计下载、输入副本、预检与校验时间；不清空 OS 缓存。入库 API token 为 0。模型生成具有随机性，耗时也受服务负载和硬件影响。

## RSI 最小实验闭环

`adapters/rsi/run.py` 提供一个持续负责分析、诊断、修复和复盘的优化 agent。系统准备工作目录，agent 通过 Bash 读取原始轨迹、编写分析脚本、直接编辑 RISE，再调用既有评估器验证。没有预装优化措施，不修改仓库工作树；候选保存在独立目录。不做多 agent、候选组合或前缀续跑。

接口简化为三个工具：通用的 **`bash`**、专用计算工具 **`metrics`**、触发既有验证流程的 **`evaluate`**。移除了 `query/annotate/diagnose/submit`、强制引用登记、唯一片段校验、诊断 ID 和精确 old/new 替换协议。

| 方法 | 实现 | 边界 |
| --- | --- | --- |
| **可查询、可核验的证据反馈** | Bash 直接查看只读原始文件；`feedback.py` 从请求和工作区计算 S1–S6 观测 | 可回查事实，但不强制核验模型填写的引用；语义充分性、覆盖范围和必要性仍是模型判断 |
| **证据约束下的成本归因与选择** | `metrics` 提供实际 usage、逐要求驻留和内容携带账；agent 结合证据必要性选择改动，`evaluate` 给出配对结果 | 不再用登记流程强制“证据约束”；由提示词引导归因，最终用实测决定接受，不能把相关成本直接当作可省成本 |

`prompts.py` 保存统一提示词、工具 schema 和工作目录说明，不包含已知有效的具体优化配方。诊断可以写在普通笔记中，不必填表；出现可检验的机制后即可编辑代码并评估，不要求先穷尽证据标注。

运行时 Bash 看到的 `/workspace`：

```text
code/src/rise/       可直接编辑的代码副本（四个运行时文件）
reference/          当前已接受版本的完整代码，只读，可比较或恢复
data/               开发题、gold、成本排序、候选结果及 RSI 投入，只读
traces/<version>/   开发题原始记录与全部 attempt，只读挂载，不做摘要转述
documents.jsonl     文档 ID 到原始文件的映射，只读
corpus/             完整固定语料，只读
notes/              agent 自行编写脚本、提取数据、记录分析
metrics/            程序计算的完整证据／成本 JSON，只读
shell/              Bash 完整输出与执行记录，只读
logs/               候选开发阶段的运行日志，只读
```

`metrics()` 返回开发题成本概览；`metrics(qid=...)` 计算单题六态和成本，返回概要及完整 JSON 路径，长结果交给 Bash 自行过滤。默认跟踪已知 gold 来源，不假定它们涵盖全部等价证据。需要事实级分析时，可自行写 `notes/evidence.json` 并传 `evidence_file`，格式见工作目录 README；它是可选分析输入，不是提交门槛。S4 无片段时标为未评估，S5 无语义判断时为 unknown。`evaluate(summary=...)` 只需简述机制、预期效果及选择理由，直接冻结当前文件；拒绝后保留编辑内容供继续修改。

**前提**：已有当前代码和依赖版本下完成评分的基线 campaign，保留 prepared 数据、`corpus.parquet` 和完整 trace。准备一个 JSON 划分文件，包含非空的 `dev`、`selection`、`test` 三个题号数组，恰好覆盖基线计划的全部题目；可增加 `groups`（题号到事件组的映射）。程序拒绝题号重叠、相同问题、共享标注来源或显式事件组跨集合；未标注的事件关联仍需人工检查。人工深入分析过的问题应归入开发集，建议先用 20–30 道开发题。

```bash
# 离线初始化：复用原始基线结果，不调用 API。
.venv/bin/python -B adapters/rsi/run.py init \
  --plan plans/enterprise_10970_stratified100.json --split plans/rsi_split.json \
  --out runs/rsi_bash_pilot --max-calls 150 --max-candidates 3 --max-total-tokens 0

# 离线检查概览；加 --qid <开发题ID> 计算单题六态与成本。
.venv/bin/python -B adapters/rsi/run.py metrics --root runs/rsi_bash_pilot

# 以下两条均会真实调用 API；仅在 Linux + bubblewrap 环境执行。
.venv/bin/python -B adapters/rsi/run.py run --root runs/rsi_bash_pilot --execute
.venv/bin/python -B adapters/rsi/run.py test --root runs/rsi_bash_pilot --execute
```

- **模型与编辑范围**：问答及 judge 继续使用原基线模型。初始化时用 `--optimizer-model` 和 `--optimizer-response-model` 单独指定优化模型；默认与基线相同，不自动调用强模型。可修改 `agent.py`、`tools.py`、`rise_agent.py`、`retrieval.py`；评分器、预算配置、日志和 RSI 控制器不开放编辑。适配器中固定的参数不属于本版编辑范围。
- **证据与成本**：S1 已知来源未观察到可达，S2 可达，S3 来源路径可见，S4 所提供的片段可见，S5 证据充分性判断，S6 最终答案状态。S1–S4 是从工作区及实际请求得到的保守观测；不能将未匹配到原文等同于事实不存在，也不核验所填片段是否确实属于来源。逐请求记录实际 usage、事实可见变化与各项要求的状态驻留成本；不同要求的驻留账重叠，不能相加。携带账记录工具操作、首末可见请求、命中片段及物理重复，包含重试。`cl100k_base` 仅作 Token 估计，不冒充 DeepSeek 账单；不可用时只报告字符量。携带范围仅覆盖工具消息，不能代替整题成本，更不能直接代表可节省量。
- **验证与接受**：每项候选先做语法检查，再调用既有真实预检和 QA 执行器；开发集全体问题作为共同对照，达标后才跑选择集。默认质量均值不得低于最初基线，Token 比当前版本至少降低 5%；这是小规模试运行的点估计门槛，不是统计非劣性结论。选择集仅返回聚合结果；测试单独在冻结后启动。
- **隔离**：优化 agent 的 Bash 无网络、无凭据，只能写工作目录，不能读取选择／测试题；默认超时 120 秒，可通过工具参数调整。完整输出落盘，返回前 16,000 字符供后续自行筛选。候选 QA 仍在另一层 bubblewrap 中执行，仅挂载代码、去除 gold/证据/metadata 的本阶段问题、只读语料、索引、输出及 API 凭据；网络为 API 保留。真实评分在宿主执行，评分日志不挂载给候选，内部 bash 继续使用原沙箱。跨挂载点无法硬链接时，对召回文件做字节一致的复制，会增加 I/O；延迟不能视为与原基线完全同环境。这不是对恶意 Python 的完备安全审计，仍应检查候选代码有无硬编码或评估特判。
- **预算与恢复**：默认最多 150 次优化请求、3 项候选，连续两项未接受即停止；默认 `--max-total-tokens 0`，不设本地总 Token 上限，供应商报错不自动重试。仍保留 usage 审计，失败请求缺少 usage 时记为未知并停止，不按零计费。设置正数 Token 阈值时只在请求和评估阶段间检查，可能超出一个阶段的费用。Bash 超时会清理子进程；已完成的调用恢复时返回原记录，中断的调用不会自动重放。语法错误或无代码变化不消耗候选名额。签名漂移会拒绝，失败现场和 `.building` 不自动清理。
- **产物**：`workspace/` 保存代码与自由分析内容；`workspace_data/` 保存只读挂载的开发题概览；`metrics/` 保存完整计算及其可选分析输入；`shell/` 保存命令和输出。`versions/*/proposal.json`、`decision.json` 和隔离代码保存候选说明、配对结果与验证投入；`evaluations/` 保存运行、宿主评分和阶段耗时；`cache/` 保存每版本索引及构建时间；`budget.json` 分列优化、预检、QA、judge 的实际 usage。历史基线投入不计入新增预算，需另报。尚未自动核算美元、CPU 和存储费用，也未实现消融、独立 RSI 重复及置信区间。

RSI 文件使用独立 manifest，不改原执行器的代码签名；候选代码各自签名，不能混合汇总。Bash 版必须新建 campaign，不能接管旧工具协议的优化对话；旧实验需使用其冻结控制器继续。完成后 `frozen.json` 指向最终代码目录，不会自动合入原始 RISE。

## 发布范围与许可

以本目录作为 GitHub 仓库根目录，只提交代码、依赖清单、judge prompt、凭据示例、必要数据 ID／校验值和本 README。`.gitignore` 排除语料、索引、运行结果、trace、虚拟环境、二进制和个人凭据。

源码中保留了被运行器导入的共用模块和 `section_one_doc.py`；无 TOC 运行不会执行章节生成。RISE 代码许可见 [LICENSE](LICENSE)，数据许可见 [data/LICENSE](data/LICENSE)。
