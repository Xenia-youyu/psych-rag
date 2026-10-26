# RAG-psy 项目状态（压缩版）

> 最后更新 2026-10-23 21:20。只留关键信息，细节见 git log 与 README。

## 一句话

心理学考研向 RAG 知识问答系统。19 本教材 / 25,291 块入库，检索正确段命中率 96%，有基线可做回归。
**范围：只做知识检索，不做咨询分流**（决定见下）。

## 环境

| 项 | 值 |
|---|---|
| 数据库 | MySQL 3306 + Milvus 19530 + Redis 6379 全在 `192.168.34.52`（判活死用端口探测，**不用 ping**） |
| 后端 | `uvicorn app.main:app --port 8000`，单实例（判据 = 谁在监听 8000，别数进程） |
| 前端 | `node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173`（`npm run dev` 在本机 git bash 下不监听端口） |
| 仓库 | github.com/Xenia-youyu/psych-rag，main 分支 |
| 栈 | FastAPI + Vue3 + Element Plus + Milvus dense-only 集合 `psych_chunks_v2` |

## 关键数字（均为实测）

```
代码     app 7,594 行 · tests 4,829 行 · scripts 4,280 行 · frontend 3,834 行
接口     20 path / 29 ops
测试     371 passed（unit + api）
数据     19 本教材 / 58 个文档记录 / 25,291 块 / 33,289 条向量
```

| 指标（黄金集 51 条，阈值 0.54，重排窗口 20） | 值 |
|---|---|
| concept（26 条） | Recall@10 **0.962** · MRR 0.745 · Hit@5 0.962 |
| compare（12 条） | 0.792 · 0.792 · 0.833 |
| multihop（13 条） | 0.846 · 0.923 · 1.000 |
| 问答延迟（热态） | 检索 4.2–5.0s，首 token 8.4–10.8s，总 12–14s |
| 入库 | 首批 2,027 页 23 分钟；OCR 9 秒/页；向量化 0.25 秒/条 |

**生成侧（2026-10-26 首测，65 条）**

| 类型 | 条数 | 要点过关 | 要点命中率 | 均引用 | 悬空引用 |
|---|---|---|---|---|---|
| concept | 26 | 23/26 | 87% | 4.1 | 0 |
| compare | 12 | 11/12 | 92% | 3.6 | 0 |
| multihop | 13 | 13/13 | 100% | 4.2 | 0 |
| refuse | 14 | — | — | 2.5 | 0 |

- 知识题合计 51 条：要点过关 **47/51（92%）**，平均命中率 91.5%，平均正文 1,001 字
- 拒答：**13/14（93%）** 明确声明「未覆盖」，其中只有 3 条是纯模板拒答
- 空答案 0（重试机制已加），悬空引用 0

基线文件：`data/golden/baseline.json`（检索）、`data/golden/generation_baseline.json`（生成）。
改任何参数后跑 `eval_run` / `eval_generation` 自动对比。

## 范围决定（2026-10-23）

**只做知识检索，不做咨询分流。** 完整记录在 `data/corpus_plan.json` 的 `scope_decisions`。

| 做 | 不做 |
|---|---|
| 教材知识问答（概念 / 对比 / 多跳 / 拒答） | 咨询语料入库（`咨询数据`、PsyQA 都不入） |
| 引用回溯到书名与页码 | 情绪困扰与心理支持对话 |
| 检索质量评估与回归 | 按意图分流的 triage 路由 |

**不做的原因**：`咨询数据` 授权链不完整且其说明明确禁止第三方 embedding（本项目走百炼 API），
含未脱敏个人信息，且任务错位——教材知识点 vs 个人倾诉，混入是污染不是增强。

**遗留风险（已知限制）**：`app/generate/safety.py` 仍在链路里但覆盖不足，
40 条危机输入只拦下 3 条（明示 15% / 隐晦 0%），7 条会被当成知识题作答。
按本次决定不加固；**若将来面向真实用户开放，必须先补**。评估集在 `data/golden/triage.jsonl`。

## 四条决定链（每条都是拿数据换的，别凭直觉改回去）

**1. 向量化走百炼 API，不走本地**
本地 BGE-M3 在这台 4 物理核机器上 3.7 秒/条，4.8 万页 = 257 小时。text-embedding-v4 降至 0.25 秒/条，全量约 15 元。`config.ini [embedding] provider` 可切回 local。

**2. 重排喂子块，父块只在选中后回填**
金标平均分 子块 0.7043 / 父块 0.6472，且 c006/c011/c016 的正确块从第 1 名被打到 13–15 名。改后 Hit@5 0.679→0.893。
链路：`去重 → 重排(子块) → 阈值 → 取前N → 回填父块 → 按预算再裁`。

**3. 阈值 0.54 + 重排窗口 20**
0.58 会砍掉排 top1 的正确答案（c021 分数 0.5464）。0.52 能救 m001 但拒答误收 33%→53%，不值。
窗口 12→20 救回 c009（gold 在去重第 13 位）。两处已四处同步（config.ini / .example / config.py ×2）。

**4. 黄金集 Recall 按「父块覆盖」算**
命中同父块任意兄弟即算命中。旧算法按子块比，把 0.96 显示成 0.29。**旧数字与新不可比。**

## 必须先记住的八个坑

1. **推理模型吃 token**：deepseek-flash 的 token 先花在思考上。`max_tokens=256` 时 content 空串、`900` 时 JSON 截断。要 JSON 就用 `settings.llm.active().max_tokens_json`（2048）。
2. **MinerU 三个坑**：必须文件入口（`python -m`，`python -c` 会 OSError）；文字在 `block.content` 的 span 列表（不是 `block.text`）；markdown 内嵌 base64 要剥。
3. **一个坏 OCR 框杀死整本书**：`warp_image` 遇 `min(size)<=0` 抛错。用 `--chunk-pages` 分段落库，再逐级细分到单页定位。
4. **`kill` 父进程不等于杀死任务**：曾漏杀子进程跑 4 小时。杀进程脚本落成文件，别用 `python -c`（会匹配自身）。
5. **Milvus `query(limit=16384)` 静默截断**：集合 33k 行时按全表核对会造假不一致。先读 row_count，再按 doc_id 逐个查。
6. **`export DASHSCOPE_API_KEY=x` 会覆盖真 key**（环境变量优先于 `[secrets]`）。跑入库/清洗时不要设。
7. **PyMuPDF 抽字符间距**：`学 习 的 结 果` 占过 19.6%。`cleaner.repair_text()` 统一处理（短语水印 → 行水印 → HTML 表格 → 中文间距）。
8. **阈值类规则必须抽看样例**：删垃圾块的规则「<80 字」差点误杀 76 字的正常句子，改成「短且汉字 <20」才准（724→630）。

## 已知限制（不打算修的）

- **3 页永久缺失**：测量 p83、统计 p54、统计 p216（OCR 检测框几何退化，MinerU 库 bug）
- **58 个文档记录里有 29 个是分段碎片**（如「（89-96页）」），功能正常但界面不好看
- **c021 / m001 / m007** 三条黄金集未命中：分别因重排截断、阈值、rerank 排序波动，捞回的代价大于收益
- **安全护栏覆盖不足**：见范围决定里的遗留风险
- **父块只存 Milvus 不存 MySQL**，且对全文独立切分，无法从子块精确重建 → 向量化失败只能整本重跑

## 下一步（按优先级）

1. 重测 intent 修复后的真实延迟（OCR 已结束，之前的 19.8s 是被抢 CPU 抬高的）
2. 对比题只拿到一边材料（0.792），试多 query 召回（`rewrite.py` 的 subquery/stepback 在评估里没启用，指标测的是单 query 上限）
3. 解析任务队列化（现在 `BackgroundTasks`，一本 300 页扫描件就阻塞 worker）
4. 生成侧指标（引用准确率、拒答正确率）—— 目前只测了检索侧

## 常用命令

```bash
python -m app.check_env                       # 环境自检
python -m scripts.eval_run --types concept,compare,multihop   # 检索评估
python -m scripts.eval_run --threshold 0.52   # 阈值 A/B，不改配置
python -m scripts.calibrate_threshold         # 阈值标定
python -m scripts._debug_retrieve <行号>       # 逐段追踪 gold 在哪一步丢的
python -m scripts.clean_existing              # 存量清洗（默认 dry-run，--apply 执行）
python scripts/_verify_backend.py             # 后端单实例校验
python scripts/_stop_backend.py               # 停后端
```
