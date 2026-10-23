# RAG-psy 项目状态（压缩版）

> 最后更新 2026-10-23 15:25。只留关键信息，细节见 git log 与 README。

## 一句话

心理学考研向 RAG 问答系统，全栈自研，检索链路与评估体系均已通，语料 17 本教材、2.17 万块入库，还剩两本扫描书因 MinerU 锁文件权限失败待补。

## 环境

| 项 | 值 |
|---|---|
| 数据库 | MySQL 3306 + Milvus 19530 + Redis 6379 全在 `192.168.34.52`（判断死活用端口探测，**不用 ping**） |
| 后端 | `uvicorn app.main:app --port 8000`，单实例（判据 = 谁在监听 8000，别数进程） |
| 前端 | `node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173`（`npm run dev` 在本机 git bash 下不监听端口） |
| 仓库 | github.com/Xenia-youyu/psych-rag，分支 main，最近提交 `b9fe0cf` |
| 栈 | FastAPI + Vue3 + Element Plus + Milvus dense-only 集合 `psych_chunks_v2` |

## 关键数字（均为实测）

```
代码     app 7,389 行 · tests 4,719 行 · frontend 3,834 行
接口     20 path / 29 ops
测试     361 passed（unit+api）· 集成 9 passed / 1 skipped
数据     31 条文档记录（17 done + 14 failed）· 21,704 块 · 约 26,000 向量
```

| 指标 | 值 |
|---|---|
| 黄金集 | 70 条（可用 66）：concept 26 / compare 12 / multihop 13 / refuse 15 |
| Recall@10（按段） | concept 0.962 · compare 0.750 · multihop 0.808 |
| MRR | 0.849 / 0.833 / 0.923 |
| Hit@5 | 0.962 / 0.833 / 1.000 |
| 问答延迟（热态） | 检索 4.2–5.0s，首 token 8.4–10.8s，总 12–14s |
| 入库 | 首批 2,027 页 23 分钟；OCR 9 秒/页；向量化 0.25 秒/条（百炼 API） |

## 四条决定链（每条都是拿数据换的，别凭直觉改回去）

**1. 向量化走百炼 API，不走本地**
本地 BGE-M3 在这台 4 物理核机器上 3.7 秒/条，4.8 万页 = 257 小时。text-embedding-v4 降至 0.25 秒/条，全量约 15 元。provider 开关在 `config.ini [embedding]`，随时可切回本地。

**2. 重排喂子块，父块只在选中后回填**
代码原注释说「先回填父块再重排更好」，实测反了：金标平均分 子块 0.7043 / 父块 0.6472，且 c006/c011/c016 三道题的正确块从第 1 名被打到 13–15 名。改后 Hit@5 0.679 → 0.893。
链路顺序：`去重 → 重排(子块) → 阈值 → 取前N → 回填父块 → 按预算再裁`。

**3. 重排阈值 0.54（不是 0.58，更不是模板里的 0.15）**
0.58 是拿 3 条查询拍的，实测会砍掉排 top1 的正确答案（c021 分数 0.5464）。标定脚本 `scripts/calibrate_threshold.py`：0.58 → 金标保留 69% / 拒答误收 27%；0.54 → 77% / 33%。四处已对齐（config.ini / .example / config.py 两处默认值）。

**4. 黄金集 Recall 按「父块覆盖」算，不按子块**
金标会扩展同父块兄弟（3–5 个），命中任意一个即拿到同一段上下文。旧算法算的是"命中几个兄弟"，把 0.96 显示成 0.29。**旧数字与新的不可比。**

## 必须先记住的七个坑

1. **推理模型吃 token**：deepseek-flash 的 token 先花在思考上。`max_tokens=256` 时 content 是空串、`max_tokens=900` 时 JSON 从中间截断。凡是要模型回 JSON，用 `settings.llm.active().max_tokens_json`（2048），别写死。
2. **MinerU 三个坑**：必须用文件入口（`python -m`，不能 `python -c`）；文字在 `block.content` 的 span 列表里（不是 `block.text`）；输出 markdown 内嵌 base64 图片要 `strip_data_uri()` 剥掉。
3. **一个坏 OCR 框杀死整本书**：`warp_image` 遇 `min(size)<=0` 抛 ValueError，整本返回 0 页。用 `--chunk-pages 64` 分段落库隔离。
4. **`kill` 父进程不等于杀死任务**：曾漏杀子进程跑 4 小时，与新任务交叉写入造成重复文档。杀进程脚本落成文件，别用 `python -c`（会匹配到自身命令行）。
5. **Milvus `query(limit=16384)` 静默截断**：集合 26k 行时按全表核对会造假不一致。核数量前先读 row_count，再按 doc_id 逐个查。
6. **`export DASHSCOPE_API_KEY=x` 会覆盖真 key**（环境变量优先于 `[secrets]`）。跑入库/清洗脚本时不要设这个变量。
7. **pyproject 抽取会带字符间距**：`学 习 的 结 果` 这种块占过 19.6%（教育心理学 82%），向量全散。`cleaner.repair_text()` 统一处理（去水印 + 去 CJK 间距）。

## 当前阻塞：MinerU 锁文件权限

恢复任务的 2/3、3/3 全部秒失败：

```
PermissionError: [WinError 5] 拒绝访问。
  'C:\Users\86188\.mineru\models\.locks\MinerU-4_models_onnx.lock'
```

前一个被强杀的 MinerU 进程留下的僵尸锁，单段耗时 0.6–4.3 秒即失败（正常应 5 分钟/64 页）。

**修法**：删掉 `~/.mineru/models/.locks/` 整个目录，再重跑两本：

```bash
rm -rf ~/.mineru/models/.locks
python -m scripts.ingest_plan --batch batch2 --match 测量 --chunk-pages 64 --report data/recover_measure.json
python -m scripts.ingest_plan --batch batch2 --match 统计学 --chunk-pages 64 --report data/recover_stat.json
```

预期：测量 363 页 ≈ 33 分钟，统计 334 页 ≈ 30 分钟。

## 待清理

- 14 条 `failed` 文档记录，其中 12 条是上面的分段失败（0 页 0 块 0 向量，删了无损失）。**删之前先确认没有进程在写它**（曾误删正在入库的书）。

## 下一步（按优先级）

1. 清锁 → 补齐测量、统计两本
2. 重测 intent 修复后的真实延迟（现在的数字被 OCR 抢 CPU 抬高过）
3. 用黄金集测「加入扫描件之后」检索质量变化
4. 对比题只拿到一边材料（75%），试多 query 召回（`rewrite.py` 的 subquery/stepback 在评估里没启用，指标测的是单 query 上限）
5. 解析任务队列化（现在 `BackgroundTasks`，一本 300 页扫描件就阻塞 worker）

## 常用命令

```bash
# 环境自检
python -m app.check_env
# 评估（含阈值 A/B）
python -m scripts.eval_run --types concept,compare,multihop
python -m scripts.eval_run --threshold 0.52
# 阈值标定
python -m scripts.calibrate_threshold
# 逐段追踪 gold 在哪一步丢失
python -m scripts._debug_retrieve 20
# 清库前必看：哪条 doc 的向量在集合里
python -m scripts.clean_existing          # 默认 dry-run
# 后端单实例校验 / 停止
python scripts/_verify_backend.py
python scripts/_stop_backend.py
```

## 已知限制（不打算修的）

- **c021**：金标去重后排第 13，被 `rerank_max_candidates=12` 截掉。捞回需抬上限 + 压阈值，一道题换每问多 0.8s、误收 33%→53%，不值。
- **父块不进 MySQL**：只存 Milvus，且是对全文独立切的，无法从子块精确重建。所以向量化失败只能整本重跑。
- **Element Plus 全量引入**，主包 1.27MB（gzip 411KB）。
