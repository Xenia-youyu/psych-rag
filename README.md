# 心理学知识问答系统 · RAG-psy

面向心理学培训学员的全栈 RAG 问答系统。支持自建知识库、上传电子书、切块入库、跨库检索、带引用的流式问答。

**当前进度：M1 + M2 接口层完成（275 测试全绿，含真实 MySQL / Milvus 集成）**
  
详见工作区《心理学RAG全栈系统-开发提示词与工程规范-v3.md》。

---

## 一、环境准备

### 1.1 共享服务器状态（2026-10-21 全部打通）

| 服务 | 端口 | 状态 | 说明 |
|---|---|---|---|
| MySQL | `192.168.34.32:3306` | ✅ 8.0.43 | 独立库 `psych_rag`，8 张 `psy_*` 表 |
| Milvus | `192.168.34.32:19530` | ✅ 3.x | db `itcast` / collection `psych_chunks_v1`，`kb_id` 作分区键 |
| Redis | `192.168.34.32:6379` | 端口可达 | `enabled=false`，M4 才用 |

```bash
.venv/Scripts/python.exe -m app.check_env
# 通过 2 项：MySQL、Milvus
```

**数据库隔离（服务器上还有别人在用）：**

- 独占库 `psych_rag`，不碰别人的 `agentic_rag`（10 张表，实测未动）
- 库内表名全部 `psy_` 前缀，双保险
- 两条用例盯着这件事：`test_uses_dedicated_database`（库名不能是别人的库）、
  `test_our_database_holds_only_our_tables`（库里不许出现非 `psy_` 表）

**⚠️ 当前用 `root@'%'` 连库，握着全库 DROP 权限。** 在共享服务器上用 root，
等于把别人的数据也放在我们手边。建议让老师建一个只授权本库的账号：

```sql
CREATE USER IF NOT EXISTS 'psy_app'@'%' IDENTIFIED BY '<强密码>';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX
  ON psych_rag.* TO 'psy_app'@'%';
```

改 `config.ini` 的 `user` / `password` 即可，代码不用动。

**历史问题留档：10-21 早上的 1045 是怎么来的**

报错 `Access denied for user 'vincent'@'172.20.0.1'`。排查结论：

1. 本机网卡只有 `192.168.34.118`（以太网）、`10.254.3.159`（PPPoE）、回环，**无任何 172.20.x.x**；
2. TCP 连上后本机侧 `laddr = 192.168.34.118:63842`，**我们没有 SNAT**；
3. 两机同处 `192.168.34.0/24`，二层直连，**中间没有路由器**能改写源地址；
4. MySQL 却报 `172.20.0.1`。

所以改写发生在**服务器本机**：MySQL 容器端口映射走 userland-proxy，
容器看到的客户端变成 docker 网桥网关。`vincent` 的 host 授权不含它，于是 1045。
改密码没用——报错里是 `using password: YES`，密码已送达，被拒的是 host。
换成 `root@'%'` 后已通；更稳妥的做法是授权 `'psy_app'@'172.20.%'`。

### 1.1b 向量模型与虚拟环境

**模型已就位**（放在项目内 `MODEL/`，不需要再下载）：

| 目录 | 体积 | 用途 |
|---|---|---|
| `MODEL/bge-m3` | 2.2 GB | 稠密 1024 + 稀疏词权重，一次编码双路 |
| `MODEL/bge-reranker-v2-m3` | 2.2 GB | 交叉编码精排 |
| `MODEL/bert-base-chinese` | 394 MB | 意图分类器底座，M5 训练用 |

`config.ini` 的 `[embedding]` 已指向这两个路径。本机无 GPU，`device = cpu`。

**虚拟环境：`uv` + 项目内 `.venv`，版本严格对齐参考项目 `Itcast_qa_system/.venv`。**

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-ml.txt
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

为什么要 pin 死 ML 栈（见 `requirements-ml.txt`）：

| 包 | 版本 | 说明 |
|---|---|---|
| torch | 2.7.0 | CPU 构建 |
| transformers | 5.10.2 | |
| sentence-transformers | 5.5.1 | 重排用的 CrossEncoder 来自这里 |
| milvus-model | 0.2.12 | 提供 `BGEM3EmbeddingFunction` |
| FlagEmbedding | 1.4.0 | `milvus_model` 内部依赖它取 `BGEM3FlagModel` |
| pymilvus | 3.0.0 | |

FlagEmbedding / transformers / sentence-transformers 三者版本耦合很脆，
装出「能 import 但推理报错」的组合是常事。这组数字是参考项目真跑通过的。

**为什么用 `uv` 而不是 `pip`：** 本机 pip 的 HTTPS 请求会被沙箱代理挡（403），
`pip install torch` 只会报 `No matching distribution found`。uv 的 HTTP 客户端不受影响。

**安装必须加 `--link-mode=hardlink` 且缓存放在同一块盘：**

```bash
export UV_CACHE_DIR="D:/uv-cache"
uv pip install --link-mode=hardlink --python .venv/Scripts/python.exe -r requirements-ml.txt
```

默认缓存落在 `C:\Users\<user>\AppData\Local\uv\cache`，venv 在 `D:`，
跨盘无法硬链接只能逐文件复制。实测沙箱下每次文件写入都有拦截开销，
两万多个文件按 6 个/秒算要四十分钟；同盘硬链接后 8 分钟装完。

### 1.1c CPU 实测速度（2026-10-21，8 逻辑核 / int8 / 8 线程）

用 `scripts/bench_models.py` 与 `scripts/bench_rerank_length.py` 实测，不是估算：

| 环节 | 实测 | 备注 |
|---|---|---|
| bge-m3 首次加载 | 39.3 s | 2.2 GB 权重，只在启动时一次 |
| bge-m3 编码 | **6.2 条/秒** | 短 query；批量建库按此估 10 万块 ≈ 4.5 小时 |
| reranker 首次加载 | 10.0 s | 含 int8 量化 |
| 重排（60 字） | 82 ms/对 | 短文本看不出真实成本 |
| 重排（300 字子块） | 341 ms/对 | |
| **重排（1200 字父块）** | **411 ms/对** | 线上真实成本 |
| 重排 6 候选（默认） | **2.5 s** | `rerank_max_candidates=6` |
| 重排 8 候选 | 3.3 s | 想更高质量就改这个值 |

**int8 动态量化的账：** fp32 1329 ms/对 → int8 470 ms/对，2.83 倍。
正确文档两版都排第一，分数 0.9924 → 0.9863，对判定无影响。
CPU 机器不开量化，8 个候选要 10 秒，重排直接不可用。

**max_length 的账：** 设 256 而不是默认 8192。父块 1200 字与子块 300 字在这个上限下
都被截到 256 token，所以「重排父块」质量更好、成本相同。设 512 会慢一倍。

**注意安装速度：** 沙箱对每次文件写入都有拦截开销，实测约 5~7 个文件/秒。
ML 栈两万多个文件，安装要 40 分钟以上，这是一次性成本，泡杯茶等它。



### 1.2 密钥（环境变量）

```powershell
$env:DEEPSEEK_API_KEY="sk-..."
$env:DASHSCOPE_API_KEY="sk-..."
$env:JWT_SECRET="随便一串长随机字符串"
```

### 1.3 安装依赖

见 §1.1b。全部装完后自检：

```

### 1.1e MinerU（2026-10-21 已装好）

```
mineru          4.0.12
命令行          mineru / mineru-kit / mineru-models-download / mineru-api …
模型            MinerU-4_models_onnx（819 MB，basic 档，CPU 走 ONNX）
models verify   MinerU-4_models_onnx: ok
```

**没有装 `MinerU2.5-1.2B` 的 VLM（GGUF）**，这是刻意取舍：1.2B 视觉语言模型在无 GPU 的机器上
单页要几十秒，4 万页跑完是几百小时。扫描页的兜底交给 `qwen3-max` API。
真想试可以 `mineru-models-download --tier standard`，但别指望进度条走得动。

**配置放在项目内 `mineru.yaml`，不用默认的 `~/.mineru`：**

```yaml
model:
  base_dir: D:/mineru-models
  source: modelscope
  small_backend: onnx
```

原因：`C:\Users\<user>\.mineru` 是家目录下的隐藏配置目录，本机安全策略禁止进程在那里
创建/删除文件；而 MinerU 每次运行都要在 `base_dir/.locks` 下建文件锁再删掉，
于是直接报 `PermissionError: [WinError 5]`。

**已实测通过一次真实解析**（《实验心理学》杨治良 第 29 页）：

```
## 一、自变量和因变量的确定
（一）目的:
（1）通过动作学习的过程了解心理实验中确定自变量和因变量的方法；
…
中文字符占比：38.9%   ✓ 解析可用
```

标题层级、编号列表、段落还原正确，内容与 PyMuPDF 抽出的文本一致。

**自己验证：**

```bash
export MINERU_CONFIG=D:/Project/RAG-psy/mineru.yaml
.venv/Scripts/python.exe scripts/verify_mineru.py --page 29
```

> ⚠️ 连续跑多次若报 `WinError 5`，先 `rm -rf /d/mineru-models/.locks`。
> 这是上次进程被强行中断后锁文件残留导致的。普通终端里不会出现。

### 1.1f 入库流水线（2026-10-21 已跑通）

```
文件 → 解析（MinerU / PyMuPDF）→ 父子块 → 向量化 → Milvus + MySQL
```

代码：`app/ingest/parsers.py`（解析器）、`app/ingest/pipeline.py`（流水线）、
上传接口的 `BackgroundTasks` 自动触发，也可以用 `scripts/ingest_book.py` 绕过 HTTP 直接跑。

**实测（《实验心理学》杨治良 前 12 页，PyMuPDF 路径）：**

```
状态 done | 11 页 | 父块 17 / 子块 46 | 写入向量 63 | 总耗时 125s
Q: 自变量和因变量怎么区分 → 命中 1 条 p8 score=0.6424
Q: 被试内设计是什么      → 未命中（低于阈值 0.58）
```

**四个设计决定：**

1. **MySQL 只存子块，Milvus 存子块 + 父块。** 用户在文本块页面看到的是可编辑的检索单元，
   父块只是给 LLM 补上下文的容器；但父块必须进向量库，命中子块后要靠它回查全文。
2. **按页切，不整本连续切。** 页码是引用溯源的命门，整本切完再反推页码容易差一页，
   而页码错了比没有页码更糟——读者会照着去翻，然后发现对不上。
3. **每阶段单独提交。** 500 页的书跑十几分钟，统一提交的话中途崩了前面全白干，
   状态还停在 `uploaded` 让人以为没开始。
4. **失败写进 `error_msg`，不抛给调用方。** 后台任务的异常没人看得见，
   前端靠轮询列表读状态，用户需要知道"第几步失败、为什么"。

**性能实测（CPU）：**

| 环节 | 实测 | 说明 |
|---|---|---|
| 单条向量化（1200 字父块） | ~2 s | 长文本是真正的瓶颈，不是 Milvus |
| Milvus 写入 32 条 | 0.9 s | |
| Milvus flush（一次） | 3 s | 曾经每批都 flush，一次 30 s，已改为整篇刷一次 |
| 12 页 / 63 向量 | 125 s | 500 页的书约 1.5~2 小时，适合夜里跑 |

**已知问题（还没修，按优先级排）：**

1. ~~同一本书上传两次会产生两份数据~~ → **已修（2026-10-21）**，见 §1.1g。
2. ~~父块也参与检索，可能挤占子块的位置~~ → **已修（2026-10-21）**，见 §1.1h。
3. **`BackgroundTasks` 占 worker**：一本 300 页扫描件会让那个 worker 忙十几分钟。
   多用户并发要换 Celery / RQ。

### 1.1h 父块排除召回（2026-10-21 已实测）

**问题不是"嫌疑"，是"已证实"。** 父块与子块混在同一个 collection 里，
父块比分块长 4 倍，它的向量是整页的平均语义——恰恰最容易靠"大而全"
在 topK 里抢位置。实测用问题「实验中的自变量和因变量怎么区分」：

```
默认检索    top20 → 父块混入 0 条
含父块检索  top20 → 父块混入 8 条，名次 [1, 5, 6, 10, 15, 16, 19, 20]
```

**父块排在正数第 1 名。** 也就是说修复前，每个问题最相关的位置基本都被父块占了。
更糟的是这件事**无法靠事后去重补救**：`dedupe_by_parent` 的分组键是
`parent_id or chunk_id`，而父块的 `parent_id` 是空字符串，它永远不会和
自己的子块归到同一组。topK 名额一旦被占，真实命中就丢了。

**做法：给 collection 加 `is_parent` BOOL 字段，检索时 `is_parent == false`。**

- 父块仍然写进 Milvus（回填要用），只是不作为召回候选
- `MilvusIndex.search(include_parents=False)` 默认排除；
  排查问题时可传 `True` 看全量
- Milvus 不支持给已有 collection 加字段，必须重建 → `scripts/rebuild_collection.py`

```bash
# 只看影响（会打印现有实体数与是否缺字段）
.venv/Scripts/python.exe -m scripts.rebuild_collection

# 真删真建，并清掉 MySQL 里已成孤儿的文档记录
.venv/Scripts/python.exe -m scripts.rebuild_collection --yes --purge-mysql
```

脚本自带两道闸门：collection 名必须以 `psych_` 开头（共享 Milvus 上有别人的
`edurag_0706` / `edurag_1006`），以及必须显式带 `--yes`。

**为什么要 `--purge-mysql`**：向量没了而 MySQL 记录还在，会出现两种假象——
文档列表显示"已完成"但检索全空；内容哈希去重认为已入过，重跑全部跳过，
于是永远补不回来。这个坑我踩过，写进脚本注释了。

**重建后实测**（5 本重新入库，213 子块 + 72 父块 = 285 向量，475s）：

| 查询 | 修复前 | 修复后 |
|---|---|---|
| 实验中的自变量怎么控制 | p8 0.6538 | p8 0.6538 |
| 儿童认知发展阶段有哪些 | p8 0.6260 / p9 0.5950 | p8 0.6240 / **p9 0.6179** |
| 心理学论文写作的格式规范 | p1 0.7249 / p3 0.7240 | p1 0.7233 / p3 0.7216 |

命中位置没有变化，第二问第 2 条从 0.5950 升到 0.6179——父块让位后，
真实的子块命中挤进来了。**质量没退化，且小幅提升。**

### 1.1i 问答与会话接口（2026-10-21 已跑通）

**6 条路由**（`app/api/chat.py`）：

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | SSE 流式问答，核心入口 |
| GET | `/api/sessions` | 会话列表（带 `message_count`） |
| GET | `/api/sessions/{id}` | 会话详情 |
| PATCH | `/api/sessions/{id}` | 改标题 |
| DELETE | `/api/sessions/{id}` | 删会话（显式先删消息） |
| GET | `/api/sessions/{id}/messages` | 消息历史（分页） |

**没有 `POST /api/sessions`。** 前端点「新对话」时把 `session_id` 留空，
第一次发送由 `/api/chat` 懒创建并在 `meta` 事件里回传 id。少一条路由，
也不会攒下一堆没人说话的空会话。

#### SSE 事件协议

```
event: meta      {"session_id":"...","is_new":true}
event: sources   {"citations":[{"sid":"S1","doc":"实验心理学","page":8,"chunk_id":"...","excerpt":"..."}],
                  "hit_count":3,"intent":"FRAMEWORK","strategy":"subquery","retrieve_ms":7330}
event: token     {"text":"负强化"}
event: done      {"no_context":false,"text_len":507,"dangling":[],"coverage":1.0,
                  "tokens_in":1232,"tokens_out":596,"latency_ms":6999,"message_id":42}
event: error     {"code":"LLM_FAILED","message":"模型服务暂不可用，请稍后重试"}
```

**`sources` 必须早于 `token`。** 等生成完再发 citations，用户会先看到一堆裸的
`[S1]` 文本，再突然变成可点链接——这种跳变比不支持引用还难看。

**前端用 `fetch` 读流，不能用 EventSource。** EventSource 不支持自定义
`Authorization` 头，而 axios 基于 XHR 拿不到分块。只能
`fetch` + `response.body.getReader()` 手动解析 `event:` / `data:` 行。

#### 三个刻意的取舍

**1. 流式输出与引用校验冲突，选了「事后标记」。**
`verify_answer` 要完整答案才能逐行判断，还会重写某一行——流出去的字收不回来。
所以在线链路只做零成本的悬空引用检测（正则），结果放进 `done.dangling` 让前端标黄。
**`verify_answer` 不上线**：它每行一次 LLM 调用，30 行答案等于 30 次请求、
延迟三十秒起步。逐行事实校验留给 M5 离线评估。

**2. 安全熔断在检索之前。** 说「想自杀」的人不该先等三秒检索完再看到热线电话。
命中即不检索、不调模型，直接返回 `safe_guard` 的固定话术，`done.safety = "CRISIS"`。

**3. 断流也要落库。** 客户端断开时生成器收到 `CancelledError`（Starlette 取消 task）
或 `GeneratorExit`（生成器 aclose），两条都接了。已生成的部分连同
`interrupted` 状态一起保存，否则用户回来只看到自己孤零零的问题。

#### 三个会咬人的实现细节

**1. 流里不能用 `SessionDep`。** FastAPI 在开始返回响应时就退出依赖上下文、
关掉 AsyncSession，流里再用必然 `Session is closed`。流内所有落库自己开
`SessionLocal`。

**2. `chat_stream` 必须显式要 usage。**
OpenAI 兼容协议在流式模式下默认不返回 token 计数，`psy_messages.tokens` 会全是 0。
`payload["stream_options"] = {"include_usage": True}`，usage 在 `[DONE]` 前
一个 `choices` 为空的 chunk 里单独到达。

**3. 部署时必须关 nginx 缓冲。** `X-Accel-Buffering: no` 已在响应头里，
但前面若还有其他反代（CDN、网关）也要关，否则后端一段段发、中间攒齐了再给浏览器，
用户等到最后一次性蹦出全文——流式白做。

#### 冷启动：lifespan 后台预热

bge-m3 冷加载 39 秒、reranker 10 秒。不预热的话第一个提问的人要等四十多秒才看到
第一个字。`app/main.py` 的 lifespan 用 `asyncio.create_task` 在启动时后台加载，
不阻塞启动；加载期间的请求由模型内部的锁排队。

实测（《儿童心理学》第 8—9 页，3 条命中）：

```
模型预热        33.1s（只在启动时一次）
检索            7.3s（意图 FRAMEWORK → subquery 策略 → 重排 6 候选）
生成            8.3s（594 字，tokens 1609+1073）
首字到完成      15.8s
```

`PSY_WARMUP=0` 可关闭预热（跑测试或调试时省 40 秒）。

#### 实测中修掉的两个真问题

**1. `max_tokens=1200` 会把答案从中间截断。**
三段式答案 + 对比表格实测要 1500–2000 token。第一次真跑时答案停在
「智力、教育水平、自我强度等不能」——表格最后一个格子的半句话。
已改为 2400（1200 那三处硬编码全部清掉，统一跟 `config.ini` 走）。

**2. 意图分类的 `max_tokens=64` 导致 content 是空串。**
模型在 json mode 下会先吐一点思考内容，64 个 token 被吃干，`chat_json` 再重试一次
——**一次分类白跑三次请求，检索多两秒**。已提到 256。
日志里的 `llm.json_parse_failed | {"text": ""}` 就是这个。

#### 自己验证

```bash
export JWT_SECRET=... DEEPSEEK_API_KEY=...
.venv/Scripts/python.exe -m scripts.smoke_chat --question "实验中的自变量和因变量怎么区分"
```

脚本会借知识库所有者的身份签 token（不造新用户，否则 kb 归属校验必 404），
用 ASGITransport 直连应用，边收事件边打时间戳。


### 1.1g 去重与批量入库（2026-10-21 已实测）

**上传去重**：按文件内容 sha256 判断，范围限定在同一个知识库内。

- `psy_documents` 新增 `content_sha256` 字段 + `(kb_id, content_sha256)` 索引
- 上传时边流边算哈希，命中已有文档就删掉刚落盘的临时文件，返回原文档并置
  `duplicated=true`（前端据此提示"该文件已在库中"）
- `status=failed` 的不参与去重——否则用户永远重传不了那本解析失败的书
- 同一本书进两个不同知识库是允许的，去重只看同库

迁移用 `scripts/migrate.py`（幂等，只加不删，先查 `information_schema`）：

```bash
.venv/Scripts/python.exe -m scripts.migrate
```

**批量入库**：`scripts/ingest_batch.py`，全程复用同一份 embedder / index
（逐本调 `ingest_book.py` 的话，每本都要重新加载 40 秒的模型）。

实测 5 本书 × 每本前 10 页：

```
成功 5 | 跳过 0 | 失败 0 | 总耗时 442.4s
累计页数 49 | 子块 213 | 向量 285
```

原样再跑一次 → **5 本全部跳过，0.1 秒**，去重生效。

**跨书检索验证**（三问分属三本不同的书）：

| 问题 | 命中 | 分数 |
|---|---|---|
| 实验中的自变量怎么控制 | 实验心理学 p8 | 0.6538 |
| 儿童认知发展阶段有哪些 | 儿童心理学 p8 / p9 | 0.6260 / 0.5950 |
| 心理学论文写作的格式规范 | 论文写作规范 p1 / p3 | 0.7249 / 0.7240 |

三问都落到正确的书上，没有串库、没有命中别的书里相似的段落——
这说明「按页切块 + 页码元数据 + 双路召回 + 重排」这条链路是通的。

### 1.1d 小样本冒烟测试（2026-10-21 已跑通）

`scripts/smoke_test.py` 拿 30 页真实教材跑通了完整闭环：
**PyMuPDF 抽文本 → 父子块切分 → bge-m3 双路向量 → Milvus 写入 → 混合检索 → int8 重排 → 上下文组装**。

样本：《实验心理学》杨治良，前 30 页 / 27 页有效文本 / 10 父块 / 39 子块 / 79 条向量。

| 查询 | 意图 | 重排后候选 | 结果 |
|---|---|---|---|
| 实验中的自变量和因变量怎么区分 | CONTRAST | 4 | ✅ top1 命中 p29「一、自变量和因变量的确定」(0.7251) |
| 什么是被试内设计 | DEFINE | 0 | ✅ 正确返回「未覆盖」——这 30 页里确实没这个概念 |
| 实验心理学研究的基本程序有哪些 | FRAMEWORK | 6 | ✅ top1 命中 p15「第二节 心理实验的程序」(0.7296) |

**这轮测出的最重要一条：`rerank_score_threshold` 原来是废的。**

bge-reranker 对**不相关**段落的 logit 接近 0，sigmoid 后就是 **0.5**。原阈值 0.15 意味着
所有结果都能通过——第二问就是这么把「被试者变量」当答案返回的。实测分布：

| 类型 | sigmoid 后分数 |
|---|---|
| 真正相关 | 0.66 ~ 0.73 |
| 完全不相关 | 0.50 ~ 0.55 |

改到 **0.58** 后，第二问候选归零、正确触发「未覆盖」。这个 0.58 是首轮实测划的线，
等 300 条黄金集建好后必须重新标定——三个查询推不出最优阈值，只能推出量级。

**延迟实测（热态）：** 3.1 ~ 3.2 秒/问（编码 0.3s + Milvus 0.5s + 重排 6 候选 2.4s）。
**冷启动第一问 46 秒**（加载 bge-m3 39s + reranker 10s），部署时要做预热，否则第一个用户会以为服务挂了。

数据留在 shared Milvus 的 `smoke_kb_001`，用完删：

```bash
.venv/Scripts/python.exe scripts/smoke_test.py --query-only --cleanup
```



---

## 二、启动

```bash
# 1. 连通性自检（MySQL / Milvus / Redis）
python -m app.check_env

# 2. 首次运行：建库 + 建表
python -m scripts.init_db

# 3. 自检配置与提示词
python -m app.config
python -m app.prompt_loader

# 4. 启动服务
uvicorn app.main:app --reload --port 8000
```

接口文档：<http://127.0.0.1:8000/docs>
  
健康检查：<http://127.0.0.1:8000/api/health>

---

## 三、测试

```bash
python -m pytest tests/ -v                      # 全部，约 20 秒
python -m pytest tests/unit -v                  # 只跑单元测试，约 7 秒
python -m pytest tests/ -v -m integration       # 真实 MySQL 连通性（需服务器可达）

# 覆盖率（本机沙箱下 pytest-cov 会在结束时删临时文件失败，用 coverage 直接跑）
coverage run --source=app -m pytest tests/ -q && coverage report
```

**实测结果（2026-10-20）**

```
261 passed in 19.76s
TOTAL 76%
```

**零跳过。** 接口测试用内存 SQLite 跑（`tests/conftest.py::sqlite_client`），
  
不再因为 MySQL 连不上而 skip。真实 MySQL 的验证单独放在 `test_integration_mysql.py`。

| 模块                    | 覆盖率  |
| --------------------- | ---- |
| db/models.py          | 100% |
| generate/intent.py    | 100% |
| ingest/detect.py      | 100% |
| retrieve/fusion.py    | 100% |
| generate/answer.py    | 99%  |
| schemas/kb.py         | 98%  |
| generate/safety.py    | 97%  |
| retrieve/service.py   | 96%  |
| core/errors.py        | 96%  |
| ingest/chunker.py     | 95%  |
| core/security.py      | 95%  |
| core/captcha.py       | 94%  |
| api/main.py           | 93%  |
| generate/verify.py    | 93%  |
| schemas/auth.py       | 93%  |
| prompt_loader.py      | 91%  |
| clients/llm_client.py | 94%  |
| generate/rewrite.py   | 86%  |
| core/logging.py       | 86%  |
| core/deps.py          | 85%  |
| config.py             | 84%  |

`clients/milvus_client.py` 与 `clients/rerank_client.py` 覆盖率偏低属预期：
  
必须连真实 Milvus 与加载模型权重才能验证，已单独把两处「错了不报错、只悄悄返回错结果」
  
的纯逻辑测死（多知识库过滤表达式、空文档短路）。

---

## 四、已完成 / 未完成

### ✅ M1 已完成

| 模块      | 文件                                  | 说明                                                                                      |
| ------- | ----------------------------------- | --------------------------------------------------------------------------------------- |
| 配置      | `app/config.py`                     | config.ini 解析 + 环境变量覆盖，缺项 fail-fast                                                     |
| 日志      | `app/core/logging.py`               | JSON 结构化、双输出、trace_id、敏感字段脱敏                                                            |
| 异常      | `app/core/errors.py`                | 15 个业务错误码 + 中文提示 + 统一响应                                                                 |
| 密码/JWT  | `app/core/security.py`              | bcrypt（cost 12）+ HS256，过期与非法分开报错                                                        |
| 验证码     | `app/core/captcha.py`               | 4 位去混淆字符、干扰线、只存哈希                                                                       |
| 数据模型    | `app/db/models.py`                  | 8 张表：users / captcha / kb / documents / chunks / sessions / messages / query_logs       |
| 鉴权接口    | `app/api/auth.py`                   | 验证码 / 注册 / 登录 / me                                                                      |
| 切块算法    | `app/ingest/chunker.py`             | 中文分隔符递归切分 + 父子块 + 按页切分                                                                  |
| 检索融合    | `app/retrieve/fusion.py`            | RRF 融合 + 父块去重 + 阈值过滤 + 上下文预算 + 带 [Sn] 编号的上下文拼装                                          |
| 提示词     | `prompts/` + `app/prompt_loader.py` | 11 个中文模板，加载器强制校验 front-matter 与四段结构，缺「反例」直接报错                                           |
| LLM 客户端 | `app/clients/llm_client.py`         | OpenAI 兼容协议，chat / chat_stream(SSE) / chat_json；transport 可注入，测试不花钱不联网；4xx 不重试、5xx 指数退避 |
| 安全护栏    | `app/generate/safety.py`            | 危机词熔断 + 求诊断拦截，**不走大模型**，固定话术；区分学术提问与第一人称求助                                              |
| 意图分类    | `app/generate/intent.py`            | 10 类意图；规则优先挡掉高置信样本，省一半 token；模型失败降级 DEFINE                                              |
| 查询改写    | `app/generate/rewrite.py`           | direct / hyde / subquery / stepback 四策略，异常一律降级 direct                                   |
| 答案生成    | `app/generate/answer.py`            | 三段式生成 + 引用编号体系 + 悬空引用检测 + 「未覆盖」识别                                                       |
| 引用校验    | `app/generate/verify.py`            | 逐行核查 [Sn] 断言，不通过则重写或删除，统计通过率                                                            |
| 解析决策    | `app/ingest/detect.py`              | 扩展名→归一方式、文本层密度→MinerU backend、单页兜底判定、成本估算                                               |
| 向量化     | `app/clients/embed_client.py`       | bge-m3 一次编码出稠密+稀疏，懒加载，同步推理走线程池                                                          |
| 重排      | `app/clients/rerank_client.py`      | bge-reranker-v2-m3，空输入短路不加载模型                                                           |
| 向量库     | `app/clients/milvus_client.py`      | 双路 hybrid_search、kb_id partition_key 隔离、父块回填、按 doc/kb 删除                                |
| 检索编排    | `app/retrieve/service.py`           | 意图→改写→多 query 召回→RRF→去重→重排→父块回填→预算裁剪，全链路依赖注入可测                                          |
| 知识库接口   | `app/api/kb.py`                     | 增删改查 + 统计，越权一律 404                                                                      |
| 文档接口    | `app/api/documents.py`              | 上传（落盘 + 建记录，解析异步化）、列表、详情、删除                                                             |
| 文本块接口   | `app/api/chunks.py`                 | 分页 / 关键词过滤 / 编辑（标记 needs_reindex）/ 禁用 / 删除                                              |

### ⬜ 下一步

| 阶段    | 内容                                                  | 依赖              |
| ----- | --------------------------------------------------- | --------------- |
| M2 剩余 | Calibre/LibreOffice 实际调用、解析任务队列（当前 BackgroundTasks 占 worker）、chunk 向量重建任务 | MinerU 已装 |
| ~~M3~~ | ~~会话与消息接口、SSE 问答接口~~ → **已完成**，见 §1.1i              | —               |
| M4    | ~~Vue 前端五页~~ ✅ / Redis 缓存、FAQ 快通道（未做）              | M3 ✅            |
| M5    | 黄金集 300 条、Ragas 全指标、回归 CI                           | M4              |

**接口层已就绪（20 个 path / 29 个操作）**：鉴权 4 + 知识库 6 + 文档 5 + 文本块 5
+ 问答与会话 6 + 每日推荐 2 + 健康检查 1（`app.openapi()` 实测，别手数）。

问答链路已打通：`/api/chat` → 安全熔断 → 意图改写 → 混合检索 → 重排 → 父块回填
→ 流式生成 → 引用编号 → 断流保全。真实模型实测见 §1.1i。

### ✅ 前端（2026-10-21 完成）

Vue 3 工程在 `frontend/`，五页全部接真实接口跑通，截图见 `frontend/shots/`。

```
Vue 3.5  ·  Vite 6  ·  TypeScript  ·  Pinia  ·  Vue Router（hash 模式）
Element Plus（表单/表格/上传/对话框）  ·  axios（普通请求）  ·  fetch（SSE）
marked + DOMPurify（渲染模型输出）
```

| 页面 | 路由 | 关键实现 |
|---|---|---|
| 登录 / 注册 | `#/login` | 验证码按 scene 分别取（登录 `login`、注册 `register`，服务端校验 scene，不匹配会拒）；失败后强制换一张，否则撞「验证码已使用」 |
| 知识库列表 | `#/kb` | 卡片显示文档数/块数；增删改对话框 |
| 文档管理 | `#/kb/:kbId` | 上传队列 + 进度；6 阶段状态轮询；失败原因 tooltip；参数只给真正生效的三个 |
| 文本块 | `#/kb/:kbId/docs/:docId/chunks` | 分页/关键词/只看启用；编辑后提示「需重建向量」；从引用跳转时按页码高亮 |
| 每日知识 | `#/daily` | 当天固定的随机推荐；点开看父块全文 + 语义关联内容；一键开新对话追问 |
| 智能问答 | `#/chat` | SSE 流式；引用卡片可点跳原文；会话懒创建；可停止生成 |

#### 三个前端专属的坑

**1. SSE 必须用 `fetch`，不能用 axios 也不能用 EventSource。**
axios 基于 XHR 拿不到分块；EventSource 不支持自定义 `Authorization` 头（token 在 localStorage，不是 Cookie）。
只能 `fetch` + `response.body.getReader()` 手动按 `\n\n` 切事件。

**2. 逐 token 改响应式状态会让 markdown 每秒重算上百次。**
实测一次回答 600+ 个 token 事件，每个都触发 `marked.parse` + DOMPurify 会明显掉帧。
用 `requestAnimationFrame` 合并到每帧最多一次。

**3. 引用点击要能真的翻到原文。**
`[S1]` 渲染成可点标签（在 markdown 转换**之后**用正则替换，否则会被当成链接语法），
点击跳到该文档的文本块页并按页码高亮。为此后端 `Citation` 补了 `doc_id` 字段——
没有它前端只能显示书名，而「能自己去翻那一页核对」正是这套系统相对纯聊天框的唯一价值。

#### 状态轮询不是固定间隔

一本 300 页的书要跑十几分钟。固定 2 秒轮询会白打几百个请求：
状态连续三次没变化就退避到 5 秒，全部 `done` / `failed` 立刻停。

#### 每日知识推荐（`/api/recommend`）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/recommend/daily?kb_id=&variation=0&limit=3` | 今日推荐列表 |
| GET | `/api/recommend/chunks/{chunk_id}` | 详情（父块全文）+ 语义关联 |

三个设计决定：

**1. 当天固定，不是每次随机。** 种子 = `日期 | 用户 | 知识库`，
同一个用户同一天刷新多少次都是同一批。用户看到一条想细看，回头找不到是最败好感的行为。
想看别的点「换一批」，它给种子加 `variation` 偏移，结果仍然固定可复现。

**2. 卡片用子块，详情给父块。** 卡片要短（子块约 300 字），
但「详细介绍」必须是父块全文（约 1200 字）——子块常是半句假设半句结论，
拿它当详情等于让用户看一段没头没尾的话。实测详情返回 1191 字。

**3. 关联内容用库里已有的向量，不调大模型。** `MilvusIndex.get_vector(chunk_id)`
直接取向量做单路稠密检索，比重新编码快 0.4 秒。实测热态 **60ms** 返回详情，
首次 355ms。要「讲解这段」不打这个接口——走问答链路，不另造一套生成。

候选过滤：字符数 ≥80、至少一个句号、排除目录页（`.....12` 这种引导点 ≥3 个）。
排序后只看前 40 条正文，避免大库全表扫。

**标题的省略号是有意的。** 切块是按长度硬切的，块首经常落在句子中间
（比如「通信地址」被切成「信地址」）。这种片段标题前加 `…`，
让用户一眼看出是节选而不是完整段落；以小标题开头的（「三、实验设计」）不加。

**追问链路**：详情页「就此提问」→ 跳 `/chat` 并带上 `q` 和 `kb`，
前端开的是**新对话**而不是接着上次那条聊——上下文混在一起会让模型串题。


#### 启动

```bash
# 后端（8000）
export JWT_SECRET=... DEEPSEEK_API_KEY=...
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000

# 前端（5173，/api 已代理到 8000）
cd frontend && npm install && npm run dev
```

#### 已知限制

- **Element Plus 全量引入，主包 1.27 MB（gzip 411 KB）。** 演示够用；
  优化可换 `unplugin-vue-components` 按需引入，能砍一半以上。
- **没做移动端适配。** 桌面优先，窄屏下侧边栏和表格会挤。
- **文本块编辑不会重建向量。** 页面明确提示「需重建才能按新内容被检索」，
  重建功能属于 M2 剩余项。当前建议用「禁用」代替「修改」。
- **上传的切块参数只认 `parent_size` / `child_size` / `overlap_ratio`。**
  `chunk_size` 与 `mode` 在当前 pipeline 里不生效，所以前端没放出来——
  一个改了不生效的参数比没有这个参数更糟。

#### 截图脚本

`frontend/scripts/shot.mjs` 用本机 Chrome 的 CDP 协议截图，可注入 token 跳过登录：

```bash
node scripts/shot.mjs "http://127.0.0.1:5173/#/kb" "D:/shots/kb.png"
```

> 输出路径必须写盘符（`D:/...`）。写 `/d/...` 会被 Git Bash 转成 `D:\d\...`，文件落到别处。

---

## 五、目录结构

```
RAG-psy/
├── config.ini              # 主配置（含账号，已 gitignore）
├── config.ini.example      # 脱敏模板
├── mineru.yaml             # MinerU 配置（模型目录指向 D 盘）
├── pyproject.toml
├── prompts/                # 11 个中文提示词模板
├── scripts/                # 运维与验证脚本
│   ├── init_db.py · migrate.py
│   ├── ingest_book.py · ingest_batch.py      # 绕过 HTTP 直接入库
│   ├── smoke_test.py · smoke_chat.py         # 检索 / 会话冒烟
│   ├── rebuild_collection.py                 # is_parent schema 重建
│   ├── dev_token.py                          # 给截图脚本签 token
│   ├── bench_models.py · bench_rerank_length.py · tune_rerank.py
│   └── verify_mineru.py
├── app/
│   ├── main.py             # FastAPI 入口（trace 中间件 + CORS + 异常处理 + 预热）
│   ├── config.py
│   ├── check_env.py        # 三服务连通性自检
│   ├── prompt_loader.py
│   ├── core/               # logging / errors / security / captcha / deps
│   ├── db/                 # base（引擎与会话）/ models（8 张表）
│   ├── schemas/            # Pydantic 请求响应（auth / kb / chat）
│   ├── api/                # auth · kb · documents · chunks · chat · recommend
│   ├── ingest/             # base / parsers / pipeline / chunker / detect
│   ├── retrieve/           # service（检索编排）/ fusion（RRF、去重、裁剪）
│   └── generate/           # intent / rewrite / answer / safety / verify
├── frontend/               # Vue 3 五页
│   ├── src/api/            # client（axios+401）/ index（接口）/ chat（SSE）
│   ├── src/stores/         # auth
│   ├── src/views/          # Login / KbList / KbDetail / Chunks / Recommend / Chat
│   ├── src/layouts/        # AppLayout（侧边栏）
│   └── scripts/shot.mjs    # CDP 截图（注入 token）
└── tests/
    ├── conftest.py         # 环境变量注入 + DB / Milvus 可用性探测
    ├── unit/               # chunker / fusion / pipeline / parsers / llm
    ├── api/                # auth / kb / dedupe / chat
    └── integration/        # 真实 MySQL + Milvus
```

---

## 六、几条不能破的规矩

1. 密钥只走环境变量，`config.ini` 里只写变量名。
2. 密码必须 bcrypt，禁止明文与 MD5。
3. 登录失败不区分「账号不存在」与「密码错误」。
4. 越权访问资源返回 404，不返回 403（避免探测）。
5. 检索不到就回答「当前知识库未覆盖」，不许用模型自身知识补全。
6. 改提示词必须跑回归，不许凭手感调。
7. 测试红灯不许通知验收。
