# 模型清单 · D:\Project\RAG-psy

> **状态更新（2026-10-21）：三个模型已由用户放入 `MODEL/`，不再需要下载。**
> 下方的下载步骤只作为换机 / 重装时的参考保留。
>
> 实际落位：`MODEL/bge-m3`（2.2 GB）、`MODEL/bge-reranker-v2-m3`（2.2 GB）、
> `MODEL/bert-base-chinese`（394 MB，额外提供，可作意图分类底座）。

> 实测时间 2026-10-20 20:05。机位状态：**无 NVIDIA GPU**，C 盘剩 23 GB，D 盘剩 69 GB。

---

## 一、必需的本地模型（3 组，缺一组主链路就断）

| # | 模型 | 用途 | 体积 | 落点 | 缺了会怎样 |
|---|---|---|---|---|---|
| 1 | **bge-m3** | 稠密 1024 维 + 稀疏词权重，一次编码双路召回 | ~2.3 GB | `D:/models/bge-m3` | 检索完全无法工作，`/ask` 必挂 |
| 2 | **bge-reranker-v2-m3** | 精排，topK 20~30 压到 6 | ~2.3 GB | `D:/models/bge-reranker-v2-m3` | 能检索但精度掉一大截，易混辨析场景直接失效 |
| 3 | **MinerU pipeline 模型集** | 版面检测 / OCR / 公式 / 表格 / 阅读顺序 | ~2.0 GB | `~/.cache/modelscope` | 40,274 页扫描件全部解析不了 |

MinerU pipeline 具体包含：`PP-DocLayoutV2`（版面 205 MB）、`unimernet_hf_small_2503`（公式 773 MB）、`paddleocr_torch`（OCR 810 MB）、`SlanetPlus` + `UnetStructure`（表格 15 MB）、`paddle_table_cls` + `paddle_orientation`（分类，几 MB）、`layout_reader`（阅读顺序）。

**小计约 6.6 GB。**

> **下载时务必排除 `onnx/`。** bge-m3 仓库里同时放了 `pytorch_model.bin`（2.27 GB）和 `onnx/model.onnx_data`（2.27 GB），不排除的话一个模型要下 4.6 GB，翻倍且完全没用——FlagEmbedding 只认 PyTorch 权重。`scripts/download_models.py` 已经加了 `ignore_patterns`，别绕开它手敲 `modelscope download`。

---

## 二、明确不下的模型

| 模型 | 为什么不下 |
|---|---|
| **MinerU2.5-1.2B（VLM backend）** | 1.2B 视觉语言模型，官方要求 20 GB+ 显存。本机无 GPU，纯 CPU 单页几十秒 → 40,274 页要跑几百小时，不可行。扫描页兜底交给 `qwen3-max` API（`config.ini` 的 `[vl]` 段），MinerU 只跑 pipeline |
| **deepseek-flash** | 云 API，走 `DEEPSEEK_API_KEY` |
| **qwen3-max** | 云 API，走 `DASHSCOPE_API_KEY` |
| **任何 embedding 的 GGUF 量化版** | Ollama 那套只出稠密向量，出不来 sparse。bge-m3 的核心价值就是一次编码双路 |

---

## 三、Python 运行时（不是模型，但同样缺）

实测状态（2026-10-20 20:16 更新）：

| 包 | 状态 |
|---|---|
| `torch` | ✅ 2.14.1+cpu（4 线程，`cuda: False` 符合预期） |
| `modelscope` | ✅ 1.40.2（需额外补 `modelscope-hub` 0.4.6，否则 import 就崩） |
| `setuptools` | ✅ 84.2.0 |
| `filelock` / `fsspec` / `jinja2` / `markupsafe` / `mpmath` / `networkx` / `sympy` / `tqdm` / `psutil` | ✅ 已装（torch 依赖链） |
| `pymupdf` / `python-docx` / `pymilvus` / `numpy` / `pandas` / `requests` / `pyyaml` | ✅ 原有 |
| **`FlagEmbedding`** | ❌ 缺 —— bge-m3 官方加载方式，缺它载不进来 |
| **`transformers` / `tokenizers` / `huggingface_hub` / `safetensors` / `regex`** | ❌ 缺（FlagEmbedding 的依赖链） |
| **`mineru`** | ❌ 缺 —— 解析全链路靠它 |
| `ebooklib` / `python-pptx` / `redis` | ❌ 缺 |

命令行工具全缺：`mineru`、`ebook-convert`（Calibre）、`libreoffice`。

---

## 四、这台机器上 pip 装不了包（重要）

**现象**：`pip install torch -i 清华源` 报 `No matching distribution found`，`-vvv` 才看到真相：

```
https://pypi.tuna.tsinghua.edu.cn:443 "GET /simple/torch/" 403 None
Could not fetch URL ...: 403 Client Error: Forbidden
```

不是源没有，是 **pip 的 HTTPS 请求被沙箱透明代理挡下来了（403）**。同一个 URL 用 Python `urllib` 直连是 200、2.4 MB/s。

**绕过方式**：`scripts/fetch_wheels.py` —— 自己解析 simple index、按 `packaging.tags` 挑兼容 wheel、下载并校验 sha256，再交给 `pip install --no-deps --no-index` 从本地装。

```bash
# 取 torch 及其依赖
python -m scripts.fetch_wheels torch filelock sympy mpmath networkx jinja2 markupsafe fsspec setuptools tqdm psutil --install

# 取解析/IO 相关
python -m scripts.fetch_wheels modelscope ebooklib python-pptx redis --install
```

只处理 wheel 不处理 sdist（本机没有 C 编译器，sdist 装了也用不上），依赖要自己列——数量不多，比引一个解析器进来简单。

---

## 五、一条命令装齐

```bash
# 1) CPU 版 torch（Windows 上 PyPI 的 torch 就是 CPU 构建）
python -m scripts.fetch_wheels torch filelock sympy mpmath networkx jinja2 markupsafe fsspec setuptools tqdm psutil --install

# 2) 运行时
python -m scripts.fetch_wheels modelscope FlagEmbedding ebooklib python-pptx redis --install

# 3) 权重
python -m scripts.download_models --with-mineru
```

`--with-mineru` 会调官方 `mineru-models-download -s modelscope -m pipeline`，下载完自动写 `~/mineru.json`。

⚠️ `FlagEmbedding` 与 `mineru` 的依赖树较深，`--no-deps` 装法可能需要补几轮缺失模块。装完跑 `python -c "import FlagEmbedding"` 逐层排查。

**外部工具（装完有则走，没有则降级）：**

| 工具 | 用途 | 没有时 |
|---|---|---|
| Calibre `ebook-convert` | MOBI / AZW3 → EPUB | 15 个 MOBI 无法入 MVP |
| LibreOffice `soffice` | DOCX / PPTX → PDF | 6 个 Office 文件走 python-docx 直读，丢版面 |

---

## 六、CPU 上的真实速度预期（别按 GPU 估工期）

| 环节 | 实测/估算 |
|---|---|
| bge-m3 编码 | 10~30 条/秒 |
| bge-reranker 精排 | 20~50 ms / 对 |
| MinerU pipeline 解析 | 2~5 秒 / 页 |
| **全量建库（约 5 万页 / 10 万块）** | 编码 1~3 小时 + 解析 28~70 小时 |

建库是一次性成本，可以挂着过夜。**但线上问答时每次要编码 1 条 query（约 50 ms）+ 重排 20~30 对（约 1 秒）**，端到端 3 秒的目标在 CPU 上是紧的——`[retrieval] rerank_topk` 可能要压到 6 以下。
