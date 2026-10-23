"""配置加载：config.ini（共享服务器账号）+ 环境变量（密钥），启动即校验。

设计说明：
1. 账号、地址、参数全部集中在 config.ini，方便团队共用一个配置文件模板。
2. API Key 与 JWT 密钥**不写进 ini**，只写「环境变量名」，代码按名字去读。
   这样 ini 可以进仓库（用 config.ini.example），密钥留在各自的环境变量里。
3. 缺关键项直接抛错退出，不要带着空 Key 跑到线上。
4. 业务代码禁止直接读 os.environ 或 configparser，统一从这里取。

自检：python -m app.config
"""

from __future__ import annotations

import configparser
import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.ini"


class ConfigError(RuntimeError):
    """配置缺失或不合法。"""


# ---------------------------------------------------------------- ini 读取
_parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=None)
if not CONFIG_PATH.exists():
    raise ConfigError(
        f"找不到配置文件：{CONFIG_PATH}\n"
        f"请把 config.ini.example 复制为 config.ini，并填写数据库账号。"
    )
_parser.read(CONFIG_PATH, encoding="utf-8")


def _str(section: str, key: str, default: str = "") -> str:
    if not _parser.has_section(section):
        return default
    return _parser.get(section, key, fallback=default).strip()


def _int(section: str, key: str, default: int = 0) -> int:
    raw = _str(section, key, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"[{section}] {key} 必须是整数，当前 {raw!r}") from exc


def _float(section: str, key: str, default: float = 0.0) -> float:
    raw = _str(section, key, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"[{section}] {key} 必须是数字，当前 {raw!r}") from exc


def _bool(section: str, key: str, default: bool = False) -> bool:
    raw = _str(section, key, "true" if default else "false").lower()
    if raw in ("true", "1", "yes", "on"):
        return True
    if raw in ("false", "0", "no", "off"):
        return False
    raise ConfigError(f"[{section}] {key} 必须是布尔值，当前 {raw!r}")


def _json_list(section: str, key: str) -> list[str]:
    raw = _str(section, key, "[]")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"[{section}] {key} 必须是 JSON 数组，例如 [\"a\", \"b\"]，当前 {raw!r}"
        ) from exc
    if not isinstance(value, list):
        raise ConfigError(f"[{section}] {key} 必须是 JSON 数组")
    return [str(v) for v in value]


def _env(primary: str, *fallback_names: str) -> str:
    """按顺序取环境变量，全空返回空串。"""
    for name in (primary, *fallback_names):
        if not name:
            continue
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _secret(env_name: str, what: str) -> str:
    """读密钥类环境变量，缺失即报错。"""
    value = _env(env_name)
    if not value:
        raise ConfigError(
            f"缺少环境变量 {env_name}（{what}）。\n"
            f"请设置：set {env_name}=your-key   （PowerShell: $env:{env_name}=\"your-key\"）"
        )
    return value


# ---------------------------------------------------------------- 配置模型
class AppConf(BaseModel):
    name: str = "心理学知识问答系统"
    env: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    upload_dir: str = "storage/uploads"
    image_dir: str = "storage/images"
    max_upload_mb: int = 200


class MySQLConf(BaseModel):
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "psych_rag"
    charset: str = "utf8mb4"
    pool_size: int = 10
    pool_recycle: int = 3600

    @property
    def dsn(self) -> str:
        return (
            f"mysql+asyncmy://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}?charset={self.charset}"
        )

    @property
    def masked(self) -> str:
        """日志用：隐藏密码。"""
        return f"{self.user}@{self.host}:{self.port}/{self.database}"


class RedisConf(BaseModel):
    host: str = "127.0.0.1"
    port: int = 6379
    password: str = ""
    db: int = 3
    enabled: bool = False


class MilvusConf(BaseModel):
    host: str = "127.0.0.1"
    port: str = "19530"
    user: str = ""
    password: str = ""
    database: str = "psych_rag"
    collection: str = "psych_chunks_v1"
    consistency_level: str = "Bounded"

    @property
    def uri(self) -> str:
        return f"http://{self.host}:{self.port}"


class LLMProviderConf(BaseModel):
    model: str
    base_url: str
    api_key: str = ""
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2400
    # 只要求模型回一小段 JSON 的场景（意图分类、查询改写）用的上限。
    # deepseek-flash 是推理模型，token 会先花在思考上：256 时 content 直接是空串，
    # chat_json 还会因此多重试一次，一次分类白跑三趟请求。2048 实测用 250 token 就够。
    max_tokens_json: int = 2048
    timeout: int = 30
    max_retries: int = 2


class LLMConf(BaseModel):
    provider: str = "deepseek"
    providers: dict[str, LLMProviderConf] = Field(default_factory=dict)

    def active(self) -> LLMProviderConf:
        conf = self.providers.get(self.provider)
        if conf is None:
            raise ConfigError(f"未配置默认 provider：{self.provider}")
        return conf


class VLConf(BaseModel):
    model: str = "qwen3-max"
    base_url: str = ""
    api_key: str = ""
    image_max_side: int = 2048
    timeout: int = 60


class EmbeddingConf(BaseModel):
    bge_m3_path: str = ""
    bge_reranker_path: str = ""
    device: str = "cuda"
    batch_size: int = 16
    # torch 的 CPU 线程数。0 表示用 os.cpu_count()。
    # 实测这台机器 torch 默认只开 4 线程（逻辑核 8），手动开满后重排快约 10%。
    num_threads: int = 0

    # local = 本地 bge-m3（稠密+稀疏）；dashscope = 百炼 text-embedding-v4（仅稠密）
    # 实测本机 4 物理核跑 bge-m3 只有 3.7 秒/向量，4 万页要 257 小时；
    # API 约 0.5 元/百万 token，全量 15 元、1.5 小时，这笔账没有第二个答案。
    provider: str = "local"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "text-embedding-v4"
    dashscope_dim: int = 1024
    # 单次请求最多 10 条，这是接口硬限制，不是调优参数
    dashscope_batch_size: int = 10
    dashscope_concurrency: int = 8
    dashscope_api_key_env: str = "DASHSCOPE_API_KEY"

    def check_paths(self) -> None:
        """本地模型目录必须存在，否则检索链路起不来。

        provider=dashscope 时不校验——向量走 API，本机不需要 bge-m3 权重。
        """
        if self.provider == "dashscope":
            return
        for label, path in (
            ("bge_m3_path", self.bge_m3_path),
            ("bge_reranker_path", self.bge_reranker_path),
        ):
            if not path:
                raise ConfigError(f"[embedding] {label} 未填写")
            if not Path(path).exists():
                raise ConfigError(f"[embedding] {label} 指向的目录不存在：{path}")


class ParserConf(BaseModel):
    text_layer_threshold: int = 100
    mineru_timeout: int = 1800
    # MinerU 档位。basic = ONNX 小模型（CPU 可用，做版面/OCR/公式/表格）；
    # standard 还要额外的 1.2B VLM，无 GPU 机器单页几十秒，不可用。
    mineru_tier: str = "basic"
    # auto：让 MinerU 自己判断每页有没有文本层，有就走文本流，没有才 OCR。
    mineru_ocr_mode: str = "auto"
    vl_fallback_limit: float = 0.15
    ebook_convert_bin: str = "ebook-convert"
    libreoffice_bin: str = "libreoffice"
    # 切块前剔除正文前的非内容页。目录里有全书章节名，向量会贴着任何问题，
    # 把真正的正文答案挤掉——实测占比 1%，但出现在召回里的概率远高于 1%。
    drop_toc: bool = True
    drop_copyright: bool = True
    # 序言默认保留：作者的研究缘起和全书脉络不是噪音。要删就打开这一项。
    drop_preface: bool = False


class ChunkConf(BaseModel):
    mode: str = "single"
    size: int = 800
    overlap_ratio: float = 0.10
    min_size: int = 80
    parent_size: int = 1200
    parent_min_size: int = 800
    child_size: int = 320
    child_min_size: int = 200
    separators: list[str] = Field(default_factory=list)

    def overlap_size(self) -> int:
        return int(self.size * self.overlap_ratio)


class RetrievalConf(BaseModel):
    dense_topk: int = 20
    sparse_topk: int = 20
    fusion: str = "rrf"
    rrf_k: int = 60
    rerank_topk: int = 5
    rerank_score_threshold: float = 0.54
    context_max_chars: int = 6000
    hybrid: bool = True
    # 重排候选上限。去重后可能有 40 个父块，全送去交叉编码在 CPU 上是灾难。
    # 只取 RRF 前 N 个送去重排——前 8 基本覆盖真实命中，剩下的本来也进不了最终 top5。
    # 实测 int8 量化后约 470 ms/对，8 条 ≈ 3.8 秒；调到 6 条 ≈ 2.8 秒。
    rerank_max_candidates: int = 20
    # 交叉编码 token 上限。实测 256 token 时，1200 字父块与 300 字子块耗时相同
    # （都被截到 256），所以"重排父块"在质量上更优、成本上不加价。
    # 512 token 会慢一倍，而父块后半段对判别帮助很小。
    rerank_max_length: int = 256
    # int8 动态量化。实测 2.83 倍加速，正确文档仍排第一。
    # CPU 机器不开这个，重排基本没法用。
    rerank_quantize: bool = True


class GenerateConf(BaseModel):
    history_rounds: int = 5
    answer_max_chars: int = 2000
    citation_verify: bool = True
    verify_pass_rate_warn: float = 0.92
    rewrite_ratio_warn: float = 0.10


class SecurityConf(BaseModel):
    jwt_expire_hours: int = 24
    bcrypt_rounds: int = 12
    captcha_length: int = 4
    captcha_expire_seconds: int = 300
    jwt_secret: str = ""
    crisis_keywords: list[str] = Field(default_factory=list)


class LoggerConf(BaseModel):
    level: str = "INFO"
    log_dir: str = "logs"
    file_name: str = "app.log"
    retain_days: int = 14
    console: bool = True


class Settings(BaseModel):
    app: AppConf = Field(default_factory=AppConf)
    mysql: MySQLConf = Field(default_factory=MySQLConf)
    redis: RedisConf = Field(default_factory=RedisConf)
    milvus: MilvusConf = Field(default_factory=MilvusConf)
    llm: LLMConf = Field(default_factory=LLMConf)
    vl: VLConf = Field(default_factory=VLConf)
    embedding: EmbeddingConf = Field(default_factory=EmbeddingConf)
    parser: ParserConf = Field(default_factory=ParserConf)
    chunk: ChunkConf = Field(default_factory=ChunkConf)
    retrieval: RetrievalConf = Field(default_factory=RetrievalConf)
    generate: GenerateConf = Field(default_factory=GenerateConf)
    security: SecurityConf = Field(default_factory=SecurityConf)
    logger: LoggerConf = Field(default_factory=LoggerConf)


# ---------------------------------------------------------------- 组装
def _inject_ini_secrets() -> None:
    """把 config.ini 的 [secrets] 段注入环境变量。

    三个消费方（VL / Qwen / Embedding）都按环境变量名取值。开发机上把 key
    写进 ini 能省掉每开一个终端就 export 一次的麻烦，而 config.ini 已在
    .gitignore 里，不会跟着代码走。

    用 setdefault 而不是覆盖：环境变量优先，ini 只兜底，
    这样 CI / 生产照旧可以用注入的环境变量。
    """
    if not _parser.has_section("secrets"):
        return
    for name, value in _parser.items("secrets"):
        cleaned = (value or "").strip()
        if cleaned:
            os.environ.setdefault(name.upper(), cleaned)


def load_settings() -> Settings:
    _inject_ini_secrets()

    # MySQL：环境变量优先，方便在不同同学机器上只改 env
    mysql = MySQLConf(
        host=_env("MYSQL_HOST") or _str("mysql", "host", "127.0.0.1"),
        port=int(_env("MYSQL_PORT") or _int("mysql", "port", 3306)),
        user=_env("MYSQL_USER") or _str("mysql", "user", "root"),
        password=_env("MYSQL_PASSWORD") or _str("mysql", "password", ""),
        database=_env("MYSQL_DATABASE") or _str("mysql", "database", "psych_rag"),
        charset=_str("mysql", "charset", "utf8mb4"),
        pool_size=_int("mysql", "pool_size", 10),
        pool_recycle=_int("mysql", "pool_recycle", 3600),
    )

    redis = RedisConf(
        host=_env("REDIS_HOST") or _str("redis", "host", "127.0.0.1"),
        port=int(_env("REDIS_PORT") or _int("redis", "port", 6379)),
        password=_env("REDIS_PASSWORD") or _str("redis", "password", ""),
        db=_int("redis", "db", 3),
        enabled=_bool("redis", "enabled", False),
    )

    milvus = MilvusConf(
        host=_env("MILVUS_HOST") or _str("milvus", "host", "127.0.0.1"),
        port=_env("MILVUS_PORT") or _str("milvus", "port", "19530"),
        user=_env("MILVUS_USER") or _str("milvus", "user", ""),
        password=_env("MILVUS_PASSWORD") or _str("milvus", "password", ""),
        database=_str("milvus", "database_name", "psych_rag"),
        collection=_str("milvus", "collection_name", "psych_chunks_v1"),
        consistency_level=_str("milvus", "consistency_level", "Bounded"),
    )

    # LLM：默认 provider + 备选 provider，密钥各自从 env 读
    providers: dict[str, LLMProviderConf] = {}
    main = LLMProviderConf(
        model=_str("llm", "model", "deepseek-flash"),
        base_url=_str("llm", "base_url", ""),
        api_key=_secret(_str("llm", "api_key_env", "DEEPSEEK_API_KEY"), "主模型 API Key"),
        temperature=_float("llm", "temperature", 0.2),
        top_p=_float("llm", "top_p", 0.9),
        max_tokens=_int("llm", "max_tokens", 2400),
        timeout=_int("llm", "timeout", 30),
        max_retries=_int("llm", "max_retries", 2),
    )
    providers[_str("llm", "provider", "deepseek")] = main

    if _parser.has_section("llm_qwen"):
        qwen_key = _env(_str("llm_qwen", "api_key_env", "DASHSCOPE_API_KEY"))
        if qwen_key:
            providers["qwen"] = LLMProviderConf(
                model=_str("llm_qwen", "model", "qwen-max"),
                base_url=_str("llm_qwen", "base_url", ""),
                api_key=qwen_key,
                temperature=main.temperature,
                top_p=main.top_p,
                max_tokens=main.max_tokens,
                timeout=main.timeout,
                max_retries=main.max_retries,
            )

    vl = VLConf(
        model=_str("vl", "model", "qwen3-max"),
        base_url=_str("vl", "base_url", ""),
        api_key=_secret(_str("vl", "api_key_env", "DASHSCOPE_API_KEY"), "多模态 API Key"),
        image_max_side=_int("vl", "image_max_side", 2048),
        timeout=_int("vl", "timeout", 60),
    )

    embedding = EmbeddingConf(
        bge_m3_path=_str("embedding", "bge_m3_path", ""),
        bge_reranker_path=_str("embedding", "bge_reranker_path", ""),
        device=_str("embedding", "device", "cuda"),
        batch_size=_int("embedding", "batch_size", 16),
        num_threads=_int("embedding", "num_threads", 0),
        provider=_str("embedding", "provider", "local"),
        dashscope_model=_str("embedding", "dashscope_model", "text-embedding-v4"),
        dashscope_dim=_int("embedding", "dashscope_dim", 1024),
        dashscope_batch_size=_int("embedding", "dashscope_batch_size", 10),
        dashscope_concurrency=_int("embedding", "dashscope_concurrency", 8),
        dashscope_api_key_env=_str("embedding", "dashscope_api_key_env", "DASHSCOPE_API_KEY"),
    )

    chunk = ChunkConf(
        mode=_str("chunk", "mode", "single"),
        size=_int("chunk", "size", 800),
        overlap_ratio=_float("chunk", "overlap_ratio", 0.10),
        min_size=_int("chunk", "min_size", 80),
        parent_size=_int("chunk", "parent_size", 1200),
        parent_min_size=_int("chunk", "parent_min_size", 800),
        child_size=_int("chunk", "child_size", 320),
        child_min_size=_int("chunk", "child_min_size", 200),
        separators=_json_list("chunk", "separators"),
    )

    security = SecurityConf(
        jwt_expire_hours=_int("security", "jwt_expire_hours", 24),
        bcrypt_rounds=_int("security", "bcrypt_rounds", 12),
        captcha_length=_int("security", "captcha_length", 4),
        captcha_expire_seconds=_int("security", "captcha_expire_seconds", 300),
        jwt_secret=_secret(_str("security", "jwt_secret_env", "JWT_SECRET"), "JWT 密钥"),
        crisis_keywords=_json_list("security", "crisis_keywords"),
    )

    return Settings(
        app=AppConf(
            name=_str("app", "name", "心理学知识问答系统"),
            env=_str("app", "env", "dev"),
            host=_str("app", "host", "0.0.0.0"),
            port=_int("app", "port", 8000),
            debug=_bool("app", "debug", False),
            upload_dir=_str("app", "upload_dir", "storage/uploads"),
            image_dir=_str("app", "image_dir", "storage/images"),
            max_upload_mb=_int("app", "max_upload_mb", 200),
        ),
        mysql=mysql,
        redis=redis,
        milvus=milvus,
        llm=LLMConf(provider=_str("llm", "provider", "deepseek"), providers=providers),
        vl=vl,
        embedding=embedding,
        parser=ParserConf(
            text_layer_threshold=_int("parser", "text_layer_threshold", 100),
            mineru_timeout=_int("parser", "mineru_timeout", 1800),
            mineru_tier=_str("parser", "mineru_tier", "basic"),
            mineru_ocr_mode=_str("parser", "mineru_ocr_mode", "auto"),
            vl_fallback_limit=_float("parser", "vl_fallback_limit", 0.15),
            ebook_convert_bin=_str("parser", "ebook_convert_bin", "ebook-convert"),
            libreoffice_bin=_str("parser", "libreoffice_bin", "libreoffice"),
        ),
        chunk=chunk,
        retrieval=RetrievalConf(
            dense_topk=_int("retrieval", "dense_topk", 20),
            sparse_topk=_int("retrieval", "sparse_topk", 20),
            fusion=_str("retrieval", "fusion", "rrf"),
            rrf_k=_int("retrieval", "rrf_k", 60),
            rerank_topk=_int("retrieval", "rerank_topk", 5),
            rerank_score_threshold=_float("retrieval", "rerank_score_threshold", 0.54),
            context_max_chars=_int("retrieval", "context_max_chars", 6000),
            hybrid=_bool("retrieval", "hybrid", True),
            rerank_max_candidates=_int("retrieval", "rerank_max_candidates", 20),
            rerank_max_length=_int("retrieval", "rerank_max_length", 256),
            rerank_quantize=_bool("retrieval", "rerank_quantize", True),
        ),
        generate=GenerateConf(
            history_rounds=_int("generate", "history_rounds", 5),
            answer_max_chars=_int("generate", "answer_max_chars", 2000),
            citation_verify=_bool("generate", "citation_verify", True),
            verify_pass_rate_warn=_float("generate", "verify_pass_rate_warn", 0.92),
            rewrite_ratio_warn=_float("generate", "rewrite_ratio_warn", 0.10),
        ),
        security=security,
        logger=LoggerConf(
            level=_str("logger", "level", "INFO"),
            log_dir=_str("logger", "log_dir", "logs"),
            file_name=_str("logger", "file_name", "app.log"),
            retain_days=_int("logger", "retain_days", 14),
            console=_bool("logger", "console", True),
        ),
    )


settings = load_settings()


if __name__ == "__main__":
    # 自检：python -m app.config
    print(f"配置文件：{CONFIG_PATH}")
    print(f"应用：{settings.app.name}（{settings.app.env}）")
    print(f"MySQL：{settings.mysql.masked}")
    print(f"Redis：{settings.redis.host}:{settings.redis.port} db={settings.redis.db} enabled={settings.redis.enabled}")
    print(f"Milvus：{settings.milvus.uri} db={settings.milvus.database} collection={settings.milvus.collection}")
    print(f"默认模型：{settings.llm.provider} / {settings.llm.active().model}")
    print(f"可切换模型：{', '.join(settings.llm.providers)}")
    print(f"多模态：{settings.vl.model}")
    print(f"切块：{settings.chunk.mode} size={settings.chunk.size} overlap={settings.chunk.overlap_size()}")
    print(f"检索：dense={settings.retrieval.dense_topk} sparse={settings.retrieval.sparse_topk} "
          f"rerank_topk={settings.retrieval.rerank_topk} hybrid={settings.retrieval.hybrid}")
    print(f"危机词：{len(settings.security.crisis_keywords)} 个")
    print("配置自检通过。")
