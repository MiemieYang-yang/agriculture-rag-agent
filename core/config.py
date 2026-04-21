"""
统一配置管理
所有参数从 .env 读取，方便后续调整不改代码
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── LLM ──────────────────────────────────────────
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    QWEN_BASE_URL: str = os.getenv(
        "QWEN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    QWEN_MODEL: str = os.getenv("QWEN_MODEL", "qwen-plus")

    # ── 向量模型 ──────────────────────────────────────
    BGE_MODEL_NAME: str = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")

    # ── ChromaDB ──────────────────────────────────────
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "agri_knowledge")

    # ── 检索参数 ──────────────────────────────────────
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # ── HyDE 配置（Phase 2）──────────────────────────
    USE_HYDE: bool = os.getenv("USE_HYDE", "false").lower() == "true"

    HYDE_PROMPT_TEMPLATE: str = """请为以下问题写一个详细的假设答案。
这个答案应该：
1. 像是从专业农业文献中摘录的内容
2. 包含具体的数据和指标
3. 使用专业术语

问题：{question}

假设答案："""

    # ── Reranker 配置（Phase 2）──────────────────────
    USE_RERANKER: bool = os.getenv("USE_RERANKER", "false").lower() == "true"
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "50"))  # 粗排数量

    # ── 混合检索配置（Phase 2）──────────────────────
    USE_HYBRID: bool = os.getenv("USE_HYBRID", "true").lower() == "true"
    HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.5"))  # 向量检索权重
    HYBRID_RRF_K: int = int(os.getenv("HYBRID_RRF_K", "60"))  # RRF 参数
    BM25_PERSIST_DIR: str = os.getenv("BM25_PERSIST_DIR", "./vectorstore/bm25_index")
    VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "50"))  # 向量检索数量
    BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "50"))  # BM25 检索数量

    # ── Prompt 模板 ───────────────────────────────────
    SYSTEM_PROMPT: str = """你是一位专业的农业气候与资源数据专家助手。
你的知识来源于权威的农业科学文献、气象数据和作物种植手册。

回答时请遵循以下原则：
1. 优先基于提供的参考资料作答，如实引用数据来源
2. 若参考资料不足以回答问题，明确告知用户，不要编造数据
3. 涉及具体数值（温度、降水量、积温等）时，尽量给出范围而非单一数字
4. 回答要专业但易懂，适当解释专业术语
"""

    RAG_PROMPT_TEMPLATE: str = """请根据以下参考资料回答用户的问题。

【参考资料】
{context}

【用户问题】
{question}

【回答要求】
- 基于参考资料中的信息作答
- 如引用具体数据，说明来自哪份资料
- 如参考资料不足以完整回答，说明哪些部分是你的推断
"""


cfg = Config()