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