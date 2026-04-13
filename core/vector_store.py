"""
向量库管理模块（ChromaDB）
负责：建库、写入、检索、去重

ChromaDB 选择理由：
- 本地持久化，不需要额外部署服务
- 支持元数据过滤（后续 Phase 3 做 Agent 时很有用）
- Python 原生，接口简洁
"""
import hashlib
import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from core.config import cfg
from core.document_processor import Document
from core.embedder import BGEEmbedder
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """
    ChromaDB 向量库封装

    使用方式:
        vs = VectorStore()

        # 写入文档
        vs.add_documents(chunks)

        # 检索
        results = vs.search("四川盆地水稻播种温度", top_k=5)
    """

    def __init__(
            self,
            embedder: Optional[BGEEmbedder] = None,
            persist_dir: str = cfg.CHROMA_PERSIST_DIR,
            collection_name: str = cfg.CHROMA_COLLECTION,
    ):
        self.embedder = embedder or BGEEmbedder()
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # 初始化 ChromaDB 客户端（持久化到本地）
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # 获取或创建 collection
        # cosine 距离适合归一化后的 BGE 向量
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB 已连接: {persist_dir}/{collection_name}"
            f"（当前文档数: {self.collection.count()}）"
        )

    # ── 写入 ─────────────────────────────────────────────────────────────────

    def add_documents(self, docs: List[Document], batch_size: int = 64) -> int:
        """
        批量写入文档
        - 自动对内容做 MD5 去重，避免重复写入
        - 返回实际写入的文档数
        """
        if not docs:
            logger.warning("没有文档需要写入")
            return 0

        # 过滤已存在的文档（用内容 MD5 作为 ID）
        new_docs = []
        for doc in docs:
            doc_id = self._content_hash(doc.content) if not doc.doc_id else doc.doc_id
            doc.doc_id = doc_id
            new_docs.append(doc)

        # 检查哪些 ID 已存在
        existing_ids = set()
        try:
            existing = self.collection.get(ids=[d.doc_id for d in new_docs])
            existing_ids = set(existing["ids"])
        except Exception:
            pass

        to_add = [d for d in new_docs if d.doc_id not in existing_ids]
        if not to_add:
            logger.info("所有文档已存在，跳过写入")
            return 0

        logger.info(f"开始写入 {len(to_add)} 条文档（跳过 {len(new_docs) - len(to_add)} 条重复）")

        # 分批处理：先编码，再写入
        written = 0
        for i in range(0, len(to_add), batch_size):
            batch = to_add[i: i + batch_size]
            texts = [d.content for d in batch]
            ids = [d.doc_id for d in batch]
            metadatas = [self._serialize_metadata(d.metadata) for d in batch]

            # 编码向量
            embeddings = self.embedder.encode_documents(texts)

            # 写入 ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            written += len(batch)
            logger.info(f"已写入 {written}/{len(to_add)}")

        logger.info(f"写入完成，库中共 {self.collection.count()} 条文档")
        return written

    # ── 检索 ─────────────────────────────────────────────────────────────────

    def search(
            self,
            query: str,
            top_k: int = cfg.RETRIEVAL_TOP_K,
            where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        语义检索

        Args:
            query:  用户问题
            top_k:  返回最相关的 K 条
            where:  元数据过滤条件，如 {"file_type": "pdf"}（Phase 3 Agent 会用到）

        Returns:
            [{"content": str, "metadata": dict, "score": float}, ...]
            score 是余弦相似度，越高越相关（范围 0~1）
        """
        query_vec = self.embedder.encode_query(query)

        kwargs = dict(
            query_embeddings=[query_vec],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        result = self.collection.query(**kwargs)

        # 整理返回格式，距离转相似度
        retrieved = []
        for doc, meta, dist in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
        ):
            retrieved.append({
                "content": doc,
                "metadata": meta,
                "score": round(1 - dist, 4),  # cosine distance → similarity
            })

        logger.debug(f"检索到 {len(retrieved)} 条，最高分: {retrieved[0]['score'] if retrieved else 'N/A'}")
        return retrieved

    # ── 状态查询 ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        """清空知识库（谨慎使用）"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"知识库 {self.collection_name} 已清空")

    # ── 工具方法 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _content_hash(text: str) -> str:
        """用内容 MD5 作为文档唯一 ID，防止重复入库"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_metadata(meta: dict) -> dict:
        """ChromaDB 元数据只支持 str/int/float/bool，需要转换"""
        serialized = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                serialized[k] = v
            else:
                serialized[k] = str(v)
        return serialized