"""
混合检索模块
融合 BM25 稀疏检索 + 向量稠密检索

融合策略：RRF（Reciprocal Rank Fusion）
- RRF 公式：score = Σ (1 / (k + rank))，k 通常取 60
- 优势：不依赖具体分数值，只依赖排名，两种检索结果可公平融合

使用方式:
    hybrid = HybridRetriever(vector_store, bm25_store)
    results = hybrid.search("水稻播种温度", top_k=10)
"""
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    混合检索器

    融合 BM25（关键词匹配）+ 向量检索（语义匹配）
    使用 RRF（Reciprocal Rank Fusion）算法融合结果
    """

    def __init__(
            self,
            vector_store,
            bm25_store,
            alpha: float = 0.5,  # 向量检索权重（BM25 权重为 1-alpha）
            rrf_k: int = 60,  # RRF 参数
            use_rrf: bool = True,  # 是否使用 RRF 融合
    ):
        """
        Args:
            vector_store: VectorStore 实例（向量检索）
            bm25_store: BM25Store 实例（BM25 检索）
            alpha: 向量检索权重，范围 0~1
                   - alpha=1.0: 纯向量检索
                   - alpha=0.0: 纯 BM25 检索
                   - alpha=0.5: 均衡融合
            rrf_k: RRF 参数，影响低排名结果的贡献度
            use_rrf: True 使用 RRF 融合，False 使用加权分数融合
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.use_rrf = use_rrf

        logger.info(
            f"混合检索器初始化: alpha={alpha}, "
            f"fusion={'RRF' if use_rrf else 'Weighted'}"
        )

    def search(
            self,
            query: str,
            top_k: int = 10,
            vector_top_k: int = 50,  # 向量检索数量
            bm25_top_k: int = 50,  # BM25 检索数量
            where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        混合检索

        流程：
        1. 分别执行向量检索和 BM25 检索
        2. 使用 RRF 融合结果
        3. 返回融合后的 top_k 结果

        Args:
            query: 用户问题
            top_k: 最终返回数量
            vector_top_k: 向量检索数量（粗排）
            bm25_top_k: BM25 检索数量（粗排）
            where: 元数据过滤（仅向量检索支持）

        Returns:
            [{"content": str, "metadata": dict, "score": float, "vector_score": float, "bm25_score": float}, ...]
        """
        # Step 1: 向量检索
        logger.info(f"向量检索: top_k={vector_top_k}")
        vector_results = self.vector_store.search(query, top_k=vector_top_k, where=where)

        # Step 2: BM25 检索
        logger.info(f"BM25 检索: top_k={bm25_top_k}")
        bm25_results = self.bm25_store.search(query, top_k=bm25_top_k)

        # Step 3: 融合结果
        if self.use_rrf:
            fused = self._rrf_fusion(vector_results, bm25_results, top_k)
        else:
            fused = self._weighted_fusion(vector_results, bm25_results, top_k)

        logger.info(f"混合检索完成: {len(fused)} 条结果")
        return fused

    def search_by_vector(
            self,
            query_vector: List[float],
            query_text: str,  # 用于 BM25 检索
            top_k: int = 10,
            vector_top_k: int = 50,
            bm25_top_k: int = 50,
            where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        混合检索（用于 HyDE 场景）

        Args:
            query_vector: 已编码的向量（如 HyDE 生成的假设答案向量）
            query_text: 原始问题文本（用于 BM25 检索）
        """
        # 向量检索
        vector_results = self.vector_store.search_by_vector(
            query_vector, top_k=vector_top_k, where=where
        )

        # BM25 检索（用原始问题）
        bm25_results = self.bm25_store.search(query_text, top_k=bm25_top_k)

        # 融合
        if self.use_rrf:
            fused = self._rrf_fusion(vector_results, bm25_results, top_k)
        else:
            fused = self._weighted_fusion(vector_results, bm25_results, top_k)

        return fused

    def _rrf_fusion(
            self,
            vector_results: List[Dict],
            bm25_results: List[Dict],
            top_k: int,
    ) -> List[Dict]:
        """
        RRF（Reciprocal Rank Fusion）融合

        公式：RRF_score(d) = Σ (1 / (k + rank(d)))

        优势：
        - 不依赖分数的绝对值，只依赖排名
        - 两种检索系统的分数范围不同也能公平融合
        """
        # 按文档内容去重，记录排名
        doc_rank_vector = {}  # {content: rank}
        doc_rank_bm25 = {}

        # 记录向量检索排名
        for rank, doc in enumerate(vector_results, 1):
            content = doc["content"]
            if content not in doc_rank_vector:
                doc_rank_vector[content] = rank

        # 记录 BM25 检索排名
        for rank, doc in enumerate(bm25_results, 1):
            content = doc["content"]
            if content not in doc_rank_bm25:
                doc_rank_bm25[content] = rank

        # 合并所有文档
        all_docs = {}
        for doc in vector_results:
            content = doc["content"]
            if content not in all_docs:
                all_docs[content] = {
                    "content": content,
                    "metadata": doc["metadata"],
                    "vector_score": doc.get("score", 0),
                    "bm25_score": 0,
                    "vector_rank": doc_rank_vector.get(content, 999),
                    "bm25_rank": 999,
                }

        for doc in bm25_results:
            content = doc["content"]
            if content in all_docs:
                all_docs[content]["bm25_score"] = doc.get("score", 0)
                all_docs[content]["bm25_rank"] = doc_rank_bm25.get(content, 999)
            else:
                all_docs[content] = {
                    "content": content,
                    "metadata": doc["metadata"],
                    "vector_score": 0,
                    "bm25_score": doc.get("score", 0),
                    "vector_rank": 999,
                    "bm25_rank": doc_rank_bm25.get(content, 999),
                }

        # 计算 RRF 分数
        for content, doc_info in all_docs.items():
            rrf_score = 0
            if doc_info["vector_rank"] < 999:
                rrf_score += 1 / (self.rrf_k + doc_info["vector_rank"])
            if doc_info["bm25_rank"] < 999:
                rrf_score += 1 / (self.rrf_k + doc_info["bm25_rank"])
            doc_info["rrf_score"] = round(rrf_score, 6)
            doc_info["score"] = doc_info["rrf_score"]

        # 按 RRF 分数排序
        sorted_docs = sorted(
            all_docs.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )[:top_k]

        logger.debug(
            f"RRF 融合: 向量 {len(vector_results)} + BM25 {len(bm25_results)} "
            f"-> {len(sorted_docs)}"
        )

        return sorted_docs

    def _weighted_fusion(
            self,
            vector_results: List[Dict],
            bm25_results: List[Dict],
            top_k: int,
    ) -> List[Dict]:
        """
        加权分数融合

        公式：score = alpha * vector_score + (1-alpha) * bm25_score_normalized

        需要注意：
        - BM25 分数范围可能很大，需要归一化
        - 向量分数范围在 0~1（余弦相似度）
        """
        # 归一化 BM25 分数
        bm25_max_score = max(
            (doc.get("score", 0) for doc in bm25_results),
            default=1
        )

        # 合并文档
        all_docs = {}

        for doc in vector_results:
            content = doc["content"]
            all_docs[content] = {
                "content": content,
                "metadata": doc["metadata"],
                "vector_score": doc.get("score", 0),
                "bm25_score": 0,
            }

        for doc in bm25_results:
            content = doc["content"]
            bm25_normalized = doc.get("score", 0) / bm25_max_score if bm25_max_score > 0 else 0
            if content in all_docs:
                all_docs[content]["bm25_score"] = bm25_normalized
            else:
                all_docs[content] = {
                    "content": content,
                    "metadata": doc["metadata"],
                    "vector_score": 0,
                    "bm25_score": bm25_normalized,
                }

        # 计算加权分数
        for doc_info in all_docs.values():
            weighted_score = (
                    self.alpha * doc_info["vector_score"] +
                    (1 - self.alpha) * doc_info["bm25_score"]
            )
            doc_info["weighted_score"] = round(weighted_score, 4)
            doc_info["score"] = doc_info["weighted_score"]

        # 排序
        sorted_docs = sorted(
            all_docs.values(),
            key=lambda x: x["weighted_score"],
            reverse=True
        )[:top_k]

        return sorted_docs

    def count(self) -> int:
        """返回向量库中的文档数量"""
        return self.vector_store.count()


if __name__ == "__main__":
    # 测试融合算法
    vector_results = [
        {"content": "doc1", "metadata": {}, "score": 0.9},
        {"content": "doc2", "metadata": {}, "score": 0.8},
        {"content": "doc3", "metadata": {}, "score": 0.7},
    ]

    bm25_results = [
        {"content": "doc2", "metadata": {}, "score": 5.0},
        {"content": "doc3", "metadata": {}, "score": 4.0},
        {"content": "doc4", "metadata": {}, "score": 3.0},
    ]

    # 模拟 RRF 融合
    retriever = HybridRetriever(None, None, use_rrf=True)
    fused = retriever._rrf_fusion(vector_results, bm25_results, 5)

    for doc in fused:
        print(f"内容: {doc['content']}, RRF分数: {doc['rrf_score']:.6f}")