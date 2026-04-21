"""
BGE-Reranker-v2 重排序模块
对向量检索结果进行精排，提升相关性

原理：
- 向量检索是粗排（快速但不精确）
- Reranker是精排（计算query与每篇文档的深度相关性分数）
- 粗排取Top-K=50，精排后返回Top-K=5

使用方式:
    reranker = BGEReranker()
    results = reranker.rerank(
        query="水稻最佳播种期",
        documents=[{"content": "...", "score": 0.8}, ...],
        top_k=5
    )
"""
import os
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BGEReranker:
    """
    BGE-Reranker-v2 重排序器

    选择 bge-reranker-v2-m3 的原因：
    - 多语言支持（中文效果优秀）
    - 相比 v1 更快、更准确
    - 与 BGE-M3 向量模型配套使用
    """

    def __init__(
            self,
            model_name: str = "BAAI/bge-reranker-v2-m3",
            use_fp16: bool = True,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None  # 懒加载，避免启动时就占内存

    @property
    def model(self):
        """懒加载模型，第一次调用时才初始化"""
        if self._model is None:
            logger.info(f"加载 BGE-Reranker-v2: {self.model_name}")
            from FlagEmbedding import FlagReranker

            # 优先尝试本地模型
            local_path = "./bge-reranker-v2-m3"
            try:
                self._model = FlagReranker(local_path, use_fp16=self.use_fp16)
                logger.info(f"从本地加载 Reranker: {local_path}")
            except Exception:
                self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16)
                logger.info(f"从网络加载 Reranker: {self.model_name}")
        return self._model

    def rerank(
            self,
            query: str,
            documents: List[Dict],
            top_k: int = 5,
    ) -> List[Dict]:
        """
        重排序

        Args:
            query: 用户问题
            documents: 向量检索返回的文档列表，格式:
                [{"content": str, "metadata": dict, "score": float}, ...]
            top_k: 返回前 K 个结果

        Returns:
            重排序后的文档列表，score 已更新为 rerank_score:
            [
                {
                    "content": str,
                    "metadata": dict,
                    "rerank_score": float,  # 重排序分数
                    "original_score": float,  # 原始向量分数
                    "score": float,  # 最终分数（使用重排序分数）
                },
                ...
            ]
        """
        if not documents:
            logger.warning("没有文档需要重排序")
            return []

        if len(documents) <= top_k:
            logger.debug(f"文档数量 {len(documents)} <= top_k {top_k}，跳过重排序")
            return documents

        logger.info(f"开始重排序: {len(documents)} 条文档 -> {top_k} 条")

        # 构建 query-doc pairs
        pairs = [[query, doc["content"]] for doc in documents]

        # 获取重排序分数
        # normalize=True 使分数在 0~1 范围
        scores = self.model.compute_score(pairs, normalize=True)

        # 处理单个分数的情况（compute_score 可能返回 float 或 List）
        if isinstance(scores, float):
            scores = [scores]

        # 合并分数并排序
        reranked = []
        for doc, rerank_score in zip(documents, scores):
            reranked.append({
                **doc,
                "rerank_score": round(rerank_score, 4),
                "original_score": doc.get("score", 0),
                "score": round(rerank_score, 4),  # 使用重排序分数作为最终分数
            })

        # 按重排序分数降序排列
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 截取 top_k
        result = reranked[:top_k]

        logger.info(
            f"重排序完成: 最高分 {result[0]['rerank_score']:.4f}, "
            f"最低分 {result[-1]['rerank_score']:.4f}"
        )

        return result

    def compute_score(
            self,
            query: str,
            documents: List[Dict],
    ) -> List[float]:
        """
        仅计算分数，不排序（用于分析对比）

        Returns:
            每个文档的重排序分数列表
        """
        if not documents:
            return []

        pairs = [[query, doc["content"]] for doc in documents]
        scores = self.model.compute_score(pairs, normalize=True)

        if isinstance(scores, float):
            return [scores]
        return list(scores)


if __name__ == "__main__":
    # 测试
    reranker = BGEReranker()

    test_query = "水稻的最佳播种温度是多少？"
    test_docs = [
        {"content": "水稻播种适宜温度为15-20摄氏度，低于10度会影响发芽。", "score": 0.85},
        {"content": "小麦的最佳播种期在秋季，温度要求较低。", "score": 0.72},
        {"content": "农业气候资源包括热量、水分、光照等要素。", "score": 0.68},
    ]

    results = reranker.rerank(test_query, test_docs, top_k=2)
    for r in results:
        print(f"分数: {r['rerank_score']:.4f} (原: {r['original_score']:.4f})")
        print(f"内容: {r['content'][:50]}...")
        print()