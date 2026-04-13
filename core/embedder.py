"""
向量化模块（BGE-M3）
BGE-M3 是多语言、多粒度模型，支持中英文，检索效果优于 OpenAI text-embedding-ada

关键点：
- 查询（query）和文档（passage）需要用不同的前缀，这是 BGE 系列的规范
- 批量编码时分 batch，避免 OOM
- 首次使用会自动从 HuggingFace 下载模型（~2GB），之后缓存本地
"""
import os
from typing import List

import numpy as np

from core.config import cfg
import logging

logger = logging.getLogger(__name__)
class BGEEmbedder:
    """
    BGE-M3 向量编码器

    使用方式:
        embedder = BGEEmbedder()

        # 编码查询（用于检索时的 query 向量）
        q_vec = embedder.encode_query("四川水稻最佳播种温度")

        # 编码文档（用于建库时的 passage 向量）
        doc_vecs = embedder.encode_documents(["文档1内容", "文档2内容"])
    """

    def __init__(self, model_name: str = cfg.BGE_MODEL_NAME, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None  # 懒加载，第一次调用 encode 时才初始化

    # @property
    # def model(self):
    #     """懒加载：第一次使用时才加载模型，避免启动时就占用大量内存"""
    #     if self._model is None:
    #         logger.info(f"加载 BGE-M3 模型: {self.model_name}（首次加载需要下载，约 2GB）")
    #         os.environ["FLAG_NO_RERANKER"] = "1"  # 禁用冲突模块
    #         from FlagEmbedding import BGEM3FlagModel
    #         self._model = BGEM3FlagModel(
    #             self.model_name,
    #             use_fp16=True,  # 半精度推理，速度 ~2x，精度损失可忽略
    #         )
    #         logger.info("BGE-M3 加载完成")
    #     return self._model
    @property
    def model(self):
        """懒加载：优先本地模型，本地不存在则自动从网上下载"""
        if self._model is None:
            os.environ["FLAG_NO_RERANKER"] = "1"
            from FlagEmbedding import BGEM3FlagModel

            # 优先尝试加载本地模型
            local_model_path = "./bge-m3"
            try:
                logger.info(f"🔵 尝试加载本地模型: {local_model_path}")
                self._model = BGEM3FlagModel(
                    local_model_path,
                    use_fp16=True
                )
                logger.info("✅ 本地模型加载成功！")
            except Exception as e:
                # 本地不存在 → 自动在线下载
                logger.warning(f"⚠️ 本地模型不存在，将自动在线下载: {self.model_name}（约2GB）")
                self._model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=True
                )
                logger.info("✅ 在线模型加载完成！")

        return self._model

    def encode_query(self, query: str) -> List[float]:
        """
        编码单条查询
        BGE 规范：query 前加 "Represent this sentence for searching relevant passages: "
        这个前缀会显著提升检索效果，不能省略
        """
        result = self.model.encode(
            [query],
            batch_size=1,
            max_length=512,
        )
        vec = result["dense_vecs"][0]
        return self._normalize(vec).tolist()

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量编码文档
        文档端不加前缀（BGE-M3 的 passage 侧不需要）
        """
        if not texts:
            return []

        all_vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            result = self.model.encode(
                batch,
                batch_size=self.batch_size,
                max_length=512,
            )
            vecs = result["dense_vecs"]
            all_vecs.extend([self._normalize(v).tolist() for v in vecs])
            logger.debug(f"已编码 {min(i + self.batch_size, len(texts))}/{len(texts)} 条")

        return all_vecs

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L2 归一化，使余弦相似度 = 点积，ChromaDB 用点积更快"""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def get_dimension(self) -> int:
        """返回向量维度（BGE-M3 dense 模式为 1024）"""
        test = self.encode_query("test")
        return len(test)