"""
BM25 稀疏检索模块
基于词频的传统检索方法，擅长精确关键词匹配

BM25 优势：
- 对精确关键词匹配效果好（如"水稻"、"积温"等专业术语）
- 计算速度快，无需向量编码
- 与向量检索互补：向量擅长语义理解，BM25擅长关键词匹配

使用方式:
    bm25 = BM25Store()
    bm25.add_documents(docs)  # 建立索引
    results = bm25.search("水稻播种温度", top_k=5)
"""
import os
import pickle
import hashlib
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BM25Store:
    """
    BM25 稀疏检索存储

    使用 rank_bm25 库实现 BM25 算法
    支持本地持久化索引
    """

    def __init__(
            self,
            persist_dir: str = "./vectorstore/bm25_index",
            k1: float = 1.5,  # BM25 参数：词频饱和度
            b: float = 0.75,  # BM25 参数：文档长度归一化
    ):
        self.persist_dir = persist_dir
        self.k1 = k1
        self.b = b

        # 索引数据
        self._bm25 = None
        self._documents = []  # 原始文档列表
        self._doc_ids = []  # 文档 ID 列表
        self._metadatas = []  # 元数据列表

        # 尝试加载已有索引
        self._load_index()

    @property
    def bm25(self):
        """懒加载 BM25 模型"""
        if self._bm25 is None and self._documents:
            self._build_index()
        return self._bm25

    def add_documents(self, docs: List[Any], batch_size: int = 1000) -> int:
        """
        添加文档到索引

        Args:
            docs: Document 对象列表（来自 document_processor）
            batch_size: 批处理大小

        Returns:
            实际添加的文档数
        """
        if not docs:
            logger.warning("没有文档需要写入 BM25 索引")
            return 0

        # 过滤已存在的文档
        existing_ids = set(self._doc_ids)
        to_add = []
        for doc in docs:
            doc_id = doc.doc_id or self._content_hash(doc.content)
            if doc_id not in existing_ids:
                to_add.append({
                    "id": doc_id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                })

        if not to_add:
            logger.info("所有文档已存在于 BM25 索引，跳过")
            return 0

        logger.info(f"开始添加 {len(to_add)} 条文档到 BM25 索引")

        # 添加到内部存储
        for item in to_add:
            self._doc_ids.append(item["id"])
            self._documents.append(item["content"])
            self._metadatas.append(item["metadata"])

        # 重建索引
        self._build_index()

        # 持久化
        self._save_index()

        logger.info(f"BM25 索引构建完成，共 {len(self._documents)} 条文档")
        return len(to_add)

    def search(
            self,
            query: str,
            top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        BM25 检索

        Args:
            query: 用户问题
            top_k: 返回最相关的 K 条

        Returns:
            [{"content": str, "metadata": dict, "score": float, "doc_id": str}, ...]
            score 是 BM25 分数，越高越相关
        """
        if self.bm25 is None or len(self._documents) == 0:
            logger.warning("BM25 索引为空")
            return []

        # 获取 BM25 分数
        scores = self.bm25.get_scores(query)
        top_k = min(top_k, len(scores))

        # 获取 top_k 索引
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # 构建返回结果
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:  # 过滤零分结果
                results.append({
                    "content": self._documents[idx],
                    "metadata": self._metadatas[idx],
                    "doc_id": self._doc_ids[idx],
                    "score": round(score, 4),
                })

        logger.debug(f"BM25 检索到 {len(results)} 条结果")
        return results

    def count(self) -> int:
        """返回索引中的文档数量"""
        return len(self._documents)

    def clear(self) -> None:
        """清空索引"""
        self._bm25 = None
        self._documents = []
        self._doc_ids = []
        self._metadatas = []

        # 删除持久化文件
        index_file = os.path.join(self.persist_dir, "bm25_index.pkl")
        if os.path.exists(index_file):
            os.remove(index_file)

        logger.warning("BM25 索引已清空")

    def _build_index(self):
        """构建 BM25 索引"""
        if not self._documents:
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("请先安装 rank_bm25: pip install rank_bm25")
            raise

        # 中文分词：简单按字符分割（可替换为 jieba）
        tokenized_docs = [self._tokenize(doc) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)

        logger.debug(f"BM25 索引构建完成: {len(self._documents)} 条文档")

    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词

        简单实现：按字符分割
        可替换为 jieba 等专业分词工具
        """
        # 移除空白字符，按字符分割
        # 对于中英文混合文本，这个简单方法效果还不错
        tokens = []
        word = ""
        for char in text:
            if char.isalpha():
                word += char
            else:
                if word:
                    tokens.append(word.lower())
                    word = ""
                # 中文字符单独作为 token
                if '\u4e00' <= char <= '\u9fff':
                    tokens.append(char)
        if word:
            tokens.append(word.lower())
        return tokens

    def _save_index(self):
        """持久化索引"""
        os.makedirs(self.persist_dir, exist_ok=True)
        index_file = os.path.join(self.persist_dir, "bm25_index.pkl")

        data = {
            "documents": self._documents,
            "doc_ids": self._doc_ids,
            "metadatas": self._metadatas,
            "k1": self.k1,
            "b": self.b,
        }

        with open(index_file, "wb") as f:
            pickle.dump(data, f)

        logger.debug(f"BM25 索引已保存: {index_file}")

    def _load_index(self):
        """加载已有索引"""
        index_file = os.path.join(self.persist_dir, "bm25_index.pkl")

        if not os.path.exists(index_file):
            logger.debug("未找到已有 BM25 索引")
            return

        try:
            with open(index_file, "rb") as f:
                data = pickle.load(f)

            self._documents = data.get("documents", [])
            self._doc_ids = data.get("doc_ids", [])
            self._metadatas = data.get("metadatas", [])

            logger.info(f"加载 BM25 索引: {len(self._documents)} 条文档")

            # 重建 BM25 模型（不序列化模型对象）
            if self._documents:
                self._build_index()

        except Exception as e:
            logger.warning(f"加载 BM25 索引失败: {e}")

    @staticmethod
    def _content_hash(text: str) -> str:
        """生成内容哈希"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    # 测试
    bm25 = BM25Store()

    # 模拟文档
    class MockDoc:
        def __init__(self, content, metadata=None):
            self.content = content
            self.metadata = metadata or {}
            self.doc_id = None

    test_docs = [
        MockDoc("水稻的最佳播种温度是15到20摄氏度，低于10度会影响发芽。", {"topic": "水稻"}),
        MockDoc("小麦的最佳播种期在秋季，温度要求较低。", {"topic": "小麦"}),
        MockDoc("农业气候资源包括热量、水分、光照等要素。", {"topic": "气候"}),
    ]

    bm25.add_documents(test_docs)

    # 检索
    results = bm25.search("水稻播种温度")
    for r in results:
        print(f"分数: {r['score']:.4f}")
        print(f"内容: {r['content']}")
        print()