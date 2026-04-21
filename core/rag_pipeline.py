"""
RAG 核心链路
串联：Query → 检索 → 构建 Prompt → LLM 生成 → 返回结果

这是整个项目最核心的文件，面试时重点讲这里的设计思路

Phase 2 检索优化：
- HyDE：假设文档嵌入
- Reranker：BGE-Reranker-v2 重排序
- 混合检索：BM25 + 向量检索 RRF 融合
"""
from typing import List, Dict, Optional, Generator, Tuple, Union


from core.config import cfg
from core.embedder import BGEEmbedder
from core.llm_client import QwenClient
from core.vector_store import VectorStore
from core.reranker import BGEReranker
from core.bm25_store import BM25Store
from core.hybrid_retriever import HybridRetriever
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    RAG 问答链路

    完整流程：
    1. 接收用户 Query
    2. BGE-M3 编码 Query → 向量
    3. ChromaDB 语义检索 → Top-K 相关文档
    4. 拼接检索结果 + 用户问题 → Prompt
    5. 调用 Qwen → 生成最终回答
    6. 返回回答 + 引用来源

    使用方式:
        pipeline = RAGPipeline()
        result = pipeline.query("四川盆地水稻最佳播种期是什么时候？")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
            self,
            vector_store: Optional[VectorStore] = None,
            llm_client: Optional[QwenClient] = None,
            embedder: Optional[BGEEmbedder] = None,
            reranker: Optional[BGEReranker] = None,
            bm25_store: Optional[BM25Store] = None,
            hybrid_retriever: Optional[HybridRetriever] = None,
            top_k: int = cfg.RETRIEVAL_TOP_K,
            min_score: float = 0.4,  # 相似度低于此阈值的结果丢弃
            use_hyde: bool = cfg.USE_HYDE,
            use_reranker: bool = cfg.USE_RERANKER,
            use_hybrid: bool = cfg.USE_HYBRID,
            rerank_top_k: int = cfg.RERANK_TOP_K,
            vector_top_k: int = cfg.VECTOR_TOP_K,
            bm25_top_k: int = cfg.BM25_TOP_K,
            hybrid_alpha: float = cfg.HYBRID_ALPHA,
    ):
        # 共享组件（避免重复初始化模型）
        self.embedder = embedder or BGEEmbedder()
        self.vector_store = vector_store or VectorStore(embedder=self.embedder)
        self.llm_client = llm_client or QwenClient()
        self.reranker = reranker
        self.bm25_store = bm25_store
        self.hybrid_retriever = hybrid_retriever
        self.top_k = top_k
        self.min_score = min_score
        self.use_hyde = use_hyde
        self.use_reranker = use_reranker
        self.use_hybrid = use_hybrid
        self.rerank_top_k = rerank_top_k
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.hybrid_alpha = hybrid_alpha

        # 初始化 Reranker（如果启用）
        if self.use_reranker and self.reranker is None:
            self.reranker = BGEReranker()

        # 初始化混合检索（如果启用）
        if self.use_hybrid and self.hybrid_retriever is None:
            self.bm25_store = bm25_store or BM25Store()
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                bm25_store=self.bm25_store,
                alpha=hybrid_alpha,
            )

        logger.info(
            f"RAG Pipeline 初始化完成 "
            f"(hyde={use_hyde}, reranker={use_reranker}, hybrid={use_hybrid}, top_k={top_k})"
        )

    # ── 主接口 ───────────────────────────────────────────────────────────────

    def query(
            self,
            question: str,
            history: Optional[List[Dict]] = None,
            where: Optional[Dict] = None,
    ) -> Dict:
        """
        RAG 问答（同步版）

        Args:
            question: 用户问题
            history:  对话历史（多轮对话时传入）
            where:    元数据过滤，如只查 PDF 资料: {"file_type": "pdf"}

        Returns:
            {
                "answer":   str,           # 最终回答
                "sources":  List[dict],    # 引用的原文片段及来源
                "question": str,           # 原始问题
                "retrieved_count": int,    # 实际检索到的文档数
            }
        """
        logger.info(f"收到问题: {question[:50]}{'...' if len(question) > 50 else ''}")

        # Step 1: 检索相关文档
        retrieved = self._retrieve(question, where)

        # Step 2: 如果知识库完全没有相关内容，走 fallback
        if not retrieved:
            logger.warning("知识库无匹配，使用模型内置知识回答")
            answer = self.llm_client.chat(
                question,
                history=history,
                system_prompt=cfg.SYSTEM_PROMPT + "\n注意：当前知识库暂无相关资料，请基于你的专业知识作答，并提醒用户这不是基于本地文献的回答。",
            )
            return {
                "answer": answer,
                "sources": [],
                "question": question,
                "retrieved_count": 0,
            }

        # Step 3: 构建 RAG Prompt
        prompt = self._build_prompt(question, retrieved)

        # Step 4: 调用 LLM
        answer = self.llm_client.chat(
            prompt,
            system_prompt=cfg.SYSTEM_PROMPT,
            history=history,
        )

        # Step 5: 整理来源信息（供前端展示引用）
        sources = self._format_sources(retrieved)

        logger.info(f"回答完成，引用 {len(sources)} 条文献")
        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "retrieved_count": len(retrieved),
        }

    def query_stream(
            self,
            question: str,
            history: Optional[List[Dict]] = None,
            where: Optional[Dict] = None,
    ) -> Generator[str, None, None]:
        """
        RAG 问答（流式版）—— Phase 4 工程化时用
        检索是同步的，生成阶段逐 token 返回

        使用方式:
            for token in pipeline.query_stream("问题"):
                print(token, end="", flush=True)
        """
        retrieved = self._retrieve(question, where)
        prompt = self._build_prompt(question, retrieved) if retrieved else question

        system = cfg.SYSTEM_PROMPT if retrieved else (
                cfg.SYSTEM_PROMPT + "\n注意：知识库无相关资料，请基于专业知识作答。"
        )

        yield from self.llm_client.chat_stream(prompt, system_prompt=system, history=history)

    # ── 私有方法 ─────────────────────────────────────────────────────────────

    def _retrieve(
            self,
            question: str,
            where: Optional[Dict] = None,
            return_hyde_answer: bool = False,
    ) -> Union[List[Dict], Tuple[List[Dict], Optional[str]]]:
        """
        检索并过滤低质量结果

        流程（Phase 2 优化后）：
        1. [可选] HyDE 生成假设答案，用假设答案检索
        2. [可选] 混合检索（BM25 + 向量检索 RRF 融合）
        3. 或纯向量检索（粗排），获取 rerank_top_k 条结果
        4. [可选] Reranker 重排序（精排），返回最终 top_k 条

        设计考量：
        - min_score 过滤：避免把完全不相关的内容喂给 LLM（会导致幻觉）
        - HyDE：问题可能表述模糊，假设答案更接近目标文档的表述风格
        - 混合检索：BM25 擅长关键词匹配，向量擅长语义匹配，融合效果更好
        - Reranker：向量检索是粗排，Reranker 做精排提升相关性
        """
        # 检查知识库是否为空
        vec_count = self.vector_store.count()
        bm25_count = self.bm25_store.count() if self.bm25_store else 0

        if vec_count == 0 and (not self.use_hybrid or bm25_count == 0):
            logger.warning("知识库为空，请先运行 ingest.py 导入文档")
            return [] if not return_hyde_answer else ([], None)

        hyde_answer = None
        results = []

        # Step 1: 确定检索方式
        if self.use_hyde:
            # HyDE 检索
            results, hyde_answer = self._hyde_retrieve(question, where)
        elif self.use_hybrid and self.hybrid_retriever:
            # 混合检索（BM25 + 向量）
            retrieval_k = self.rerank_top_k if self.use_reranker else self.top_k
            logger.info(f"混合检索: vector_top_k={self.vector_top_k}, bm25_top_k={self.bm25_top_k}")

            if self.vector_top_k > 0 and vec_count > 0:
                results = self.hybrid_retriever.search(
                    question,
                    top_k=retrieval_k,
                    vector_top_k=self.vector_top_k,
                    bm25_top_k=self.bm25_top_k,
                    where=where,
                )
            elif bm25_count > 0:
                # 仅 BM25（向量库为空）
                results = self.bm25_store.search(question, top_k=retrieval_k)

            # 过滤低质量结果
            results = [r for r in results if r.get("score", 0) >= self.min_score]
        else:
            # 常规向量检索
            retrieval_k = self.rerank_top_k if self.use_reranker else self.top_k
            results = self.vector_store.search(question, top_k=retrieval_k, where=where)

            # 过滤低质量结果
            results = [r for r in results if r["score"] >= self.min_score]

        # Step 2: Reranker 精排
        if self.use_reranker and self.reranker and len(results) > self.top_k:
            logger.info(f"Reranker 精排: {len(results)} -> {self.top_k}")
            results = self.reranker.rerank(question, results, top_k=self.top_k)

            # Reranker 结果也需要过滤（可能重排后分数低于阈值）
            results = [r for r in results if r.get("rerank_score", r.get("score", 0)) >= self.min_score]

        logger.info(f"最终检索结果: {len(results)} 条")

        if return_hyde_answer:
            return results, hyde_answer
        return results

    def _hyde_retrieve(
            self,
            question: str,
            where: Optional[Dict] = None,
    ) -> Tuple[List[Dict], str]:
        """
        HyDE (Hypothetical Document Embedding) 检索

        原理：
        1. LLM 生成一个假设性答案（hypothetical answer）
        2. 将假设答案编码为向量
        3. 用假设答案向量检索（而非原始问题向量）

        优势：问题可能表述模糊，但假设答案更接近目标文档的表述风格

        Returns:
            (检索结果列表, 假设答案文本)
        """
        logger.info(f"HyDE 模式: 生成假设答案...")

        # Step 1: 生成假设答案
        hyde_prompt = cfg.HYDE_PROMPT_TEMPLATE.format(question=question)
        hypothetical_answer = self.llm_client.chat(
            hyde_prompt,
            system_prompt="你是一位农业气候领域的专家，请生成专业的假设答案。",
            temperature=0.7,  # 稍微提高随机性，鼓励生成多样答案
            max_tokens=512,
        )
        logger.debug(f"假设答案: {hypothetical_answer[:100]}...")

        # Step 2: 编码假设答案
        hyde_vector = self.embedder.encode_query(hypothetical_answer)

        # Step 3: 用假设答案向量检索
        retrieval_k = self.rerank_top_k if self.use_reranker else self.top_k

        # 如果启用混合检索，用混合检索器
        if self.use_hybrid and self.hybrid_retriever:
            logger.info("HyDE + 混合检索")
            results = self.hybrid_retriever.search_by_vector(
                hyde_vector,
                query_text=question,  # BM25 用原始问题
                top_k=retrieval_k,
                vector_top_k=self.vector_top_k,
                bm25_top_k=self.bm25_top_k,
                where=where,
            )
        else:
            results = self.vector_store.search_by_vector(
                hyde_vector,
                top_k=retrieval_k,
                where=where,
            )

        # 过滤低质量结果
        filtered = [r for r in results if r.get("score", 0) >= self.min_score]

        logger.info(f"HyDE 检索到 {len(filtered)} 条结果")
        return filtered, hypothetical_answer

    def _build_prompt(self, question: str, retrieved: List[Dict]) -> str:
        """
        构建 RAG Prompt

        Prompt 设计要点：
        1. 每条检索结果标注编号和来源，方便模型引用
        2. 相似度高的排在前面（已由检索保证）
        3. 明确告知模型"基于以下资料回答"，减少幻觉
        """
        # 拼接检索到的上下文
        context_parts = []
        for i, doc in enumerate(retrieved, 1):
            source = doc["metadata"].get("filename", "未知来源")
            page = doc["metadata"].get("page", "")
            page_str = f" 第{page}页" if page else ""
            score = doc["score"]

            context_parts.append(
                f"【资料{i}】来源：{source}{page_str}（相关度: {score:.2f}）\n{doc['content']}"
            )

        context = "\n\n".join(context_parts)

        prompt = cfg.RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        logger.debug(f"Prompt 长度: {len(prompt)} 字符，使用 {len(retrieved)} 条参考资料")
        return prompt

    @staticmethod
    def _format_sources(retrieved: List[Dict]) -> List[Dict]:
        """整理来源信息，供 API 返回给前端展示"""
        sources = []
        seen = set()  # 去重同一文件的多个 chunk
        for doc in retrieved:
            filename = doc["metadata"].get("filename", "未知")
            if filename not in seen:
                seen.add(filename)
                sources.append({
                    "filename": filename,
                    "source": doc["metadata"].get("source", ""),
                    "page": doc["metadata"].get("page"),
                    "score": doc["score"],
                    "snippet": doc["content"][:150] + "...",  # 前150字作为预览
                })
        return sources