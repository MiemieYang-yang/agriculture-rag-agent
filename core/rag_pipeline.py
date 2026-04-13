"""
RAG 核心链路
串联：Query → 检索 → 构建 Prompt → LLM 生成 → 返回结果

这是整个项目最核心的文件，面试时重点讲这里的设计思路
"""
from typing import List, Dict, Optional, Generator


from core.config import cfg
from core.embedder import BGEEmbedder
from core.llm_client import QwenClient
from core.vector_store import VectorStore
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
            top_k: int = cfg.RETRIEVAL_TOP_K,
            min_score: float = 0.4,  # 相似度低于此阈值的结果丢弃
    ):
        # 共享组件（避免重复初始化模型）
        embedder = BGEEmbedder()
        self.vector_store = vector_store or VectorStore(embedder=embedder)
        self.llm_client = llm_client or QwenClient()
        self.top_k = top_k
        self.min_score = min_score

        logger.info("RAG Pipeline 初始化完成")

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

    def _retrieve(self, question: str, where: Optional[Dict]) -> List[Dict]:
        """
        检索并过滤低质量结果

        设计考量：
        - min_score 过滤：避免把完全不相关的内容喂给 LLM（会导致幻觉）
        - 后续 Phase 2 可在这里加 Reranker 进行二次排序
        """
        if self.vector_store.count() == 0:
            logger.warning("知识库为空，请先运行 ingest.py 导入文档")
            return []

        results = self.vector_store.search(question, top_k=self.top_k, where=where)

        # 过滤相似度过低的结果
        filtered = [r for r in results if r["score"] >= self.min_score]
        if len(filtered) < len(results):
            logger.debug(f"过滤 {len(results) - len(filtered)} 条低质量结果（阈值: {self.min_score}）")

        return filtered

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