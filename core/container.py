from loguru import logger


class Container:

    def __init__(self):
        self._embedder = None
        self._vector_store = None
        self._bm25_store = None
        self._reranker = None
        self._hybrid_retriever = None
        self._rag_pipeline = None
        self._llm_client = None
        self._agent = None

    # ── 数据层 ──────────────────────────────────────────────

    @property
    def embedder(self):
        if self._embedder is None:
            from core.embedder import BGEEmbedder
            logger.info("初始化 BGEEmbedder...")
            self._embedder = BGEEmbedder()
        return self._embedder

    @property
    def vector_store(self):
        if self._vector_store is None:
            from core.vector_store import VectorStore
            self._vector_store = VectorStore(embedder=self.embedder)
        return self._vector_store

    @property
    def bm25_store(self):
        if self._bm25_store is None:
            from core.bm25_store import BM25Store
            self._bm25_store = BM25Store()
        return self._bm25_store

    @property
    def reranker(self):
        if self._reranker is None:
            from core.reranker import BGEReranker
            self._reranker = BGEReranker()
        return self._reranker

    @property
    def hybrid_retriever(self):
        if self._hybrid_retriever is None:
            from core.hybrid_retriever import HybridRetriever
            self._hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                bm25_store=self.bm25_store,
            )
        return self._hybrid_retriever

    # ── 检索层 ──────────────────────────────────────────────

    @property
    def rag_pipeline(self):
        if self._rag_pipeline is None:
            from core.rag_pipeline import RAGPipeline
            self._rag_pipeline = RAGPipeline(
                hybrid_retriever=self.hybrid_retriever,
                llm_client=self.llm_client,
            )
        return self._rag_pipeline

    # ── LLM 层 ──────────────────────────────────────────────

    @property
    def llm_client(self):
        if self._llm_client is None:
            from core.llm_client import QwenClient
            self._llm_client = QwenClient()
        return self._llm_client

    # ── Agent 层 ─────────────────────────────────────────────

    @property
    def agent(self):
        if self._agent is None:
            from core.agent.agent import AgricultureAgent
            self._agent = AgricultureAgent(
                rag_pipeline=self.rag_pipeline,
                llm_client=self.llm_client,
            )
        return self._agent

    # ── 生命周期 ─────────────────────────────────────────────

    async def startup(self):
        """
        FastAPI lifespan startup 时调用
        预热必要依赖，避免首次请求延迟过高
        """
        logger.info("Container startup：预热 LLM 客户端...")
        _ = self.llm_client
        logger.info("Container startup 完成")

    async def shutdown(self):
        """
        FastAPI lifespan shutdown 时调用
        优雅释放资源
        """
        self._embedder = None
        self._vector_store = None
        self._agent = None
        logger.info("Container shutdown 完成")


# 全局单例
container = Container()