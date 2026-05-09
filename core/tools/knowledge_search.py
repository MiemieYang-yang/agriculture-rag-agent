"""
知识库检索工具
封装 RAG Pipeline 作为 Agent 可调用的工具
"""
from typing import Dict, List, Optional
import logging

from core.tools.base import BaseTool, ToolResult
from core.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class KnowledgeSearchTool(BaseTool):
    """知识库检索工具"""

    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        """
        初始化知识库检索工具

        Args:
            rag_pipeline: RAG Pipeline 实例，如不传入则延迟初始化
        """
        self._rag_pipeline = rag_pipeline

    @property
    def rag_pipeline(self) -> RAGPipeline:
        """延迟初始化 RAG Pipeline"""
        if self._rag_pipeline is None:
            self._rag_pipeline = RAGPipeline()
        return self._rag_pipeline

    @property
    def name(self) -> str:
        return "knowledge_search"

    @property
    def description(self) -> str:
        return "从农业知识库中检索相关信息。适用于查询农业技术、作物种植、气候区划、土壤条件等专业知识问题。当用户问题涉及具体农业知识时优先使用此工具。"

    @property
    def parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询语句，应该是一个具体的问题或关键词"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量，默认5条",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    def execute(self, **kwargs) -> ToolResult:
        """
        执行知识库检索

        Args:
            query: 检索查询
            top_k: 返回结果数量
        """
        query = kwargs.get("query", "")
        top_k = kwargs.get("top_k", 5)

        if not query:
            return ToolResult(
                name=self.name,
                success=False,
                summary="",
                error="检索查询不能为空",
            )

        try:
            # 调用 RAG Pipeline 检索（包含 LLM 生成的回答）
            result = self.rag_pipeline.query(query)

            # 整理检索结果
            sources = result.get("sources", [])
            retrieved_count = result.get("retrieved_count", 0)
            answer = result.get("answer", "")

            # 构建返回数据
            search_result = {
                "query": query,
                "retrieved_count": retrieved_count,
                "contexts": [],
                "sources": sources,
                "answer": answer,  # 包含 RAG 生成的完整回答
            }

            # 如果有检索结果，提取文本内容
            if retrieved_count > 0:
                search_result["contexts"] = [
                    {
                        "content": src.get("snippet", ""),
                        "source": src.get("filename", ""),
                        "score": src.get("score", 0),
                    }
                    for src in sources
                ]

            # 打印详细检索结果（调试用）
            logger.info(f"知识库检索完成: 查询='{query[:30]}...', 结果数={retrieved_count}")
            if sources:
                logger.info("=" * 60)
                logger.info("【检索到的文献】")
                for i, src in enumerate(sources, 1):
                    logger.info(f"  [{i}] 文件: {src.get('filename', '未知')}")
                    logger.info(f"      相关度: {src.get('score', 0):.4f}")
                    logger.info(f"      页码: {src.get('page', '无')}")
                    logger.info(f"      内容: {src.get('snippet', '')[:200]}...")
                logger.info("=" * 60)
            logger.info(f"【RAG 生成的回答】\n{answer[:500]}...")

            # 生成摘要（包含完整回答，让 Agent 直接使用）
            summary = self._generate_summary(query, retrieved_count, sources, answer)

            return ToolResult(
                name=self.name,
                success=True,
                summary=summary,
                data=search_result,
            )

        except Exception as e:
            logger.error(f"知识库检索失败: {e}")
            return ToolResult(
                name=self.name,
                success=False,
                summary="",
                error=f"知识库检索失败: {str(e)}",
            )

    def _generate_summary(self, query: str, count: int, sources: List[Dict], answer: str = "") -> str:
        """生成给 LLM 的检索摘要"""
        if count == 0:
            return f"知识库中未找到与「{query}」相关的信息。请基于你的专业知识回答。"

        # 直接返回 RAG 生成的完整回答，让 Agent 可以直接使用
        if answer:
            return answer

        # 备用：如果没有 answer，返回检索结果摘要
        summary_parts = [f"知识库检索到 {count} 条相关结果："]
        for i, src in enumerate(sources[:3], 1):
            filename = src.get("filename", "未知来源")
            snippet = src.get("snippet", "")[:100]
            summary_parts.append(f"{i}. {filename}: {snippet}...")
        return "\n".join(summary_parts)

    def search_only(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        仅执行检索，不调用 LLM 生成回答

        用于需要获取原始检索结果的场景
        """
        try:
            # 使用内部的 _retrieve 方法
            results = self.rag_pipeline._retrieve(query)
            return results[:top_k]
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
