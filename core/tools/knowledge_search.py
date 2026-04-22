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
                success=False,
                data=None,
                error_message="检索查询不能为空",
            )

        try:
            # 调用 RAG Pipeline 检索
            result = self.rag_pipeline.query(query)

            # 整理检索结果
            sources = result.get("sources", [])
            retrieved_count = result.get("retrieved_count", 0)

            # 构建返回数据
            search_result = {
                "query": query,
                "retrieved_count": retrieved_count,
                "contexts": [],
                "sources": sources,
            }

            # 如果有检索结果，提取文本内容
            if retrieved_count > 0:
                # 从 RAG 结果中提取上下文
                # 注意：RAGPipeline.query() 返回的 answer 包含了 LLM 生成的回答
                # 这里我们只返回检索到的原始内容
                search_result["contexts"] = [
                    {
                        "content": src.get("snippet", ""),
                        "source": src.get("filename", ""),
                        "score": src.get("score", 0),
                    }
                    for src in sources
                ]

            logger.info(f"知识库检索完成: 查询='{query[:30]}...', 结果数={retrieved_count}")

            return ToolResult(
                success=True,
                data=search_result,
                metadata={
                    "query": query,
                    "top_k": top_k,
                    "retrieved_count": retrieved_count,
                }
            )

        except Exception as e:
            logger.error(f"知识库检索失败: {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=f"知识库检索失败: {str(e)}",
            )

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
