"""
FastAPI 接口层
提供 RESTful API，方便集成测试和后续对接前端

接口列表：
- POST /api/query      → 单次 RAG 问答
- POST /api/chat       → 多轮对话
- GET  /api/stats      → 知识库统计
- GET  /health         → 健康检查
"""
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.rag_pipeline import RAGPipeline

import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# RAG Pipeline 单例，避免每次请求重新初始化模型
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ── 数据模型 ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索文档数量")
    stream: bool = Field(default=False, description="是否流式返回")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    history: List[dict] = Field(default=[], description="对话历史")


class SourceInfo(BaseModel):
    filename: str
    page: Optional[int]
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    question: str
    retrieved_count: int


# ── 接口实现 ─────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, summary="单次 RAG 问答")
async def query(req: QueryRequest):
    """
    RAG 问答接口

    流程：
    1. 接收问题
    2. 语义检索相关文档
    3. 构建 Prompt 调用 LLM
    4. 返回回答 + 来源引用
    """
    if req.stream:
        # 流式响应
        pipeline = get_pipeline()

        async def event_generator():
            for token in pipeline.query_stream(req.question):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        pipeline = get_pipeline()
        result = pipeline.query(req.question)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", summary="多轮对话")
async def chat(req: ChatRequest):
    """
    支持多轮对话的问答接口
    前端需要在每次请求中携带完整对话历史
    """
    try:
        pipeline = get_pipeline()
        result = pipeline.query(req.question, history=req.history)

        # 更新对话历史
        updated_history = req.history + [
            {"role": "user", "content": req.question},
            {"role": "assistant", "content": result["answer"]},
        ]
        result["history"] = updated_history
        return result
    except Exception as e:
        logger.error(f"对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="知识库统计")
async def stats():
    """查看当前知识库状态"""
    try:
        pipeline = get_pipeline()
        return {
            "total_documents": pipeline.vector_store.count(),
            "collection": pipeline.vector_store.collection_name,
            "model": pipeline.llm_client.model,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))