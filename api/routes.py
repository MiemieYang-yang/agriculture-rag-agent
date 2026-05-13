"""
FastAPI 接口层
提供 RESTful API，方便集成测试和后续对接前端

接口列表：
- POST /api/query      → 单次 RAG 问答
- POST /api/chat       → 多轮对话
- POST /api/chat/stream → 多轮对话（流式）
- POST /api/agent/query → Agent 智能问答（支持工具调用）
- GET  /api/stats      → 知识库统计
- GET  /health         → 健康检查
"""
from typing import List, Optional, Dict
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.container import container
from core.config import cfg

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


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


# ── Agent 接口数据模型（Phase 3）─────────────────────────────────────────────

class ToolCallInfo(BaseModel):
    id: str
    name: str
    arguments: Dict
    result: Dict
    success: bool


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, description="用户问题")
    history: List[dict] = Field(default=[], description="对话历史")
    enable_tools: bool = Field(default=True, description="是否启用工具调用")


class AgentQueryResponse(BaseModel):
    answer: str
    tool_calls: List[ToolCallInfo] = Field(default=[], description="工具调用记录")
    sources: List[SourceInfo] = Field(default=[], description="引用来源")
    iterations: int = Field(default=0, description="迭代次数")


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
        pipeline = container.rag_pipeline

        async def event_generator():
            for token in pipeline.query_stream(req.question):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        pipeline = container.rag_pipeline
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
        pipeline = container.rag_pipeline
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


@router.post("/chat/stream", summary="多轮对话（流式）")
async def chat_stream(req: ChatRequest):
    """
    支持流式输出的多轮对话接口
    
    使用 Server-Sent Events (SSE) 格式：
    - event: sources - 返回引用来源（可选）
    - event: chunk - 返回文本片段
    - event: done - 对话完成，包含完整答案和历史
    - event: error - 错误信息
    
    前端使用 EventSource 或 fetch + ReadableStream 接收
    """
    async def generate():
        try:
            pipeline = container.rag_pipeline
            
            # 先执行检索（同步）
            retrieved = pipeline._retrieve(req.question)
            
            # 如果有检索结果，发送来源信息
            if retrieved:
                sources = pipeline._format_sources(retrieved)
                yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"
            
            # 构建 prompt
            prompt = pipeline._build_prompt(req.question, retrieved) if retrieved else req.question
            
            system = cfg.SYSTEM_PROMPT if retrieved else (
                cfg.SYSTEM_PROMPT + "\n注意：知识库无相关资料，请基于专业知识作答。"
            )
            
            # 流式生成回答
            answer_chunks = []
            for token in pipeline.llm_client.chat_stream(
                prompt, 
                system_prompt=system, 
                history=req.history
            ):
                answer_chunks.append(token)
                # 发送文本片段
                yield f"event: chunk\ndata: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            
            # 组装完整答案
            full_answer = "".join(answer_chunks)
            
            # 更新对话历史
            updated_history = req.history + [
                {"role": "user", "content": req.question},
                {"role": "assistant", "content": full_answer},
            ]
            
            # 发送完成信号
            yield f"event: done\ndata: {json.dumps({'answer': full_answer, 'history': updated_history, 'retrieved_count': len(retrieved) }, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"流式对话失败: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        }
    )


@router.get("/stats", summary="知识库统计")
async def stats():
    """查看当前知识库状态"""
    try:
        pipeline = container.rag_pipeline
        return {
            "total_documents": pipeline.vector_store.count(),
            "collection": pipeline.vector_store.collection_name,
            "model": pipeline.llm_client.model,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Agent 接口（Phase 3）─────────────────────────────────────────────────────

@router.post("/agent/query", response_model=AgentQueryResponse, summary="Agent 智能问答")
async def agent_query(req: AgentQueryRequest):
    """
    Agent 智能问答接口

    特点：
    - 支持工具调用（天气查询、农学计算、知识库检索）
    - 支持多轮追问
    - 自动判断是否需要调用工具
    - 基于 LangGraph 实现 ReAct 循环

    流程：
    1. 接收问题
    2. Agent 判断是否需要调用工具
    3. 如需要，执行工具并将结果整合
    4. 返回最终答案 + 工具调用记录
    """
    try:
        agent = container.agent

        # 执行 Agent
        result = agent.process(
            question=req.question,
            history=req.history if req.history else None,
        )

        # 构建响应（result 是 AgentResult dataclass）
        tool_calls = [
            ToolCallInfo(
                id=tc.id,
                name=tc.name,
                arguments=tc.arguments,
                result=tc.result,
                success=tc.success,
            )
            for tc in result.tool_calls
        ]

        sources = [
            SourceInfo(
                filename=src.get("filename", ""),
                page=src.get("page"),
                score=src.get("score", 0),
                snippet=src.get("snippet", ""),
            )
            for src in result.sources
        ]

        return AgentQueryResponse(
            answer=result.answer,
            tool_calls=tool_calls,
            sources=sources,
            iterations=result.iterations,
        )

    except Exception as e:
        logger.error(f"Agent 查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent/tools", summary="获取可用工具列表")
async def list_tools():
    """列出 Agent 可用的所有工具"""
    try:
        agent = container.agent
        tools = agent.tool_registry.list_tools()
        return {
            "tools": tools,
            "framework": "ReAct (Qwen Tool Calling)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))