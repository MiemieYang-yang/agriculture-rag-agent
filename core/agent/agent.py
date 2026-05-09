"""
Agent 核心实现
基于 Qwen Tool Calling 的 ReAct 循环

流程：
1. 接收用户问题
2. LLM 决定是否需要调用工具
3. 如需要，执行工具调用，将结果喂回 LLM
4. 重复 2-3 直到 LLM 给出最终回答
5. 返回答案 + 工具调用记录
"""
import json
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, field
import logging

from core.config import cfg
from core.llm_client import QwenClient
from core.rag_pipeline import RAGPipeline
from core.agent.tool_registry import ToolRegistry
from core.agent.prompts import AGENT_SYSTEM_PROMPT
from core.tools.base import ToolResult

# 导入工具包用于自动发现
import core.tools as tools_package

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """工具调用记录"""
    id: str
    name: str
    arguments: Dict
    result: Dict
    success: bool

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "success": self.success,
        }


@dataclass
class AgentResult:
    """Agent 执行结果"""
    answer: str = ""  # 提供默认值
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    iterations: int = 0

    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "sources": self.sources,
            "iterations": self.iterations,
        }


class AgricultureAgent:
    """
    农业 Agent

    基于 Qwen Tool Calling 实现 ReAct 循环
    """

    def __init__(
            self,
            rag_pipeline: Optional[RAGPipeline] = None,
            llm_client: Optional[QwenClient] = None,
            tool_registry: Optional[ToolRegistry] = None,
            max_iterations: int = cfg.AGENT_MAX_ITERATIONS,
    ):
        """
        初始化 Agent

        Args:
            rag_pipeline: RAG Pipeline 实例（用于知识库检索工具）
            llm_client: LLM 客户端
            tool_registry: 工具注册中心（如不传入则自动发现）
            max_iterations: 最大迭代次数（防止无限循环）
        """
        self.llm_client = llm_client or QwenClient()
        self.rag_pipeline = rag_pipeline

        # 使用自动发现机制注册工具
        if tool_registry is None:
            self.tool_registry = ToolRegistry.discover(tools_package)
            logger.info("使用工具自动发现机制")
        else:
            self.tool_registry = tool_registry

        self.max_iterations = max_iterations

        # 如果提供了 RAG Pipeline，更新知识库检索工具
        if rag_pipeline:
            from core.tools.knowledge_search import KnowledgeSearchTool
            self.tool_registry.register(KnowledgeSearchTool(rag_pipeline))

        logger.info(
            f"Agent 初始化完成，可用工具: {self.tool_registry.list_tool_names()}, "
            f"最大迭代: {max_iterations}"
        )

    def process(
            self,
            question: str,
            history: Optional[List[Dict]] = None,
    ) -> AgentResult:
        """
        Agent 主处理流程

        Args:
            question: 用户问题
            history: 对话历史

        Returns:
            AgentResult: 包含答案、工具调用记录等
        """
        logger.info(f"Agent 开始处理问题: {question[:50]}...")

        # 获取工具 schema
        tools = self.tool_registry.get_tool_schemas()
        if not tools:
            logger.warning("没有可用工具，降级为普通 RAG")
            return self._fallback_rag(question, history)

        # 初始化结果
        result = AgentResult()
        messages = history or []

        # ReAct 循环
        current_query = question
        for iteration in range(self.max_iterations):
            result.iterations = iteration + 1
            logger.debug(f"Agent 迭代 {iteration + 1}/{self.max_iterations}")

            try:
                # 调用 LLM
                response = self.llm_client.chat_with_tools(
                    user_message=current_query,
                    tools=tools,
                    system_prompt=AGENT_SYSTEM_PROMPT,
                    history=messages,
                )

                finish_reason = response.get("finish_reason", "stop")
                tool_calls = response.get("tool_calls", [])
                content = response.get("content", "")

                # 如果 LLM 直接返回回答，结束循环
                if finish_reason == "stop" and not tool_calls:
                    result.answer = content
                    logger.info(f"Agent 完成，迭代次数: {result.iterations}")
                    break

                # 如果有工具调用，执行工具
                if tool_calls:
                    # 先添加助手的工具调用消息到历史
                    assistant_message = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                }
                            }
                            for tc in tool_calls
                        ]
                    }
                    messages.append(assistant_message)

                    # 执行每个工具调用
                    tool_results = []
                    for tc in tool_calls:
                        tool_name = tc["name"]
                        tool_id = tc["id"]

                        # 解析参数
                        try:
                            arguments = json.loads(tc["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}

                        # 执行工具（返回 ToolResult）
                        tool_result = self.tool_registry.execute_tool(tool_name, **arguments)

                        # 记录工具调用
                        record = ToolCallRecord(
                            id=tool_id,
                            name=tool_name,
                            arguments=arguments,
                            result=tool_result.to_dict(),
                            success=tool_result.success,
                        )
                        result.tool_calls.append(record)

                        # 收集结果
                        tool_results.append({
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "result": tool_result.to_prompt_text(),  # 给 LLM 的摘要
                        })

                        # 如果是知识库检索，收集来源
                        if tool_name == "knowledge_search" and tool_result.success:
                            sources = tool_result.data.get("sources", [])
                            result.sources.extend(sources)

                    # 提交工具结果给 LLM
                    current_query = ""  # 后续迭代不需要用户消息
                    response = self.llm_client.submit_tool_results(
                        tool_call_results=tool_results,
                        system_prompt=AGENT_SYSTEM_PROMPT,
                        history=messages,
                        tools=tools,
                    )

                    # 检查响应
                    finish_reason = response.get("finish_reason", "stop")
                    new_tool_calls = response.get("tool_calls", [])
                    content = response.get("content", "")

                    if finish_reason == "stop" and not new_tool_calls:
                        result.answer = content
                        logger.info(f"Agent 完成，迭代次数: {result.iterations}")
                        break
                    elif new_tool_calls:
                        # LLM 请求更多工具调用，继续循环
                        tool_calls = new_tool_calls
                        continue
                    else:
                        # 其他情况，用已有内容作为答案
                        result.answer = content or "抱歉，我无法处理您的请求。"
                        break

                else:
                    # 没有工具调用，使用内容作为答案
                    result.answer = content
                    break

            except Exception as e:
                logger.error(f"Agent 迭代异常: {e}")
                result.answer = f"处理过程中出现错误: {str(e)}"
                break

        # 达到最大迭代次数
        if not result.answer:
            result.answer = "抱歉，我无法在有限的步骤内完成您的请求，请尝试简化问题。"

        return result

    def _fallback_rag(
            self,
            question: str,
            history: Optional[List[Dict]],
    ) -> AgentResult:
        """降级到普通 RAG 模式"""
        logger.info("Agent 降级为普通 RAG 模式")

        if self.rag_pipeline is None:
            self.rag_pipeline = RAGPipeline()

        rag_result = self.rag_pipeline.query(question, history=history)

        return AgentResult(
            answer=rag_result.get("answer", ""),
            sources=rag_result.get("sources", []),
            iterations=1,
        )

    def stream_process(
            self,
            question: str,
            history: Optional[List[Dict]] = None,
    ) -> Generator[str, None, None]:
        """
        流式处理

        流程：
        1. 同步执行工具调用（无法流式）
        2. 最终回答使用流式输出

        Args:
            question: 用户问题
            history: 对话历史

        Yields:
            流式输出的文本片段
        """
        logger.info(f"Agent 开始流式处理问题: {question[:50]}...")

        # 获取工具 schema
        tools = self.tool_registry.get_tool_schemas()
        if not tools:
            logger.warning("没有可用工具，降级为普通 RAG 流式")
            if self.rag_pipeline is None:
                self.rag_pipeline = RAGPipeline()
            # RAG 流式
            for chunk in self.rag_pipeline.query_stream(question, history=history):
                yield chunk
            return

        messages = history or []

        # ReAct 循环（同步执行工具）
        current_query = question
        final_stream_ready = False
        tool_call_results = []

        for iteration in range(self.max_iterations):
            logger.debug(f"Agent 流式迭代 {iteration + 1}/{self.max_iterations}")

            try:
                # 调用 LLM（同步）
                response = self.llm_client.chat_with_tools(
                    user_message=current_query,
                    tools=tools,
                    system_prompt=AGENT_SYSTEM_PROMPT,
                    history=messages,
                )

                finish_reason = response.get("finish_reason", "stop")
                tool_calls = response.get("tool_calls", [])
                content = response.get("content", "")

                # 如果 LLM 直接返回回答，流式输出
                if finish_reason == "stop" and not tool_calls:
                    yield content
                    return

                # 如果有工具调用，执行工具
                if tool_calls:
                    # 添加助手的工具调用消息到历史
                    assistant_message = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                }
                            }
                            for tc in tool_calls
                        ]
                    }
                    messages.append(assistant_message)

                    # 执行每个工具调用
                    for tc in tool_calls:
                        tool_name = tc["name"]
                        tool_id = tc["id"]

                        try:
                            arguments = json.loads(tc["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}

                        # 执行工具
                        tool_result = self.tool_registry.execute_tool(tool_name, **arguments)

                        # 收集结果
                        tool_call_results.append({
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "result": tool_result.to_prompt_text(),
                        })

                    # 提交工具结果，流式输出最终回答
                    current_query = ""
                    for chunk in self.llm_client.submit_tool_results_stream(
                        tool_call_results=tool_call_results,
                        system_prompt=AGENT_SYSTEM_PROMPT,
                        history=messages,
                    ):
                        yield chunk
                    return

                else:
                    # 没有工具调用
                    yield content
                    return

            except Exception as e:
                logger.error(f"Agent 流式迭代异常: {e}")
                yield f"处理过程中出现错误: {str(e)}"
                return

        # 达到最大迭代次数
        yield "抱歉，我无法在有限的步骤内完成您的请求，请尝试简化问题。"


if __name__ == '__main__':
    agent = AgricultureAgent()
    agent.process("四川盆地水稻播种期")
