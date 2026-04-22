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
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
import re

from core.config import cfg
from core.llm_client import QwenClient
from core.rag_pipeline import RAGPipeline
from core.agent.tool_registry import ToolRegistry
from core.agent.prompts import AGENT_SYSTEM_PROMPT, build_agent_prompt

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """
    Agent 多轮对话上下文

    用于处理追问场景，如：
    - 用户: "四川盆地水稻最佳播种期？" → 助手回答
    - 用户: "那玉米呢？" → 需要知道用户问的是"四川盆地玉米最佳播种期"
    """
    last_crop: Optional[str] = None      # 最近提到的作物
    last_location: Optional[str] = None  # 最近提到的地点
    last_date: Optional[str] = None      # 最近提到的时间
    last_topic: Optional[str] = None     # 最近的话题类型

    def update_from_query(self, query: str):
        """从用户查询中提取并更新实体"""
        # 提取作物名
        crops = ["水稻", "小麦", "玉米", "大豆", "棉花", "油菜", "马铃薯", "甘薯",
                 "花生", "甘蔗", "柑橘", "苹果", "茶叶"]
        for crop in crops:
            if crop in query:
                self.last_crop = crop
                break

        # 提取地点（简单规则）
        locations = ["北京", "成都", "哈尔滨", "广州", "西安", "南京", "武汉",
                     "昆明", "乌鲁木齐", "拉萨", "四川", "东北", "华北", "华南",
                     "西南", "西北", "华东", "华中"]
        for loc in locations:
            if loc in query:
                self.last_location = loc
                break

        # 提取时间
        time_patterns = ["今天", "明天", "后天", "昨天", "本周", "下周", "本月", "下月"]
        for t in time_patterns:
            if t in query:
                self.last_date = t
                break

    def is_follow_up(self, query: str) -> bool:
        """判断是否为追问"""
        follow_up_patterns = [
            r"那(\S+)呢",
            r"那呢",
            r"那\b",
            r"如果.*呢",
            r"另外",
            r"还有",
        ]
        for pattern in follow_up_patterns:
            if re.search(pattern, query):
                return True
        return False

    def enrich_query(self, query: str) -> str:
        """
        补全追问中的缺失信息

        例如：用户说"那玉米呢"，补全为"那[四川盆地]玉米[最佳播种期]呢"
        """
        if not self.is_follow_up(query):
            return query

        # 检测追问中缺失的信息
        enriched_parts = []

        # 如果追问中未提及作物但历史中有
        if self.last_crop and self.last_crop not in query:
            enriched_parts.append(f"（参考作物：{self.last_crop}）")

        # 如果追问中未提及地点但历史中有
        if self.last_location and self.last_location not in query:
            enriched_parts.append(f"（参考地点：{self.last_location}）")

        if enriched_parts:
            return query + " " + " ".join(enriched_parts)
        return query


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
    answer: str
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    iterations: int = 0
    context: Optional[AgentContext] = None

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
            tool_registry: 工具注册中心
            max_iterations: 最大迭代次数（防止无限循环）
        """
        self.llm_client = llm_client or QwenClient()
        self.rag_pipeline = rag_pipeline
        self.tool_registry = tool_registry or ToolRegistry()
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
            context: Optional[AgentContext] = None,
    ) -> AgentResult:
        """
        Agent 主处理流程

        Args:
            question: 用户问题
            history: 对话历史
            context: 多轮对话上下文

        Returns:
            AgentResult: 包含答案、工具调用记录等
        """
        logger.info(f"Agent 开始处理问题: {question[:50]}...")

        # 初始化上下文
        if context is None:
            context = AgentContext()

        # 更新上下文实体
        context.update_from_query(question)

        # 检测是否为追问
        is_follow_up = context.is_follow_up(question)
        if is_follow_up:
            question = context.enrich_query(question)
            logger.info(f"检测到追问，补全后问题: {question}")

        # 获取工具 schema
        tools = self.tool_registry.get_tool_schemas()
        if not tools:
            logger.warning("没有可用工具，降级为普通 RAG")
            return self._fallback_rag(question, history, context)

        # 初始化结果
        result = AgentResult(context=context)
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
                    system_prompt=build_agent_prompt({
                        "is_follow_up": is_follow_up,
                        "last_crop": context.last_crop,
                        "last_location": context.last_location,
                        "last_date": context.last_date,
                    }),
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

                        # 执行工具
                        tool_result = self.tool_registry.execute_tool(tool_name, **arguments)

                        # 记录工具调用
                        record = ToolCallRecord(
                            id=tool_id,
                            name=tool_name,
                            arguments=arguments,
                            result=tool_result,
                            success=tool_result.get("success", False),
                        )
                        result.tool_calls.append(record)

                        # 收集结果
                        tool_results.append({
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "result": json.dumps(tool_result, ensure_ascii=False),
                        })

                        # 如果是知识库检索，收集来源
                        if tool_name == "knowledge_search" and tool_result.get("success"):
                            sources = tool_result.get("data", {}).get("sources", [])
                            result.sources.extend(sources)

                    # 提交工具结果给 LLM
                    current_query = ""  # 后续迭代不需要用户消息
                    response = self.llm_client.submit_tool_results(
                        tool_call_results=tool_results,
                        system_prompt=build_agent_prompt({
                            "is_follow_up": False,
                            "last_crop": context.last_crop,
                            "last_location": context.last_location,
                            "last_date": context.last_date,
                        }),
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
            context: AgentContext,
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
            context=context,
        )

    def stream_process(
            self,
            question: str,
            history: Optional[List[Dict]] = None,
            context: Optional[AgentContext] = None,
    ):
        """
        流式处理（Phase 4 实现）

        目前降级为同步处理后一次性返回
        """
        result = self.process(question, history, context)
        yield result.answer
