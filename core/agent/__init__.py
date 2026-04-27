"""
Agent 模块
实现基于 Tool Calling 的农业 Agent

提供两种实现：
1. AgricultureAgent - 原生实现（手写 ReAct 循环）
2. LangGraphAgent - LangGraph 实现（推荐）
"""
from core.agent.agent import AgricultureAgent, AgentContext
from core.agent.tool_registry import ToolRegistry
from core.agent.langgraph_agent import LangGraphAgent

# 默认使用 LangGraph Agent
__all__ = [
    "AgricultureAgent",  # 原生实现
    "LangGraphAgent",    # LangGraph 实现（推荐）
    "AgentContext",
    "ToolRegistry",
]