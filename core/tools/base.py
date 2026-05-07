"""
工具基类定义
所有工具继承此基类，统一接口规范
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """
    工具执行结果（统一返回格式）

    设计原则：
    - name: 工具名，便于日志和追踪
    - success: 执行是否成功
    - summary: 给 LLM 看的自然语言摘要（拼进 Prompt）
    - data: 给前端展示的结构化数据
    - error: 失败时的错误描述
    """
    name: str
    success: bool
    summary: str                                    # 给 LLM 的文本
    data: Dict[str, Any] = field(default_factory=dict)  # 给前端的结构化数据
    error: str = ""                                 # 失败时的错误描述

    def to_dict(self) -> Dict:
        """序列化为字典（兼容旧接口）"""
        return {
            "success": self.success,
            "data": self.data,
            "summary": self.summary,
            "error": self.error,
            "name": self.name,
        }

    def to_prompt_text(self) -> str:
        """
        拼进 LLM Prompt 时使用
        成功返回 summary，失败返回带标记的错误信息
        """
        if self.success:
            return self.summary
        return f"[工具执行失败] {self.name}: {self.error}"


class BaseTool(ABC):
    """
    工具抽象基类

    所有工具必须实现：
    - name: 工具名称（唯一标识）
    - description: 工具描述（LLM 用于判断何时调用）
    - parameters_schema: OpenAI tool calling 兼容的参数定义
    - execute(): 实际执行逻辑
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，用于 LLM 理解工具用途"""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict:
        """
        OpenAI tool calling 兼容的参数定义

        示例：
        {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "date": {"type": "string", "description": "日期，格式：YYYY-MM-DD"}
            },
            "required": ["city"]
        }
        """
        pass

    def get_tool_schema(self) -> Dict:
        """
        生成 OpenAI tool calling 格式的工具定义

        返回格式：
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询指定城市的天气信息",
                "parameters": {...}
            }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            }
        }

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        执行工具

        Args:
            **kwargs: 工具参数，由 LLM 生成

        Returns:
            ToolResult: 执行结果（不抛异常，错误通过 ToolResult.success=False 表达）
        """
        pass

    def safe_run(self, **kwargs) -> ToolResult:
        """
        Agent 统一调用入口，捕获未预期异常

        确保工具执行不会抛出异常，所有错误都通过 ToolResult 返回
        """
        try:
            return self.execute(**kwargs)
        except Exception as e:
            logger.error(f"工具 {self.name} 执行异常: {e}")
            return ToolResult(
                name=self.name,
                success=False,
                summary="",
                error=str(e),
            )

    def validate_parameters(self, kwargs: Dict) -> Optional[str]:
        """
        验证参数（可选实现）

        Returns:
            None 如果验证通过，否则返回错误信息
        """
        required = self.parameters_schema.get("required", [])
        for param in required:
            if param not in kwargs or kwargs[param] is None:
                return f"缺少必需参数: {param}"
        return None
