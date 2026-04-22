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
    """工具执行结果"""
    success: bool
    data: Any
    error_message: Optional[str] = None
    metadata: Optional[Dict] = field(default_factory=dict)  # 额外信息（如 API 耗时）

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


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
            ToolResult: 执行结果
        """
        pass

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
