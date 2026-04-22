"""
工具注册中心
管理所有工具实例，生成 OpenAI tool schema
"""
from typing import Dict, List, Optional, Type
import logging

from core.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    工具注册中心

    功能：
    - 注册和管理工具实例
    - 生成 OpenAI tool calling schema
    - 根据 tool_name 获取工具
    """

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        """
        初始化工具注册中心

        Args:
            tools: 工具列表，默认注册所有内置工具
        """
        self._tools: Dict[str, BaseTool] = {}

        if tools:
            for tool in tools:
                self.register(tool)
        else:
            # 默认注册所有内置工具
            self._register_default_tools()

    def _register_default_tools(self):
        """注册默认工具"""
        try:
            from core.tools.weather_tool import WeatherTool
            from core.tools.agri_calculator import AgriCalculatorTool
            from core.tools.knowledge_search import KnowledgeSearchTool

            self.register(WeatherTool())
            self.register(AgriCalculatorTool())
            # KnowledgeSearchTool 需要延迟初始化（依赖 RAG Pipeline）
            self.register(KnowledgeSearchTool())

            logger.info(f"已注册 {len(self._tools)} 个默认工具")
        except Exception as e:
            logger.warning(f"注册默认工具失败: {e}")

    def register(self, tool: BaseTool):
        """
        注册工具

        Args:
            tool: 工具实例
        """
        if tool.name in self._tools:
            logger.warning(f"工具 {tool.name} 已存在，将被覆盖")
        self._tools[tool.name] = tool
        logger.debug(f"已注册工具: {tool.name}")

    def unregister(self, tool_name: str):
        """注销工具"""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.debug(f"已注销工具: {tool_name}")

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        根据名称获取工具

        Args:
            tool_name: 工具名称

        Returns:
            工具实例，如果不存在返回 None
        """
        return self._tools.get(tool_name)

    def get_all_tools(self) -> List[BaseTool]:
        """获取所有工具实例"""
        return list(self._tools.values())

    def get_tool_schemas(self) -> List[Dict]:
        """
        生成 OpenAI tool calling 格式的工具定义列表

        Returns:
            工具 schema 列表，可直接传给 LLM
        """
        return [tool.get_tool_schema() for tool in self._tools.values()]

    def has_tool(self, tool_name: str) -> bool:
        """检查工具是否存在"""
        return tool_name in self._tools

    def list_tool_names(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())

    def execute_tool(self, tool_name: str, **kwargs) -> Dict:
        """
        执行指定工具

        Args:
            tool_name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果（Dict 格式）
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            logger.error(f"工具不存在: {tool_name}")
            return {
                "success": False,
                "error": f"工具 {tool_name} 不存在",
            }

        try:
            # 验证参数
            error = tool.validate_parameters(kwargs)
            if error:
                logger.warning(f"工具参数验证失败: {error}")
                return {
                    "success": False,
                    "error": error,
                }

            # 执行工具
            result = tool.execute(**kwargs)
            logger.info(f"工具 {tool_name} 执行完成: success={result.success}")

            return result.to_dict()

        except Exception as e:
            logger.error(f"工具 {tool_name} 执行异常: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools