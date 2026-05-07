"""
工具注册中心
管理所有工具实例，生成 OpenAI tool schema
支持自动发现和注册工具
"""
from typing import Dict, List, Optional
import pkgutil
import importlib
import inspect
import logging

from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    工具注册中心

    功能：
    - 注册和管理工具实例
    - 自动发现 tools_package 包下的所有工具类
    - 生成 OpenAI tool calling schema
    - 根据 tool_name 获取工具
    """

    def __init__(self):
        """
        初始化工具注册中心
        """
        self._tools: Dict[str, BaseTool] = {}

    @classmethod
    def discover(cls, tools_package) -> "ToolRegistry":
        """
        自动发现并注册工具

        自动扫描 tools_package 包下所有模块，
        找到继承 BaseTool 的非抽象类并实例化注册

        Args:
            tools_package: 工具模块包（如 core.tools）

        Returns:
            ToolRegistry: 包含所有自动发现工具的注册中心
        """
        registry = cls()

        logger.info(f"开始自动发现工具，扫描路径: {tools_package.__path__}")

        for _, module_name, _ in pkgutil.walk_packages(
            tools_package.__path__,
            prefix=tools_package.__name__ + "."
        ):
            # 跳过 base.py（只定义基类）
            if module_name.endswith(".base"):
                continue

            try:
                module = importlib.import_module(module_name)
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    # 检查是否是 BaseTool 的子类（但不是 BaseTool 本身，也不是抽象类）
                    if (
                        issubclass(obj, BaseTool)
                        and obj is not BaseTool
                        and not inspect.isabstract(obj)
                    ):
                        # 确保类是在该模块中定义的（不是从其他模块导入的）
                        if obj.__module__ == module_name:
                            instance = obj()
                            registry.register(instance)
                            logger.info(f"自动发现工具: {instance.name} ({module_name})")
            except Exception as e:
                logger.warning(f"扫描模块 {module_name} 时出错: {e}")

        logger.info(f"工具自动发现完成，共发现 {len(registry._tools)} 个工具")
        return registry

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

    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        执行指定工具

        Args:
            tool_name: 工具名称
            **kwargs: 工具参数

        Returns:
            ToolResult: 工具执行结果
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            logger.error(f"工具不存在: {tool_name}")
            return ToolResult(
                name=tool_name,
                success=False,
                summary="",
                error=f"工具 {tool_name} 不存在",
            )

        # 验证参数
        error = tool.validate_parameters(kwargs)
        if error:
            logger.warning(f"工具参数验证失败: {error}")
            return ToolResult(
                name=tool_name,
                success=False,
                summary="",
                error=error,
            )

        # 使用 safe_run 执行工具（捕获异常）
        result = tool.safe_run(**kwargs)
        logger.info(f"工具 {tool_name} 执行完成: success={result.success}")

        return result

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools