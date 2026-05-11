"""
虚谷数据库查询工具
查询作物产量数据表 AGME_SUB_AGI_YIELD_TYPE_TAB
"""
from typing import Dict, Optional, List
import logging
from dataclasses import dataclass

from common.XuguConnectionManager import XuguConnectionManager
from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class crop_yeild_tool(BaseTool):
    """虚谷数据库查询工具 - 查询作物产量数据"""

    def __init__(self, connection_config: Optional[Dict] = None):
        """
        初始化数据库连接配置

        Args:
            connection_config: 数据库连接配置，包含 host, port, database, user, password 等
        """
        self.connection_config = connection_config or {}
        self._connection = None

    @property
    def name(self) -> str:
        return "query_yield_development"

    @property
    def description(self) -> str:
        return (
            "使用模糊查询规范的作物产量数据，如提问产量，则优先使用该工具。"
            "可按作物名称等条件筛选，返回启用的产量表格数据。"
            "如果作物信息无法精准匹配，如查询水稻，此为不规范作物名称，应为一季稻，可以传入稻。"
        )

    @property
    def parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "crop_name": {
                    "type": "string",
                    "description": "作物名称，如：冬小麦、一季稻、春玉米。可选参数，不传则查询所有作物。"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制，默认20条，最多100条",
                    "default": 20
                }
            },
            "required": []
        }

    def execute(self, **kwargs) -> ToolResult:
        """
        执行查询

        Args:
            crop_name: 作物名称（可选筛选条件）
            limit: 返回结果数量限制
        """
        crop_name = kwargs.get("crop_name")
        limit = min(kwargs.get("limit", 20), 100)

        try:
            # 获取数据库连接
            connection = self._get_connection()

            # 构建 SQL 查询
            sql = self._build_query(crop_name, limit)

            # 执行查询
            results = self._execute_query(connection, sql, crop_name)

            # 生成摘要
            summary = self._generate_summary(results, crop_name)

            logger.info(f"虚谷数据库查询成功, 返回 {len(results)} 条记录")

            return ToolResult(
                name=self.name,
                success=True,
                summary=summary,
                data={
                    "total": len(results),
                    "records": results
                }
            )

        except Exception as e:
            logger.error(f"虚谷数据库查询失败: {e}")
            return ToolResult(
                name=self.name,
                success=False,
                summary="",
                error=f"数据库查询失败: {str(e)}"
            )

    def _get_connection(self):
        """
        获取数据库连接
        """
        try:
            self._connection = XuguConnectionManager()
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")


    def _build_query(self, crop_name: Optional[str], limit: int) -> str:
        """
        构建 SQL 查询语句

        Args:
            crop_name: 作物名称筛选条件
            limit: 结果数量限制

        Returns:
            SQL 查询字符串
        """
        base_query = """
            SELECT
            CROP_NAME,
            CROP_CODE,
            V_STUC_NAME,
            V_STUC_UNIT
            FROM AGME_SUB_AGI_YIELD_TYPE_TAB
            WHERE IS_USED = 1
        """

        conditions = []

        if crop_name:
            conditions.append(f"CROP_NAME LIKE '%{crop_name}%'")

        if conditions:
            base_query += " AND " + " AND ".join(conditions)

        base_query += f" LIMIT {limit}"

        print(base_query)
        return base_query

    def _execute_query(self, connection, sql: str, crop_name: Optional[str]) -> List[Dict]:
        """
        执行 SQL 查询并返回结果

        Args:
            connection: 数据库连接
            sql: SQL 查询语句
            crop_name: 作物名称参数

        Returns:
            查询结果列表
        """
        data_list = self._connection.execute(sql)
        return [
            {
                "crop_name": row[0],
                "crop_code": row[1],
                "struc_name": row[2],
                "struc_unit": row[3]
            }
            for row in data_list
        ]

    def _generate_summary(self, results: List[Dict], crop_name: Optional[str]) -> str:
        """生成给 LLM 的摘要"""
        if not results:
            if crop_name:
                return f"未找到作物「{crop_name}」的产量数据。"
            return "未找到任何产量数据。"

        crop_filter = f"作物「{crop_name}」的" if crop_name else ""
        summary_parts = [f"查询到 {len(results)} 条{crop_filter}产量数据："]

        for i, record in enumerate(results[:5]):  # 最多展示5条
            crop = record.get("crop_name", "未知")
            dev_stage = record.get("dev_name", "未知阶段")
            summary_parts.append(f"  {i+1}. {crop} - {dev_stage}")

        if len(results) > 5:
            summary_parts.append(f"  ... 还有 {len(results) - 5} 条记录")

        return "\n".join(summary_parts)

    def close(self):
        """关闭数据库连接"""
        if self._connection:
            try:
                self._connection.close()
            except Exception as e:
                logger.warning(f"关闭数据库连接失败: {e}")
            finally:
                self._connection = None

if __name__ == "__main__":
    tool = crop_yeild_tool()
    result = tool.execute(crop_name="稻")
    print(result)