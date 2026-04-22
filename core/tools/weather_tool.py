"""
天气查询工具
使用模拟数据实现，避免依赖外部 API

支持功能：
- 查询城市天气
- 查询温度、降水、风力等信息
"""
from typing import Dict, Optional
import logging
from datetime import datetime, timedelta

from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


# 模拟天气数据（中国主要农业城市）
MOCK_WEATHER_DATA = {
    "北京": {
        "temperature": {"current": 18, "high": 22, "low": 12},
        "weather": "晴",
        "humidity": 45,
        "wind": "东北风 3级",
        "rainfall": 0,
        "forecast": [
            {"date": "今天", "weather": "晴", "high": 22, "low": 12},
            {"date": "明天", "weather": "多云", "high": 20, "low": 11},
            {"date": "后天", "weather": "小雨", "high": 16, "low": 10},
        ]
    },
    "成都": {
        "temperature": {"current": 20, "high": 24, "low": 16},
        "weather": "多云",
        "humidity": 70,
        "wind": "微风",
        "rainfall": 2.5,
        "forecast": [
            {"date": "今天", "weather": "多云", "high": 24, "low": 16},
            {"date": "明天", "weather": "小雨", "high": 22, "low": 15},
            {"date": "后天", "weather": "阴", "high": 21, "low": 14},
        ]
    },
    "哈尔滨": {
        "temperature": {"current": 8, "high": 14, "low": 2},
        "weather": "晴",
        "humidity": 55,
        "wind": "西北风 4级",
        "rainfall": 0,
        "forecast": [
            {"date": "今天", "weather": "晴", "high": 14, "low": 2},
            {"date": "明天", "weather": "晴", "high": 16, "low": 4},
            {"date": "后天", "weather": "多云", "high": 15, "low": 3},
        ]
    },
    "广州": {
        "temperature": {"current": 28, "high": 32, "low": 24},
        "weather": "多云",
        "humidity": 80,
        "wind": "南风 2级",
        "rainfall": 5.0,
        "forecast": [
            {"date": "今天", "weather": "多云", "high": 32, "low": 24},
            {"date": "明天", "weather": "雷阵雨", "high": 30, "low": 23},
            {"date": "后天", "weather": "阵雨", "high": 29, "low": 23},
        ]
    },
    "西安": {
        "temperature": {"current": 16, "high": 22, "low": 10},
        "weather": "晴",
        "humidity": 50,
        "wind": "东风 2级",
        "rainfall": 0,
        "forecast": [
            {"date": "今天", "weather": "晴", "high": 22, "low": 10},
            {"date": "明天", "weather": "晴", "high": 24, "low": 12},
            {"date": "后天", "weather": "多云", "high": 23, "low": 11},
        ]
    },
    "南京": {
        "temperature": {"current": 20, "high": 25, "low": 15},
        "weather": "多云",
        "humidity": 65,
        "wind": "东南风 3级",
        "rainfall": 0,
        "forecast": [
            {"date": "今天", "weather": "多云", "high": 25, "low": 15},
            {"date": "明天", "weather": "小雨", "high": 22, "low": 14},
            {"date": "后天", "weather": "阴", "high": 21, "low": 13},
        ]
    },
    "武汉": {
        "temperature": {"current": 22, "high": 27, "low": 17},
        "weather": "多云",
        "humidity": 72,
        "wind": "南风 2级",
        "rainfall": 1.5,
        "forecast": [
            {"date": "今天", "weather": "多云", "high": 27, "low": 17},
            {"date": "明天", "weather": "小雨", "high": 24, "low": 16},
            {"date": "后天", "weather": "中雨", "high": 21, "low": 15},
        ]
    },
    "昆明": {
        "temperature": {"current": 18, "high": 22, "low": 12},
        "weather": "晴",
        "humidity": 60,
        "wind": "西南风 2级",
        "rainfall": 0,
        "forecast": [
            {"date": "今天", "weather": "晴", "high": 22, "low": 12},
            {"date": "明天", "weather": "多云", "high": 21, "low": 11},
            {"date": "后天", "weather": "晴", "high": 23, "low": 13},
        ]
    },
    "乌鲁木齐": {
        "temperature": {"current": 12, "high": 18, "low": 5},
        "weather": "晴",
        "humidity": 35,
        "wind": "西北风 4级",
        "rainfall": 0,
        "forecast": [
            {"date": "今天", "weather": "晴", "high": 18, "low": 5},
            {"date": "明天", "weather": "晴", "high": 20, "low": 7},
            {"date": "后天", "weather": "多云", "high": 17, "low": 4},
        ]
    },
    "拉萨": {
        "temperature": {"current": 10, "high": 16, "low": 2},
        "weather": "晴",
        "humidity": 30,
        "wind": "西风 3级",
        "rainfall": 0,
        "forecast": [
            {"date": "今天", "weather": "晴", "high": 16, "low": 2},
            {"date": "明天", "weather": "晴", "high": 17, "low": 3},
            {"date": "后天", "weather": "多云", "high": 15, "low": 1},
        ]
    },
}


class WeatherTool(BaseTool):
    """天气查询工具"""

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return "查询指定城市的天气信息，包括温度、天气状况、湿度、风力、降水等。可用于农业气象条件分析。"

    @property
    def parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如：北京、成都、广州"
                },
                "date": {
                    "type": "string",
                    "description": "查询日期，可选。支持：今天、明天、后天，或具体日期（YYYY-MM-DD）。默认查询今天。"
                },
                "info_type": {
                    "type": "string",
                    "description": "查询信息类型，可选。支持：current（当前天气）、forecast（天气预报）。默认 current。",
                    "enum": ["current", "forecast"]
                }
            },
            "required": ["city"]
        }

    def execute(self, **kwargs) -> ToolResult:
        """
        执行天气查询

        Args:
            city: 城市名称
            date: 日期（可选）
            info_type: 信息类型（可选）
        """
        city = kwargs.get("city", "")
        date = kwargs.get("date", "今天")
        info_type = kwargs.get("info_type", "current")

        # 标准化城市名
        city_normalized = self._normalize_city_name(city)

        if city_normalized not in MOCK_WEATHER_DATA:
            available_cities = list(MOCK_WEATHER_DATA.keys())
            return ToolResult(
                success=False,
                data=None,
                error_message=f"暂不支持查询该城市的天气。支持的城市：{', '.join(available_cities)}",
            )

        weather_data = MOCK_WEATHER_DATA[city_normalized]

        try:
            if info_type == "forecast":
                result = self._get_forecast(weather_data, city_normalized)
            else:
                result = self._get_current_weather(weather_data, city_normalized, date)

            logger.info(f"天气查询成功: {city_normalized}, {info_type}")
            return ToolResult(
                success=True,
                data=result,
                metadata={"city": city_normalized, "date": date, "info_type": info_type}
            )

        except Exception as e:
            logger.error(f"天气查询失败: {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=f"天气查询失败: {str(e)}",
            )

    def _normalize_city_name(self, city: str) -> str:
        """标准化城市名称"""
        # 移除常见后缀
        city = city.replace("市", "").replace("省", "").strip()

        # 常见别名映射
        aliases = {
            "蓉城": "成都",
            "锦城": "成都",
            "羊城": "广州",
            "穗城": "广州",
            "冰城": "哈尔滨",
            "春城": "昆明",
            "金陵": "南京",
            "江城": "武汉",
            "蓉": "成都",
            "京": "北京",
        }

        return aliases.get(city, city)

    def _get_current_weather(self, data: Dict, city: str, date: str) -> Dict:
        """获取当前天气"""
        return {
            "city": city,
            "date": date,
            "weather": data["weather"],
            "temperature": {
                "current": data["temperature"]["current"],
                "high": data["temperature"]["high"],
                "low": data["temperature"]["low"],
                "unit": "摄氏度"
            },
            "humidity": f"{data['humidity']}%",
            "wind": data["wind"],
            "rainfall": f"{data['rainfall']}mm",
            "agriculture_tip": self._get_agri_tip(data)
        }

    def _get_forecast(self, data: Dict, city: str) -> Dict:
        """获取天气预报"""
        return {
            "city": city,
            "forecast": data["forecast"],
            "agriculture_tip": "建议关注未来几天天气变化，合理安排农事活动。"
        }

    def _get_agri_tip(self, data: Dict) -> str:
        """根据天气生成农业建议"""
        temp = data["temperature"]["current"]
        weather = data["weather"]
        rainfall = data["rainfall"]

        tips = []

        if temp < 10:
            tips.append("气温较低，注意作物防寒保暖")
        elif temp > 30:
            tips.append("气温较高，注意灌溉和遮阳")

        if "雨" in weather or rainfall > 0:
            tips.append("有降水，注意田间排水")

        if "晴" in weather:
            tips.append("天气晴好，适合田间作业")

        return "；".join(tips) if tips else "天气条件适中，正常开展农事活动即可。"
