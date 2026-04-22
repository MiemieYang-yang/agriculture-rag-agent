"""
农学指标计算工具

支持功能：
- 积温计算（GDD, Growing Degree Days）
- 降水统计
- 作物发育期推算
"""
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass

from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


# 作物基准温度（base temperature）
CROP_BASE_TEMPS = {
    "水稻": 10,
    "小麦": 0,
    "玉米": 10,
    "大豆": 10,
    "棉花": 15,
    "油菜": 5,
    "马铃薯": 7,
    "甘薯": 10,
    "花生": 12,
    "甘蔗": 20,
    "柑橘": 12,
    "苹果": 10,
    "茶叶": 10,
}

# 作物所需有效积温参考值
CROP_REQUIRED_GDD = {
    "水稻早稻": (1800, 2200),
    "水稻中稻": (2400, 2800),
    "水稻晚稻": (3000, 3500),
    "春小麦": (1500, 1800),
    "冬小麦": (1800, 2200),
    "春玉米": (2200, 2600),
    "夏玉米": (1800, 2200),
    "大豆": (2200, 2800),
    "棉花": (3200, 3800),
    "油菜": (1400, 1800),
}


@dataclass
class WeatherData:
    """天气数据结构"""
    date: str
    high_temp: float  # 最高温度
    low_temp: float   # 最低温度
    rainfall: float = 0.0  # 降水量(mm)


class AgriCalculatorTool(BaseTool):
    """农学指标计算工具"""

    @property
    def name(self) -> str:
        return "agri_calculator"

    @property
    def description(self) -> str:
        return "计算农业气象指标，包括积温(GDD)、降水量统计、作物发育期推算等。支持根据温度数据计算有效积温，评估作物生长条件。"

    @property
    def parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "calculation_type": {
                    "type": "string",
                    "description": "计算类型",
                    "enum": ["gdd", "rainfall_sum", "rainfall_avg", "growing_period"]
                },
                "crop_name": {
                    "type": "string",
                    "description": "作物名称，如：水稻、小麦、玉米。用于积温计算时确定基准温度。"
                },
                "temperature_data": {
                    "type": "array",
                    "description": "温度数据列表，每项包含 date(日期)、high_temp(最高温)、low_temp(最低温)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "high_temp": {"type": "number"},
                            "low_temp": {"type": "number"}
                        }
                    }
                },
                "rainfall_data": {
                    "type": "array",
                    "description": "降水数据列表，每项包含 date(日期)、rainfall(降水量mm)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "rainfall": {"type": "number"}
                        }
                    }
                },
                "start_date": {
                    "type": "string",
                    "description": "起始日期，格式：YYYY-MM-DD"
                },
                "end_date": {
                    "type": "string",
                    "description": "结束日期，格式：YYYY-MM-DD"
                }
            },
            "required": ["calculation_type"]
        }

    def execute(self, **kwargs) -> ToolResult:
        """
        执行农学计算

        Args:
            calculation_type: 计算类型
            crop_name: 作物名称（积温计算时需要）
            temperature_data: 温度数据
            rainfall_data: 降水数据
        """
        calc_type = kwargs.get("calculation_type")

        try:
            if calc_type == "gdd":
                result = self._calculate_gdd(kwargs)
            elif calc_type == "rainfall_sum":
                result = self._calculate_rainfall_sum(kwargs)
            elif calc_type == "rainfall_avg":
                result = self._calculate_rainfall_avg(kwargs)
            elif calc_type == "growing_period":
                result = self._estimate_growing_period(kwargs)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"不支持的计算类型: {calc_type}",
                )

            logger.info(f"农学计算成功: {calc_type}")
            return ToolResult(
                success=True,
                data=result,
                metadata={"calculation_type": calc_type}
            )

        except Exception as e:
            logger.error(f"农学计算失败: {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=f"计算失败: {str(e)}",
            )

    def _calculate_gdd(self, kwargs: Dict) -> Dict:
        """
        计算有效积温（GDD）

        公式：GDD = ((最高温 + 最低温) / 2) - 基准温度
        如果平均温度低于基准温度，则 GDD = 0
        """
        crop_name = kwargs.get("crop_name", "水稻")
        temperature_data = kwargs.get("temperature_data", [])

        if not temperature_data:
            return {
                "error": "缺少温度数据",
                "explanation": "请提供每日温度数据（日期、最高温、最低温）"
            }

        # 获取基准温度
        base_temp = CROP_BASE_TEMPS.get(crop_name, 10)

        daily_gdd = []
        total_gdd = 0

        for day in temperature_data:
            high = float(day.get("high_temp", 0))
            low = float(day.get("low_temp", 0))
            avg_temp = (high + low) / 2
            gdd = max(0, avg_temp - base_temp)

            daily_gdd.append({
                "date": day.get("date", ""),
                "avg_temp": avg_temp,
                "gdd": round(gdd, 1)
            })
            total_gdd += gdd

        return {
            "crop": crop_name,
            "base_temperature": base_temp,
            "total_gdd": round(total_gdd, 1),
            "days": len(daily_gdd),
            "daily_gdd": daily_gdd,
            "explanation": f"{crop_name}的基准温度为{base_temp}℃，累计有效积温为{round(total_gdd, 1)}℃·天",
            "required_gdd": CROP_REQUIRED_GDD.get(f"{crop_name}早稻") or CROP_REQUIRED_GDD.get(crop_name),
        }

    def _calculate_rainfall_sum(self, kwargs: Dict) -> Dict:
        """计算降水总量"""
        rainfall_data = kwargs.get("rainfall_data", [])

        if not rainfall_data:
            return {
                "error": "缺少降水数据",
                "explanation": "请提供每日降水数据（日期、降水量）"
            }

        total_rainfall = sum(float(day.get("rainfall", 0)) for day in rainfall_data)
        days = len(rainfall_data)

        return {
            "total_rainfall": round(total_rainfall, 1),
            "days": days,
            "avg_daily": round(total_rainfall / days, 1) if days > 0 else 0,
            "explanation": f"统计{days}天，累计降水量{round(total_rainfall, 1)}mm，日均{round(total_rainfall/days, 1) if days > 0 else 0}mm"
        }

    def _calculate_rainfall_avg(self, kwargs: Dict) -> Dict:
        """计算平均降水量"""
        rainfall_data = kwargs.get("rainfall_data", [])

        if not rainfall_data:
            return {
                "error": "缺少降水数据",
                "explanation": "请提供每日降水数据"
            }

        values = [float(day.get("rainfall", 0)) for day in rainfall_data]
        avg = sum(values) / len(values)
        max_val = max(values)
        min_val = min(values)

        return {
            "avg_rainfall": round(avg, 1),
            "max_rainfall": max_val,
            "min_rainfall": min_val,
            "days": len(values),
            "explanation": f"平均日降水量{round(avg, 1)}mm，最大{max_val}mm，最小{min_val}mm"
        }

    def _estimate_growing_period(self, kwargs: Dict) -> Dict:
        """
        根据积温估算作物发育期
        """
        crop_name = kwargs.get("crop_name", "水稻")
        total_gdd = kwargs.get("total_gdd", 0)

        if not crop_name or total_gdd <= 0:
            return {
                "error": "缺少必要参数",
                "explanation": "请提供作物名称和已累计的有效积温"
            }

        # 查找该作物的所需积温范围
        required_range = None
        for key, value in CROP_REQUIRED_GDD.items():
            if crop_name in key:
                required_range = value
                break

        if not required_range:
            return {
                "crop": crop_name,
                "current_gdd": total_gdd,
                "explanation": f"暂无{crop_name}的标准积温参考数据"
            }

        min_gdd, max_gdd = required_range
        progress = (total_gdd / max_gdd) * 100

        if total_gdd < min_gdd:
            stage = "生长初期"
            remaining = min_gdd - total_gdd
        elif total_gdd < max_gdd:
            stage = "生长中期"
            remaining = max_gdd - total_gdd
        else:
            stage = "生长后期/成熟期"
            remaining = 0

        return {
            "crop": crop_name,
            "current_gdd": total_gdd,
            "required_range": f"{min_gdd}-{max_gdd}℃·天",
            "progress": f"{round(min(progress, 100), 1)}%",
            "stage": stage,
            "remaining_gdd": round(remaining, 1),
            "explanation": f"{crop_name}已累计有效积温{total_gdd}℃，处于{stage}，进度约{round(min(progress, 100), 1)}%"
        }
