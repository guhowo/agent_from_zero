"""Step3 Skills —— 「技能（Skill）」抽象与天气查询技能。

与 step2 的「工具（tool）」相比，技能是一个更完整、可复用、可被发现的能力单元：
每个技能都自带名字、功能描述、参数说明和执行函数，统一注册到「技能库」里。
Agent 不需要硬编码每个能力，而是通过阅读技能库的描述来「发现」有哪些技能，
再决定何时调用哪个技能。这样新增能力时，只要往技能库里注册一个新技能即可。

本文件对外暴露：
    SKILL_REGISTRY  —— 全局技能库（已注册 query_weather 天气查询技能）
    Skill           —— 技能数据结构
    SkillRegistry   —— 技能库容器
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import requests


# ---------------------------------------------------------------------------
# 1. 技能（Skill）：一个自包含的能力单元
# ---------------------------------------------------------------------------
@dataclass
class Skill:
    """描述一个可被 Agent 发现并调用的技能。

    Attributes:
        name:        技能唯一标识，Agent 调用时用它来指定技能。
        description: 技能能做什么、什么场景下该用它（给 LLM 阅读）。
        args_hint:   参数说明（给 LLM 阅读，让它知道要传哪些参数）。
        func:        真正干活的函数，接收关键字参数、返回字符串结果。
    """

    name: str
    description: str
    args_hint: str
    func: Callable[..., str]

    def run(self, **kwargs) -> str:
        """执行技能。"""
        return self.func(**kwargs)


# ---------------------------------------------------------------------------
# 2. 技能库（SkillRegistry）：统一注册、发现、描述所有技能
# ---------------------------------------------------------------------------
class SkillRegistry:
    """技能库：负责注册技能，并把技能清单描述成便于塞进 prompt 的文本。"""

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """注册一个技能（同名会被覆盖）。"""
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        """按名字取技能，取不到返回 None。"""
        return self._skills.get(name)

    def names(self) -> List[str]:
        """所有已注册技能的名字。"""
        return list(self._skills.keys())

    def describe(self) -> str:
        """把技能清单格式化成文本，供 Agent 放进 prompt 里「发现」可用技能。"""
        if not self._skills:
            return "（当前没有可用技能）"
        blocks = []
        for skill in self._skills.values():
            blocks.append(
                f"- 技能名：{skill.name}\n"
                f"  功能：{skill.description}\n"
                f"  参数：{skill.args_hint}"
            )
        return "\n".join(blocks)


# ---------------------------------------------------------------------------
# 3. 天气查询技能的具体实现
#    沿用 step2/step4 的做法：请求 wttr.in，网络不通时回退到模拟数据。
# ---------------------------------------------------------------------------
def query_weather(city: str) -> str:
    """查询指定城市的当前天气，包括温度、天气状况、湿度、风速。

    Args:
        city: 城市名称（中文或英文均可）
    """
    try:
        url = f"https://wttr.in/{city}?format=j1"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "curl/7.68.0"})
        # 请求如有问题，在此直接抛出异常，进入下方的模拟数据兜底
        resp.raise_for_status()
        data = resp.json()
        # 获取实时天气
        cur = data["current_condition"][0]
        area = data["nearest_area"][0]
        return (
            f"【{area['areaName'][0]['value']}天气】"
            f"{cur['weatherDesc'][0]['value']}，"
            f"{cur['temp_C']}°C（体感{cur['FeelsLikeC']}°C），"
            f"湿度{cur['humidity']}%，风速{cur['windspeedKmph']}km/h"
        )
    except Exception:
        return (
            f"【{city}天气】（模拟数据，网络不通时显示）\n"
            f"天气：晴，温度：28°C（体感30°C），湿度：65%，风速：12km/h"
        )


# ---------------------------------------------------------------------------
# 4. 组装全局技能库：把天气查询注册成一个技能
# ---------------------------------------------------------------------------
weather_skill = Skill(
    name="query_weather",
    description="查询指定城市的当前天气（温度、天气状况、湿度、风速）。"
    "当用户问某个城市天气、气温、冷不冷、热不热、下不下雨时使用。",
    args_hint='{"city": "城市名，例如 杭州 / Beijing"}',
    func=query_weather,
)

# 对外暴露的全局技能库；要新增能力，只需继续 register 新的 Skill 即可
SKILL_REGISTRY = SkillRegistry()
SKILL_REGISTRY.register(weather_skill)
