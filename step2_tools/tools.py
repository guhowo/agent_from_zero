import requests
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气，包括温度、天气状况、湿度、风速。
    当用户问某个城市天气、气温、冷不冷、热不热、下不下雨时用这个工具。

    Args:
        city: 城市名称
    """

    try:
        url = f"https://wttr.in/{city}?format=1&lang=zh"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "curl/7.68.0"})
        # 请求如有问题，在此直接抛出异常，而不是往后走
        resp.raise_for_status()
        data = resp.json()
        # 获取实时天气
        cur = data["current_condition"][0]
        area = data["nearest_area"][0]
        return (
            f"【{area['areaName'][0]['value']}天气】"
            f"{cur['weatherDesc'][0]['value']}, "
            f"{cur['temp_C']}°C（体感{cur['FeelsLikeC']}°C）， "
            f"湿度{cur['humidity']}%, 风速{cur['windspeedKmph']}km/h"
        )
    except Exception:
        return (
            f"【{city}天气】（模拟数据，网络不通时显示）\n"
            f"天气：晴，温度：28°C（体感30°C），湿度：65%，风速：12km/h"
        )