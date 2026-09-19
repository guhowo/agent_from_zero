"""
Step4 Planning —— 根据天气制定旅行计划的 Agent（Plan-and-Execute / langgraph）。

对应 README 中描述的「先规划后执行」模式：
    Planner（规划器）    -> 把「旅行计划」任务拆解成结构化步骤列表
    Executor（执行器）   -> 逐步执行，执行过程中可调用 get_weather 工具查询天气
    Replanner（重规划器）-> 每步执行后判断：继续执行 / 重新规划 / 输出最终旅行计划

与 ReAct「边想边做」不同，这里先显式生成整体计划，再一步步执行，
并在执行中根据已查到的天气等结果动态重规划，降低漏步、跳步、顺序错等问题。

运行方式：
    python -m step4_planning.agent
或直接：
    python step4_planning/agent.py
"""

import json
import os
import re
import sys
from typing import List, TypedDict

import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

# ---------------------------------------------------------------------------
# 1. 初始化 LLM（沿用项目里 step1/step2 的 .env 配置方式）
# ---------------------------------------------------------------------------
load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DATA_MODEL"),
    api_key=os.getenv("DATA_API_KEY"),
    base_url=os.getenv("DATA_BASE_URL"),
    temperature=0.3,  # 规划类任务用较低温度，输出更稳定
)

# 执行单个步骤时，允许调用工具的最大轮数，防止无限循环
MAX_TOOL_ITER = 3


# ---------------------------------------------------------------------------
# 2. 定义图状态（State）
#    langgraph 的每个节点都会读取并更新这个共享状态
# ---------------------------------------------------------------------------
class PlanExecuteState(TypedDict):
    task: str                    # 用户的原始旅行需求（含目的地、天数等）
    plan: List[str]              # 尚未执行的步骤列表
    past_steps: List[List[str]]  # 已执行步骤及其结果，形如 [[step, result], ...]
    response: str                # 最终旅行计划（有值时代表任务完成）


# ---------------------------------------------------------------------------
# 3. 工具函数：健壮地从 LLM 文本里解析出 JSON
#    不依赖模型的 function-calling 能力，兼容任意 OpenAI 兼容端点
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict:
    """从模型返回文本中提取第一个 JSON 对象，容忍代码块包裹和多余文字。"""
    text = text.strip()
    # 去掉 ```json ... ``` 这类代码块围栏
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 退而求其次：抓取第一个 { 到最后一个 } 之间的内容
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"无法从模型输出中解析 JSON：\n{text}")


def format_past_steps(past_steps: List[List[str]]) -> str:
    """把已执行步骤格式化成便于塞进 prompt 的文本。"""
    if not past_steps:
        return "（暂无）"
    lines = []
    for i, (step, result) in enumerate(past_steps, 1):
        lines.append(f"{i}. 步骤：{step}\n   结果：{result}")
    return "\n".join(lines)


def fix_surrogates(text: str) -> str:
    """修复因终端 locale 非 UTF-8 导致 input() 用 surrogateescape 解码出的代理字符。

    例如中文「杭」的 UTF-8 首字节 0xE6 会被误解为 \\udce6，直接发给 API 会报
    'utf-8' codec can't encode character ... surrogates not allowed。
    这里把代理字符按原始字节还原为正确的 UTF-8 文本。
    """
    try:
        return text.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    except Exception:
        return text


def get_weather(city: str) -> str:
    """查询指定城市的当前天气，包括温度、天气状况、湿度、风速。
    当需要某个城市天气、气温、冷不冷、热不热、下不下雨时用这个工具。

    Args:
        city: 城市名称
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
            f"{cur['weatherDesc'][0]['value']}, "
            f"{cur['temp_C']}°C（体感{cur['FeelsLikeC']}°C）， "
            f"湿度{cur['humidity']}%, 风速{cur['windspeedKmph']}km/h"
        )
    except Exception:
        return (
            f"【{city}天气】（模拟数据，网络不通时显示）\n"
            f"天气：晴，温度：28°C（体感30°C），湿度：65%，风速：12km/h"
        )


def call_tool(name: str, args: dict) -> str:
    """执行器可调用的工具集合。目前提供天气查询。"""
    if name == "get_weather":
        return get_weather(args.get("city", ""))
    return f"未知工具：{name}"


# ---------------------------------------------------------------------------
# 4. 节点一：Planner（规划器）
#    接收旅行需求，生成结构化的步骤列表
# ---------------------------------------------------------------------------
def plan_step(state: PlanExecuteState) -> dict:
    task = state["task"]
    prompt = f"""你是一个旅行规划专家。请把下面的旅行需求拆解成一个简单、可执行的步骤列表。

要求：
1. 步骤要按执行顺序排列，数量控制在 3~5 步，不要过度拆分。
2. 其中必须包含一步「查询目的地天气」，因为旅行计划要根据天气来定。
3. 其余步骤可包括：根据天气推荐穿衣/装备、安排景点与行程、给出交通与注意事项等。
4. 每一步都要具体、可独立完成，后一步可以利用前一步的结果。
5. 只输出 JSON，不要输出多余解释。

输出格式：
{{"steps": ["第一步...", "第二步..."]}}

旅行需求：{task}
"""
    resp = llm.invoke(prompt)
    data = extract_json(resp.content)
    steps = data.get("steps", [])
    print("\n📋 【生成的旅行规划步骤】")
    for i, s in enumerate(steps, 1):
        print(f"   {i}. {s}")
    # 更新状态：写入初始计划
    return {"plan": steps}


# ---------------------------------------------------------------------------
# 5. 节点二：Executor（执行器）
#    取出计划里的第一步去执行；执行时可调用 get_weather 工具，
#    拿到天气后再产出该步骤结果，并记录到状态。
# ---------------------------------------------------------------------------
def execute_step(state: PlanExecuteState) -> dict:
    plan = state["plan"]
    if not plan:
        return {}

    current_step = plan[0]
    tool_notes: List[str] = []  # 记录本步骤中调用工具得到的信息
    result = None

    for _ in range(MAX_TOOL_ITER):
        tool_hint = (
            "\n已获得的工具调用结果：\n" + "\n".join(tool_notes)
            if tool_notes
            else ""
        )
        prompt = f"""你正在执行一个「根据天气制定旅行计划」的多步骤任务，请专注完成「当前步骤」。

原始旅行需求：{state["task"]}

已完成的步骤及结果：
{format_past_steps(state["past_steps"])}
{tool_hint}

当前需要执行的步骤：{current_step}

你可以使用以下工具：
- get_weather(city)：查询指定城市的当前天气（温度、天气状况、湿度、风速）

请判断当前步骤是否需要调用工具，然后二选一，只输出 JSON：
1. 若需要查询天气才能完成本步骤，输出：
   {{"action": "tool", "tool": "get_weather", "args": {{"city": "城市名"}}}}
2. 若已具备足够信息、可以直接给出本步骤结果，输出：
   {{"action": "result", "result": "当前步骤的完成结果"}}
"""
        resp = llm.invoke(prompt)
        data = extract_json(resp.content)

        if data.get("action") == "tool":
            tool_name = data.get("tool", "get_weather")
            args = data.get("args", {})
            out = call_tool(tool_name, args)
            print(f"\n🌤️  【调用工具】{tool_name}({args}) -> {out}")
            tool_notes.append(f"{tool_name}({json.dumps(args, ensure_ascii=False)}) -> {out}")
            continue

        # action == "result"
        result = data.get("result", resp.content.strip())
        break

    if result is None:
        # 达到最大轮数仍未给出结果，用已有工具信息兜底
        result = "；".join(tool_notes) or "（该步骤无结果）"

    print(f"\n⚙️  【执行步骤】{current_step}")
    print(f"   ✅ 结果：{result}")

    # 更新状态：剩余计划去掉第一步，并把 (步骤, 结果) 追加到 past_steps
    return {
        "plan": plan[1:],
        "past_steps": state["past_steps"] + [[current_step, result]],
    }


# ---------------------------------------------------------------------------
# 6. 节点三：Replanner（重规划器）
#    综合原始需求、已执行步骤及结果（含天气），决定：输出最终旅行计划 or 继续
# ---------------------------------------------------------------------------
def replan_step(state: PlanExecuteState) -> dict:
    prompt = f"""你是一个旅行规划专家。请根据当前进度决定下一步该做什么。

原始旅行需求：{state["task"]}

已完成的步骤及结果（其中包含目的地天气）：
{format_past_steps(state["past_steps"])}

剩余未执行的计划：{json.dumps(state["plan"], ensure_ascii=False)}

决策规则（务必遵守）：
- 如果「剩余未执行的计划」还不为空（不是 []），说明还有步骤没做完，必须选择 action=plan，把这些剩余步骤（可适当细化）继续执行，不要提前结束。
- 只有当「剩余未执行的计划」为空 [] 时，才选择 action=response，并基于以上所有已完成步骤的结果（尤其是真实天气数据），撰写一份详细、可执行的完整旅行计划正文。

只输出 JSON，二选一：
1. 继续执行剩余步骤：
   {{"action": "plan", "steps": ["下一步...", "再下一步..."]}}
2. 给出最终旅行计划（response 必须是真实、完整的计划正文，包含穿衣建议、每日行程、景点、餐饮、交通与注意事项；严禁照抄本提示里的任何占位说明文字）：
   {{"action": "response", "response": "<在这里撰写完整的旅行计划正文>"}}
"""
    resp = llm.invoke(prompt)
    data = extract_json(resp.content)
    action = data.get("action", "response")

    if action == "response":
        response = data.get("response", "")
        print("\n🏁 【规划完成，生成最终旅行计划】")
        return {"response": response}

    # action == "plan"：用新计划替换剩余计划，继续执行
    new_steps = data.get("steps", [])
    print("\n🔄 【重新规划】")
    for i, s in enumerate(new_steps, 1):
        print(f"   {i}. {s}")
    return {"plan": new_steps}


# ---------------------------------------------------------------------------
# 7. 条件路由：Replanner 之后走向哪里
#    有 response -> 结束；否则 -> 回到 Executor 继续执行
# ---------------------------------------------------------------------------
def should_continue(state: PlanExecuteState) -> str:
    if state.get("response"):
        return END
    # 没有最终答案，且也没有剩余步骤时，兜底结束，避免死循环
    if not state.get("plan"):
        return END
    return "execute_step"


# ---------------------------------------------------------------------------
# 8. 组装 langgraph 有向图
# ---------------------------------------------------------------------------
def build_graph():
    workflow = StateGraph(PlanExecuteState)

    # 添加节点
    workflow.add_node("plan_step", plan_step)
    workflow.add_node("execute_step", execute_step)
    workflow.add_node("replan_step", replan_step)

    # 添加边
    workflow.add_edge(START, "plan_step")            # 入口 -> 规划
    workflow.add_edge("plan_step", "execute_step")   # 规划 -> 执行
    workflow.add_edge("execute_step", "replan_step")  # 执行 -> 重规划
    # 重规划 -> 条件路由（继续执行 or 结束）
    workflow.add_conditional_edges(
        "replan_step",
        should_continue,
        {"execute_step": "execute_step", END: END},
    )

    return workflow.compile()


# 编译好的图（全局复用）
agent = build_graph()


# ---------------------------------------------------------------------------
# 9. 交互式主循环
# ---------------------------------------------------------------------------
def run_task(task: str) -> str:
    """对单个旅行需求运行完整的 Plan-and-Execute 流程，返回最终旅行计划。"""
    result = agent.invoke(
        {
            "task": task,
            "plan": [],
            "past_steps": [],
            "response": "",
        }
    )
    return result.get("response", "（未生成最终旅行计划）")


if __name__ == "__main__":
    # 修复部分终端 locale 非 UTF-8（如 POSIX/C）时，中文输入被 surrogateescape 解码、
    # 以及中文输出报错的问题：强制把标准流重设为 UTF-8。
    for _stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=== Step4 Planning Demo：根据天气制定旅行计划（langgraph）===")
    print("示例输入：帮我制定一个杭州两天一夜的旅行计划")
    print("Agent 会先规划步骤 -> 查询目的地天气 -> 结合天气给出完整旅行计划。")
    print("输入 q 退出。\n")
    while True:
        try:
            user_input = fix_surrogates(input("你（请输入旅行需求）：").strip())
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not user_input:
            continue
        if user_input.lower() in {"q", "quit", "exit"}:
            print("再见！")
            break

        try:
            final_answer = run_task(user_input)
        except Exception as e:  # 捕获 LLM 调用等异常，避免整个交互程序崩溃
            msg = str(e)
            print("\n" + "=" * 50)
            print("❌ 运行出错了：", msg)
            low = msg.lower()
            if "insufficient_quota" in low or "quota" in low or "403" in low:
                print(
                    "\n【原因】阿里云百炼免费额度已用尽 (insufficient_quota)，这不是代码问题。\n"
                    "【解决办法】任选其一：\n"
                    "  1. 登录阿里云百炼控制台充值；\n"
                    "  2. 或关闭工作空间的『仅使用免费额度 (use free tier only)』模式；\n"
                    "  3. 或把 .env 里的 DATA_MODEL 换成仍有可用额度的模型。\n"
                    "处理后重新运行即可。"
                )
            print("=" * 50 + "\n")
            continue

        print("\n" + "=" * 50)
        print("【最终旅行计划】：\n", final_answer)
        print("=" * 50 + "\n")
