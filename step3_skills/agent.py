"""Step3 Skills —— 用 langgraph 写一个简单的「查天气」Agent。

设计要点：
    1. 天气查询不是写死在 Agent 里的普通函数，而是封装成一个「技能（Skill）」，
       注册在技能库 SKILL_REGISTRY 中（见 step3_skills/skills.py）。
    2. Agent 用 langgraph 的 StateGraph 搭成一张有向图，包含两个节点：
           agent 节点  —— 让 LLM 阅读技能清单 + 对话历史，决定「调用技能」还是「直接回答」
           skill 节点  —— 真正执行技能，把结果作为观察（ToolMessage）回灌给 agent
       两者之间用条件边循环，直到 agent 给出最终答案（END）。
    3. 与 step4 一致：不依赖模型的 function-calling，改用 JSON 解析来兼容
       任意 OpenAI 兼容端点（如阿里云百炼）。

运行方式：
    python -m step3_skills.agent
或直接：
    python step3_skills/agent.py
"""

import json
import os
import re
import sys
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from step3_skills.skills import SKILL_REGISTRY

# ---------------------------------------------------------------------------
# 1. 初始化 LLM（沿用项目里 step1/step2 的 .env 配置方式）
# ---------------------------------------------------------------------------
load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DATA_MODEL"),
    api_key=os.getenv("DATA_API_KEY"),
    base_url=os.getenv("DATA_BASE_URL"),
    temperature=0.7,
)

# 单个用户问题里，允许 agent 反复调用技能的最大轮数，防止无限循环
MAX_ITER = 5

SYSTEM_PROMPT = """你是天气小助手，专业、友好、简洁。
你可以使用「技能库」里的技能来完成任务，规则如下：
1. 问天气必须调用 query_weather 技能获取真实数据，不要凭空编造。
2. 用户没说城市时，礼貌地先问清楚城市，不要乱猜。
3. 拿到技能返回的天气后，用自然的话回答，并根据温度给穿衣建议（<10度保暖，>30度防暑）。
4. 非天气问题，礼貌说明你主要提供天气服务。"""


# ---------------------------------------------------------------------------
# 2. 定义图状态（State）
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    # add_messages reducer：节点返回的新消息会自动追加/按 id 合并到列表
    messages: Annotated[list, add_messages]
    # agent 决定要调用的技能，形如 {"skill": "query_weather", "args": {"city": "杭州"}}
    next_skill: Optional[dict]
    # 已循环轮数，用于兜底防止死循环
    iterations: int


# ---------------------------------------------------------------------------
# 3. 工具函数
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict:
    """从模型返回文本中提取第一个 JSON 对象，容忍代码块包裹和多余文字。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        # 实在解析不出 JSON，就把整段文本当成最终回答
        return {"action": "answer", "answer": text}


def format_messages(messages: list) -> str:
    """把消息历史渲染成便于塞进 prompt 的对话文本。"""
    lines = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue  # 系统提示单独放，不重复渲染
        if isinstance(m, HumanMessage):
            lines.append(f"用户：{m.content}")
        elif isinstance(m, ToolMessage):
            lines.append(f"技能执行结果：{m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"助手：{m.content}")
    return "\n".join(lines) if lines else "（暂无对话）"


def fix_surrogates(text: str) -> str:
    """修复因终端 locale 非 UTF-8 导致 input() 用 surrogateescape 解码出的代理字符。"""
    try:
        return text.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    except Exception:
        return text


# ---------------------------------------------------------------------------
# 4. 节点一：agent（决策）
#    读技能清单 + 对话历史，让 LLM 决定「调用技能」还是「直接回答」。
# ---------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    iterations = state.get("iterations", 0) + 1

    prompt = f"""{SYSTEM_PROMPT}

【可用技能清单】
{SKILL_REGISTRY.describe()}

【对话历史】
{format_messages(state["messages"])}

请决定下一步怎么做，只输出 JSON，二选一：
1. 需要调用某个技能：
   {{"action": "skill", "skill": "技能名", "args": {{"参数名": "参数值"}}}}
2. 已经可以直接回答用户：
   {{"action": "answer", "answer": "给用户的自然语言回复"}}
"""
    resp = llm.invoke(prompt)
    data = extract_json(resp.content)

    # 超过最大轮数，强制收尾，避免死循环
    if iterations >= MAX_ITER and data.get("action") == "skill":
        return {
            "messages": [AIMessage(content="抱歉，我暂时没能完成这次查询，请稍后再试或换个说法。")],
            "next_skill": None,
            "iterations": iterations,
        }

    if data.get("action") == "skill":
        skill_name = data.get("skill", "")
        args = data.get("args", {}) or {}
        print(f"\n🧠 【agent 决策】调用技能 {skill_name}({args})")
        return {
            # 记一条简短的助手消息，说明它准备去用哪个技能
            "messages": [AIMessage(content=f"（准备调用技能：{skill_name}）")],
            "next_skill": {"skill": skill_name, "args": args},
            "iterations": iterations,
        }

    # action == "answer"
    answer = data.get("answer", resp.content.strip())
    return {
        "messages": [AIMessage(content=answer)],
        "next_skill": None,
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# 5. 节点二：skill（执行）
#    从技能库取出对应技能并执行，把结果作为 ToolMessage 回灌给 agent。
# ---------------------------------------------------------------------------
def skill_node(state: AgentState) -> dict:
    call = state.get("next_skill") or {}
    name = call.get("skill", "")
    args = call.get("args", {}) or {}

    skill = SKILL_REGISTRY.get(name)
    if skill is None:
        result = f"未找到名为 {name} 的技能。"
    else:
        try:
            result = skill.run(**args)
        except Exception as e:  # 技能内部报错不应让整个图崩掉
            result = f"技能 {name} 执行失败：{e}"

    print(f"🛠️  【技能执行】{name}({args}) -> {result}")

    return {
        "messages": [ToolMessage(content=result, tool_call_id=f"skill_{name}")],
        "next_skill": None,
    }


# ---------------------------------------------------------------------------
# 6. 条件路由：agent 之后去哪里
#    有待调用的技能 -> skill 节点；否则 -> 结束
# ---------------------------------------------------------------------------
def route_after_agent(state: AgentState) -> str:
    if state.get("next_skill"):
        return "skill_node"
    return END


# ---------------------------------------------------------------------------
# 7. 组装 langgraph 有向图
# ---------------------------------------------------------------------------
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent_node", agent_node)
    workflow.add_node("skill_node", skill_node)

    workflow.add_edge(START, "agent_node")          # 入口 -> 决策
    workflow.add_edge("skill_node", "agent_node")    # 技能执行完 -> 回到决策
    workflow.add_conditional_edges(                  # 决策 -> 调技能 or 结束
        "agent_node",
        route_after_agent,
        {"skill_node": "skill_node", END: END},
    )

    return workflow.compile()


# 编译好的图（全局复用）
agent = build_graph()


# ---------------------------------------------------------------------------
# 8. 交互式主循环
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 修复部分终端 locale 非 UTF-8（如 POSIX/C）时，中文输入/输出报错的问题
    for _stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=== Step3 Skills Demo：用 langgraph + 技能库查天气 ===")
    print(f"已注册技能：{SKILL_REGISTRY.names()}")
    print("示例输入：杭州今天天气怎么样？")
    print("输入 q 退出。\n")

    history = []  # 跨轮次保留的完整消息历史
    while True:
        try:
            user_input = fix_surrogates(input("你：").strip())
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not user_input:
            continue
        if user_input.lower() in {"q", "quit", "exit"}:
            print("再见！")
            break

        history.append(HumanMessage(content=user_input))
        try:
            result = agent.invoke(
                {"messages": history, "next_skill": None, "iterations": 0}
            )
        except Exception as e:  # 捕获 LLM 调用等异常，避免交互程序崩溃
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
            # 出错时把刚加入的用户消息回滚，避免污染历史
            history.pop()
            continue

        history = result["messages"]  # 用图返回的完整消息更新历史
        print("【agent回答】：", history[-1].content, "\n")
