import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from step2_tools.tools import get_weather

load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("DATA_MODEL"),
    api_key=os.getenv("DATA_API_KEY"),
    base_url=os.getenv("DATA_BASE_URL"),
    temperature = 0.7,
)

history = []
tools = [get_weather]

agent = create_agent(
    model = llm,
    system_prompt="""
            你是天气小助手，专业友好。
                1. 问天气必须调用get_weather工具，不要编造  
                2. 回答简洁自然，根据温度给穿衣建议（<10度保暖，>30度防暑）  
                3. 用户没说城市时礼貌询问  
                4. 非天气问题礼貌说明你主要提供天气服务
            """,
    tools=tools,
)

while True:
    user_input = input("你：")
    # 1、添加用户消息到历史消息中
    history.append({"role": "user", "content" : user_input})
    # 2、调用agent，传入完整消息
    response = agent.invoke({"messages": history})
    # 3、用agent的返回完整的消息列表，更新历史消息
    history = response["messages"]
    # 4、最后一条消息就是最终答案
    print("【agent回答】：", history[-1].content)
