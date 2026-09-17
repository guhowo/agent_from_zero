import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("DATA_MODEL"),
    api_key=os.getenv("DATA_API_KEY"),
    base_url=os.getenv("DATA_BASE_URL"),
    temperature = 0.7,
)

history = []
tools = []

agent = create_agent(
    model = llm,
    system_prompt="你是一个 helpful的助手。",
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
