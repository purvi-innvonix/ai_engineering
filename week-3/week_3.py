import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.3,
)

prompts = [
    "suggest me the nature places in the gujrat"
]

for prompt in prompts:
    response = llm.invoke([
        SystemMessage(content="You are a good AI tutor."),
        HumanMessage(content=prompt),
    ])

    print(f"\nPROMPT: {prompt}")
    print(response.content)