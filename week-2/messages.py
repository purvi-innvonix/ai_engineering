import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

load_dotenv()

# Topic: LangChain Messages
# Messages structure a conversation between system, human, and AI
# SystemMessage  → instructions to the AI (how to behave)
# HumanMessage   → the user's question or input
# AIMessage      → the AI's response
# ToolMessage    → result returned by a tool (used in agents)

# --- SystemMessage ---
system_msg = SystemMessage(content="You are a polite and concise assistant.")
print("=== SystemMessage ===")
print("Type:", system_msg.type)
print("Content:", system_msg.content)

# --- HumanMessage ---
human_msg = HumanMessage(content="What is the capital of France?")
print("\n=== HumanMessage ===")
print("Type:", human_msg.type)
print("Content:", human_msg.content)

# --- AIMessage ---
ai_msg = AIMessage(content="The capital of France is Paris.")
print("\n=== AIMessage ===")
print("Type:", ai_msg.type)
print("Content:", ai_msg.content)

# --- ToolMessage ---
tool_msg = ToolMessage(content="Population of Paris: 2.1 million", tool_call_id="lookup-001")
print("\n=== ToolMessage ===")
print("Type:", tool_msg.type)
print("Content:", tool_msg.content)

# --- Full Conversation Flow ---
print("\n=== Message Flow (no API needed) ===")
conversation = [system_msg, human_msg, ai_msg, tool_msg]
for msg in conversation:
    print(f"  [{msg.type.upper()}]: {msg.content}")
