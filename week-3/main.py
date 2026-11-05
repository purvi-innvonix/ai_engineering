# main.py
import os
import json
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain Core & Groq Integrations
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

# LangChain SQL Tools
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent

load_dotenv()

app = FastAPI(title="StorySpace AI Agent (LangChain Edition)")

INTERNAL_KEY = os.environ["AI_INTERNAL_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]
GROQ_MODEL = "openai/gpt-oss-120b"

# ==========================================
# 1. LangChain Model Integration
# ==========================================
llm = ChatGroq(
    model_name=GROQ_MODEL,
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.0
)

# ==========================================
# 2. SQL Database Setup
# ==========================================
db = SQLDatabase.from_uri(DATABASE_URL)

# Create SQL toolkit
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = sql_toolkit.get_tools()

# ==========================================
# 3. General Conversation Tool
# ==========================================
@tool
async def general_conversation(query: str) -> str:
    """Handle general conversation, suggestions, and non-database queries.
    
    Use this for:
    - Username suggestions
    - Creative ideas
    - General advice
    - Non-database related questions
    
    Args:
        query: The user's question or request
        
    Returns:
        A helpful response
    """
    # Use the LLM directly for general conversation
    response = await llm.ainvoke(query)
    return response.content

# ==========================================
# 4. Combine All Tools
# ==========================================
tools = sql_tools + [general_conversation]

# ==========================================
# 5. Create Agent with All Tools
# ==========================================
system_prompt = """You are StorySpace's friendly assistant with access to a PostgreSQL database and general conversation capabilities.

Available tools:
- SQL tools: Use these when users ask about stories,likes,comments,followers-following & search for content, or need database information
  - sql_db_query: Execute SQL queries on the database
  - sql_db_schema: Check table schemas
  - sql_db_list_tables: List available tables
- general_conversation: Use this for suggestions, creative ideas, general advice, or non-database questions

Database tables:
- stories: Contains story information (id, title, slug, content, summary, status, visibility, etc.)
- users: User information
- comments: Comments on stories
- likes: Likes on stories
- follows: User follow relationships

When to use SQL tools:
- User asks about stories, likes, comments, followers/following, search for content, or needs story information
- Filter for status='published' and visibility='public' when querying stories
- Only use SELECT queries, never modify data

When to use general_conversation:
- User asks for suggestions (username, story ideas, etc.)
- User asks for creative advice
- User asks general questions not related to the database
- User wants help with anything that doesn't require database access"""

agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    debug=True
)


# ==========================================
# API Request Model
# ==========================================
class AgentRequest(BaseModel):
    message: str
    userId: str
    role: str


# ==========================================
# FastAPI Agent Endpoint
# ==========================================
@app.post("/agent")
async def agent(req: AgentRequest, x_internal_key: str = Header()):
    if x_internal_key != INTERNAL_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Invoke the agent with tools enabled
    # The agent will automatically decide between SQL tools and general conversation
    result = await agent_executor.ainvoke({
        "messages": [{"role": "user", "content": req.message}]
    })

    # Extract the last message (the AI's response)
    messages = result.get("messages", [])
    if messages:
        answer = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
    else:
        answer = "No response generated"

    # Validate Output
    if len(answer) > 3000:
        answer = answer[:3000] + "…"

    if req.role != "admin" and "rejectionReason" in answer.lower():
        answer = "[Content moderated]"

    return {"reply": answer}


@app.get("/health")
def health():
    return {"status": "ok"}

