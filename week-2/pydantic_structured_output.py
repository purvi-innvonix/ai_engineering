import os
import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Literal
from groq import Groq
from langchain_groq import ChatGroq


load_dotenv()

# Topic: Pydantic Schema + Structured Output

# Pydantic = Python data validation library
# Schema   = defines what fields the data MUST have
# Structured Output = forces LLM response into a defined structure


# 1. Define a Pydantic schema
class Employee(BaseModel):
    name: str = Field(description="Full name of the employee")
    age: int = Field(
        ge=18,
        le=65,
        description="Age between 18 and 65"
    )
    department: str = Field(description="Department name")
    skills: List[str] = Field(description="List of skills")


print("=== Pydantic Schema (no API) ===")

emp = Employee(
    name="John Doe",
    age=28,
    department="Engineering",
    skills=["Python", "SQL"]
)

print(emp)
print("As dict:", emp.model_dump())
print("As JSON:", emp.model_dump_json())


# 2. Define the structured output schema
class BookSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="Book title")
    author: str = Field(description="Author name")

    genre: Literal[
        "fiction",
        "fantasy",
        "non-fiction",
        "science",
        "history"
    ] = Field(description="Book genre")

    one_line_summary: str = Field(
        description="One sentence summary of the book"
    )


# 3. Input text
text = """
The Hobbit was written by J.R.R. Tolkien in 1937.
It is a fantasy novel about a hobbit named Bilbo Baggins
who goes on an adventure.
"""

# Method 1 :
# 4. Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# 5. Ask Groq for structured JSON
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": text
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "book_summary",
            "strict": True,
            "schema": BookSummary.model_json_schema()
        }
    }
)

# 6. Convert JSON response → Python dictionary
data = json.loads(
    response.choices[0].message.content
)


# 7. Validate dictionary using Pydantic
result = BookSummary.model_validate(data)


# 8. Display result
print("\n=== Structured Output — Groq ===")

print(result)

print("\nAs dictionary:")
print(result.model_dump())

print("\nAs JSON:")
print(result.model_dump_json())


# Method 2 : call the llm using the invoke method 

# ─── Step 1: Create the LLM ───
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b")
# ─── Step 2: Bind the Pydantic schema to the LLM ───
structured_llm = llm.with_structured_output(BookSummary)
# ─── Step 3: Invoke — returns a BookSummary object directly ───
result: BookSummary = structured_llm.invoke(text)
print(result)
print("\nAs dictionary:", result.model_dump())
print("\nAs JSON:", result.model_dump_json())