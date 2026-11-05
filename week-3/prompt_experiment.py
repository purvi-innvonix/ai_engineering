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

task = "Explain quantum computing to a 10-year-old"

# ==================================
# PROMPT 1: Basic, no guidance
# ==================================
print("\n" + "="*60)
print("PROMPT 1: Basic, no guidance")
print("="*60)

response1 = llm.invoke([
    HumanMessage(content=task)
])
print(response1.content)

# ==================================
# PROMPT 2: With persona and audience
# ==================================
print("\n" + "="*60)
print("PROMPT 2: With persona and audience specification")
print("="*60)

response2 = llm.invoke([
    SystemMessage(content="You are a patient and creative science teacher who specializes in explaining complex topics to children."),
    HumanMessage(content=task)
])
print(response2.content)

# ==================================
# PROMPT 3: With structure and examples
# ==================================
print("\n" + "="*60)
print("PROMPT 3: With structure and examples")
print("="*60)

response3 = llm.invoke([
    SystemMessage(content="""You are a science educator. When explaining complex topics:
1. Start with a simple analogy
2. Explain the core concept in 2-3 sentences
3. Give one concrete example
4. Keep sentences short and simple
5. Avoid technical jargon"""),
    HumanMessage(content=task)
])
print(response3.content)

# ==================================
# PROMPT 4: With constraints and format
# ==================================
print("\n" + "="*60)
print("PROMPT 4: With constraints and format")
print("="*60)

response4 = llm.invoke([
    SystemMessage(content="""You are explaining quantum computing to a 10-year-old.
Format your answer as:
- ANALOGY: [simple comparison]
- WHAT IT IS: [2-3 sentence explanation]
- EXAMPLE: [one real-world use case]
- FUN FACT: [one interesting detail]

Keep each section under 30 words. Use simple words only."""),
    HumanMessage(content=task)
])
print(response4.content)

# ==================================
# OBSERVATIONS
# ==================================
print("\n" + "="*60)
print("OBSERVATIONS")
print("="*60)
print("""
Prompt 1 (Basic): 
- Response is generic and may be too technical
- No clear structure or target audience consideration

Prompt 2 (Persona + Audience):
- More engaging and child-friendly tone
- Uses simpler language and relatable examples
- Better suited for the target audience

Prompt 3 (Structure + Examples):
- Well-organized with clear sections
- Follows the specified structure consistently
- More comprehensive but still accessible

Prompt 4 (Constraints + Format):
- Most structured and controlled output
- Strict adherence to format requirements
- Concise and easy to read
- Best for consistent, predictable outputs

Key Takeaway: Adding persona, structure, and constraints significantly improves the quality and consistency of AI responses for specific tasks and audiences.
""")
