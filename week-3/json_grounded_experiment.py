
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.3,
)

# ---------------------------------
# EXPERIMENT 1: JSON OUTPUT
# ---------------------------------

json_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Analyze the product review and return ONLY valid JSON.
        Do not include markdown or explanations outside JSON.

        Use these exact keys:
        sentiment, product, reason
        """
    ),
    ("human", "Review: {review}")
])

json_chain = json_prompt | llm

review = "These headphones have excellent sound quality, but the battery is poor."

response = json_chain.invoke({"review": review})

print("\n--- JSON OUTPUT EXPERIMENT ---")
print(response.content)

# Parse the response into a Python dictionary
try:
    data = json.loads(response.content)
    print("\nParsed Python dictionary:")
    print(data)

    print("\nSentiment:", data["sentiment"])
    print("Product:", data["product"])

except (json.JSONDecodeError, KeyError) as error:
    print("Could not parse the expected JSON:", error)


# ---------------------------------
# EXPERIMENT 2: GROUNDED ANSWERS
# ---------------------------------

grounded_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Answer using ONLY the provided context.

        Rules:
        - Do not use outside knowledge.
        - If the answer is not in the context, say:
          "I don't know based on the provided context."
        - Keep the answer short.
        """
    ),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
])

grounded_chain = grounded_prompt | llm

context = """
StorySpace is a social storytelling application.
Users can publish stories and follow other users.
The backend uses Node.js, Express.js, TypeScript,
TypeORM, and PostgreSQL.
"""

print("\n--- GROUNDED ANSWERS EXPERIMENT ---")

questions = [
    "Which database does StorySpace use?",
    "Can users follow other users?",
    "Which vector database does StorySpace use?",
]

for question in questions:
    answer = grounded_chain.invoke({
        "context": context,
        "question": question,
    })

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer.content}")
