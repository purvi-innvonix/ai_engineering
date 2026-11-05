import os
import json
import time
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq

load_dotenv()

# ============================================================
# Week 2 Pipeline Improvement
# ============================================================
# This file improves the prompt from Week 2's review_analysis_pipeline.py
# Original prompt was basic and lacked detailed instructions
# Improved version adds: persona, clear guidelines, examples, and constraints
# ============================================================

# ─────────────────────────────────────────────────────────────
# 1. Pydantic Schema (same as Week 2)
# ─────────────────────────────────────────────────────────────

class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    key_positives: list[str] = Field(description="Specific praise points. Empty list if none.")
    key_issues: list[str] = Field(description="Specific complaints. Empty list if none.")
    summary: str = Field(description="One neutral sentence about the review")
    recommendation: Literal["low", "medium", "high"] = Field(description="Follow-up urgency")

# ─────────────────────────────────────────────────────────────
# 2. Sample review for testing
# ─────────────────────────────────────────────────────────────

sample_review = {
    "id": 1,
    "product": "Wireless Headphones",
    "text": "The sound quality is amazing but the battery only lasts 4 hours. Comfortable for long sessions though."
}

# ─────────────────────────────────────────────────────────────
# 3. ORIGINAL PROMPT (from Week 2)
# ─────────────────────────────────────────────────────────────

original_prompt = ChatPromptTemplate.from_template(
    "Analyze this '{product}' review and extract structured details:\n\n{review_text}"
)

print("="*60)
print("ORIGINAL PROMPT (Week 2)")
print("="*60)
print(original_prompt.format(product=sample_review["product"], review_text=sample_review["text"]))

# ─────────────────────────────────────────────────────────────
# 4. IMPROVED PROMPT (Week 3)
# ─────────────────────────────────────────────────────────────

improved_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert product review analyst with 10 years of experience in customer feedback analysis.

Your task is to analyze product reviews and extract structured insights.

ANALYSIS GUIDELINES:
1. Sentiment Classification:
   - POSITIVE: Overall satisfied, praise outweighs complaints
   - NEGATIVE: Overall dissatisfied, complaints outweigh praise
   - NEUTRAL: Balanced feedback or factual statements
   - MIXED: Strong positive AND strong negative elements present

2. Key Points Extraction:
   - Extract specific, actionable points (not generic statements)
   - Group related points together
   - Use customer's own words when possible
   - Maximum 3-5 points per category

3. Summary Writing:
   - One neutral, objective sentence
   - Capture the main theme without bias
   - Under 25 words

4. Recommendation Level:
   - HIGH: Urgent issues, safety concerns, or severe dissatisfaction
   - MEDIUM: Moderate issues that need attention
   - LOW: Minor issues or general feedback

OUTPUT FORMAT:
Return valid JSON with these exact keys: sentiment, key_positives, key_issues, summary, recommendation"""),
    ("human", """Product: {product}

Review Text:
{review_text}

Analyze this review following the guidelines above.""")
])

print("\n" + "="*60)
print("IMPROVED PROMPT (Week 3)")
print("="*60)
print(improved_prompt.format(product=sample_review["product"], review_text=sample_review["text"]))

# ─────────────────────────────────────────────────────────────
# 5. COMPARISON TEST
# ─────────────────────────────────────────────────────────────

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.3,
)

parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

# Original chain
chain_original = original_prompt | llm.with_structured_output(ReviewAnalysis)

# Improved chain
chain_improved = improved_prompt | llm.with_structured_output(ReviewAnalysis)

print("\n" + "="*60)
print("TESTING BOTH PROMPTS")
print("="*60)

test_reviews = [
    {"product": "Wireless Headphones", "text": "The sound quality is amazing but the battery only lasts 4 hours. Comfortable for long sessions though."},
    {"product": "Running Shoes", "text": "Terrible quality! Fell apart after 2 weeks. Do not buy."},
    {"product": "Coffee Maker", "text": "Works fine, makes good coffee. Nothing special but does the job."},
]

print("\n--- ORIGINAL PROMPT RESULTS ---")
for i, review in enumerate(test_reviews, 1):
    result = chain_original.invoke(review)
    print(f"\nReview {i} ({review['product']}):")
    print(f"  Sentiment: {result.sentiment}")
    print(f"  Positives: {result.key_positives}")
    print(f"  Issues: {result.key_issues}")
    print(f"  Summary: {result.summary}")
    print(f"  Recommendation: {result.recommendation}")

print("\n--- IMPROVED PROMPT RESULTS ---")
for i, review in enumerate(test_reviews, 1):
    result = chain_improved.invoke(review)
    print(f"\nReview {i} ({review['product']}):")
    print(f"  Sentiment: {result.sentiment}")
    print(f"  Positives: {result.key_positives}")
    print(f"  Issues: {result.key_issues}")
    print(f"  Summary: {result.summary}")
    print(f"  Recommendation: {result.recommendation}")

# ─────────────────────────────────────────────────────────────
# 6. OBSERVATIONS
# ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("OBSERVATIONS")
print("="*60)
print("""
IMPROVEMENTS MADE:

1. Added Persona:
   - "Expert product review analyst with 10 years experience"
   - Gives the model a specific role and expertise level

2. Detailed Guidelines:
   - Clear definitions for each sentiment category
   - Specific rules for extracting key points
   - Word count constraints for summary
   - Criteria for recommendation levels

3. Structure:
   - Organized into numbered sections
   - Easier for the model to follow
   - Reduces ambiguity

4. Output Format:
   - Explicitly mentions JSON format
   - Lists exact keys required
   - Ensures consistent structure

EXPECTED BENEFITS:
- More consistent sentiment classification
- Better extraction of specific points (fewer generic statements)
- More appropriate recommendation levels
- More neutral, concise summaries
- Overall more reliable and actionable analysis

The improved prompt should produce more consistent, accurate, and actionable results compared to the basic original prompt.
""")
