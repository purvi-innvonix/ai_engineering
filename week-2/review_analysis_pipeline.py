import os
import json
import time
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from model_factory import get_model

load_dotenv()

# ============================================================
# Topic : Review Analysis Pipeline  (CAPSTONE)
# ============================================================
#
# 10 reviews → Method A (with_structured_output) vs
#              Method B (Prompt | LLM | PydanticOutputParser)
# Uses .batch() to process all reviews at once per run.
# Saves results to output/results_<provider>_<method>.json
# ============================================================


# ─────────────────────────────────────────────────────────────
# 1. Pydantic Schema
# ─────────────────────────────────────────────────────────────

class ReviewAnalysis(BaseModel):
    sentiment:      Literal["positive", "negative", "neutral", "mixed"]
    key_positives:  list[str]   = Field(description="Specific praise points. Empty list if none.")
    key_issues:     list[str]   = Field(description="Specific complaints. Empty list if none.")
    summary:        str         = Field(description="One neutral sentence about the review")
    recommendation: Literal["low", "medium", "high"] = Field(description="Follow-up urgency")


# ─────────────────────────────────────────────────────────────
# 2. Load reviews from sample.json
# ─────────────────────────────────────────────────────────────

with open(os.path.join(os.path.dirname(__file__), "sample.json")) as f:
    reviews: list[dict] = json.load(f)

batch_inputs = [{"product": r["product"], "review_text": r["text"]} for r in reviews]

print(f"Loaded {len(reviews)} reviews\n")


# ─────────────────────────────────────────────────────────────
# 3. Method A — with_structured_output()
#    Chain : Prompt | LLM (schema bound)  → ReviewAnalysis
# ─────────────────────────────────────────────────────────────

prompt_a = ChatPromptTemplate.from_template(
    "Analyze this '{product}' review and extract structured details:\n\n{review_text}"
)

timing = []

for provider in ["groq", "mistral", "gemini"]:
    try:
        llm_a = get_model(provider).with_structured_output(ReviewAnalysis)
    except ValueError as e:
        print(f"[SKIP] {provider}: {e}\n")
        continue

    chain_a = prompt_a | llm_a

    print(f"=== Method A | {provider.upper()} ===")
    start = time.time()
    try:
        results_a = chain_a.batch(batch_inputs)
        elapsed = round(time.time() - start, 2)

        for review, result in zip(reviews, results_a):
            print(f"\n  [{review['id']}] {review['product']}")
            print(f"    Sentiment      : {result.sentiment}")
            print(f"    Recommendation : {result.recommendation}")
            print(f"    Positives      : {', '.join(result.key_positives) or '—'}")
            print(f"    Issues         : {', '.join(result.key_issues) or '—'}")
            print(f"    Summary        : {result.summary}")

        # Save to JSON
        os.makedirs("output", exist_ok=True)
        out = [
            {**r.model_dump(), "id": reviews[i]["id"], "product": reviews[i]["product"]}
            for i, r in enumerate(results_a)
        ]
        with open(f"output/results_{provider}_method_a.json", "w") as f:
            json.dump(out, f, indent=2)

        print(f"\n  -> OK in {elapsed}s | saved: output/results_{provider}_method_a.json\n")
        timing.append((provider, "method_a", elapsed, "OK"))

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"  [FAILED] {e}\n")
        timing.append((provider, "method_a", elapsed, "FAILED"))


# ─────────────────────────────────────────────────────────────
# 4. Method B — Prompt | LLM | PydanticOutputParser
#    Format instructions injected via .partial()
# ─────────────────────────────────────────────────────────────

parser   = PydanticOutputParser(pydantic_object=ReviewAnalysis)

prompt_b = ChatPromptTemplate.from_messages([
    ("system", "You are a review analyst. Respond ONLY in this JSON format:\n{format_instructions}"),
    ("human",  "Product: {product}\n\nReview:\n{review_text}")
]).partial(format_instructions=parser.get_format_instructions())

for provider in ["groq", "mistral", "gemini"]:
    try:
        llm_b = get_model(provider)
    except ValueError as e:
        print(f"[SKIP] {provider}: {e}\n")
        continue

    chain_b = prompt_b | llm_b | parser

    print(f"=== Method B | {provider.upper()} ===")
    start = time.time()
    try:
        results_b = chain_b.batch(batch_inputs)
        elapsed = round(time.time() - start, 2)

        for review, result in zip(reviews, results_b):
            print(f"\n  [{review['id']}] {review['product']}")
            print(f"    Sentiment      : {result.sentiment}")
            print(f"    Recommendation : {result.recommendation}")
            print(f"    Positives      : {', '.join(result.key_positives) or '—'}")
            print(f"    Issues         : {', '.join(result.key_issues) or '—'}")
            print(f"    Summary        : {result.summary}")

        os.makedirs("output", exist_ok=True)
        out = [
            {**r.model_dump(), "id": reviews[i]["id"], "product": reviews[i]["product"]}
            for i, r in enumerate(results_b)
        ]
        with open(f"output/results_{provider}_method_b.json", "w") as f:
            json.dump(out, f, indent=2)

        print(f"\n  -> OK in {elapsed}s | saved: output/results_{provider}_method_b.json\n")
        timing.append((provider, "method_b", elapsed, "OK"))

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"  [FAILED] {e}\n")
        timing.append((provider, "method_b", elapsed, "FAILED"))


# ─────────────────────────────────────────────────────────────
# 5. Timing summary
# ─────────────────────────────────────────────────────────────

print("=" * 55)
print("TIMING SUMMARY")
print("=" * 55)
print(f"{'Provider':<12} {'Method':<12} {'Time(s)':>10} {'Status':>10}")
print("-" * 50)
for provider, method, elapsed, status in timing:
    print(f"{provider:<12} {method:<12} {elapsed:>10.2f} {status:>10}")

print("\nMethod A → schema bound to model  — fastest, cleanest")
print("Method B → schema in prompt+parser — more transparent")
print(".batch() → all 10 reviews at once, not one by one")
