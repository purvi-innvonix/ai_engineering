import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ============================================================
# Ollama Experiment: Local vs Cloud Model Comparison
# ============================================================
# This script compares responses from a local Ollama model
# with a cloud-based Groq model for the same prompts.
# ============================================================

# ─────────────────────────────────────────────────────────────
# 1. Model Setup
# ─────────────────────────────────────────────────────────────

# Cloud model (Groq)
cloud_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.3,
)

# Local model (Ollama)
# Make sure Ollama is running and the model is pulled first:
# ollama pull llama3.2
local_llm = ChatOllama(
    model="llama3.2",
    temperature=0.3,
)

# ─────────────────────────────────────────────────────────────
# 2. Test Prompts
# ─────────────────────────────────────────────────────────────

test_prompts = [
    "Explain machine learning in one sentence.",
    "What is the capital of France?",
    "Write a haiku about programming.",
    "Summarize the benefits of renewable energy.",
]

# ─────────────────────────────────────────────────────────────
# 3. Comparison Function
# ─────────────────────────────────────────────────────────────

def compare_models(prompt):
    print("\n" + "="*70)
    print(f"PROMPT: {prompt}")
    print("="*70)
    
    # Test Cloud Model
    print("\n--- CLOUD MODEL (Groq - openai/gpt-oss-120b) ---")
    start_cloud = time.time()
    try:
        cloud_response = cloud_llm.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt)
        ])
        cloud_time = time.time() - start_cloud
        print(f"Response: {cloud_response.content}")
        print(f"Time: {cloud_time:.2f}s")
        cloud_success = True
    except Exception as e:
        print(f"Error: {e}")
        cloud_response = None
        cloud_time = 0
        cloud_success = False
    
    # Test Local Model
    print("\n--- LOCAL MODEL (Ollama - llama3.2) ---")
    start_local = time.time()
    try:
        local_response = local_llm.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt)
        ])
        local_time = time.time() - start_local
        print(f"Response: {local_response.content}")
        print(f"Time: {local_time:.2f}s")
        local_success = True
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure Ollama is running with: ollama serve")
        print("And the model is pulled with: ollama pull llama3.2")
        local_response = None
        local_time = 0
        local_success = False
    
    # Comparison
    if cloud_success and local_success:
        print("\n--- COMPARISON ---")
        print(f"Cloud time: {cloud_time:.2f}s | Local time: {local_time:.2f}s")
        print(f"Speed difference: {abs(cloud_time - local_time):.2f}s")
        
        if local_time < cloud_time:
            print(f"Local model was {cloud_time/local_time:.1f}x faster")
        else:
            print(f"Cloud model was {local_time/cloud_time:.1f}x faster")
    
    return {
        "prompt": prompt,
        "cloud_response": cloud_response.content if cloud_success else None,
        "local_response": local_response.content if local_success else None,
        "cloud_time": cloud_time,
        "local_time": local_time,
        "cloud_success": cloud_success,
        "local_success": local_success
    }

# ─────────────────────────────────────────────────────────────
# 4. Run Experiments
# ─────────────────────────────────────────────────────────────

print("="*70)
print("OLLAMA EXPERIMENT: LOCAL VS CLOUD MODEL COMPARISON")
print("="*70)
print("\nPrerequisites:")
print("1. Ollama must be installed and running: ollama serve")
print("2. Model must be pulled: ollama pull llama3.2")
print("3. OPENAI_API_KEY must be set in .env for Groq")
print("\nStarting experiments...\n")

results = []

for prompt in test_prompts:
    result = compare_models(prompt)
    results.append(result)

# ─────────────────────────────────────────────────────────────
# 5. Summary
# ─────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

cloud_times = [r["cloud_time"] for r in results if r["cloud_success"]]
local_times = [r["local_time"] for r in results if r["local_success"]]

if cloud_times and local_times:
    avg_cloud = sum(cloud_times) / len(cloud_times)
    avg_local = sum(local_times) / len(local_times)
    
    print(f"\nAverage Cloud Response Time: {avg_cloud:.2f}s")
    print(f"Average Local Response Time: {avg_local:.2f}s")
    
    if avg_local < avg_cloud:
        print(f"\nLocal model was on average {avg_cloud/avg_local:.1f}x faster")
    else:
        print(f"\nCloud model was on average {avg_local/avg_cloud:.1f}x faster")

print("\n" + "="*70)
print("OBSERVATIONS")
print("="*70)
print("""
KEY DIFFERENCES BETWEEN LOCAL AND CLOUD MODELS:

1. Privacy & Security:
   - Local: Data stays on your machine, no external API calls
   - Cloud: Data sent to external servers, privacy concerns

2. Latency:
   - Local: No network latency, but depends on hardware
   - Cloud: Network latency added, but often faster on powerful servers

3. Cost:
   - Local: Free after initial setup, uses your hardware
   - Cloud: Pay-per-use, but no hardware costs

4. Model Quality:
   - Local: Limited to models that run on your hardware
   - Cloud: Access to latest, largest models (GPT-4, Claude, etc.)

5. Reliability:
   - Local: Depends on your setup, no downtime from provider
   - Cloud: Provider may have outages, rate limits

6. Setup Complexity:
   - Local: Requires installation, hardware, maintenance
   - Cloud: Simple API integration, no setup

USE CASES:

Choose LOCAL when:
- Privacy is critical (healthcare, finance, sensitive data)
- Offline access needed
- No budget for API costs
- Want full control over the model

Choose CLOUD when:
- Need the highest quality responses
- Have budget for API costs
- Want simplicity and minimal setup
- Need access to multiple models

RECOMMENDATION:
For development and experimentation, cloud models are often easier.
For production with sensitive data, consider local models or hybrid approaches.
""")
