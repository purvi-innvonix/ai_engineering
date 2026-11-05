# What Is AI Engineering?

AI Engineering is the discipline of designing, developing, integrating, testing, and deploying software applications that use Artificial Intelligence (AI) technologies.

It combines traditional software engineering practices with AI models, data processing, APIs, and automation to build reliable, scalable, and maintainable AI-powered applications.

## Key Responsibilities of AI Engineering

- Integrating Large Language Models (LLMs): Incorporating LLMs into applications to generate text, answer questions, and perform language-based tasks.
- Developing Retrieval-Augmented Generation (RAG) Systems: Connecting AI models to external knowledge sources to provide context-aware and relevant responses.
- Implementing AI Agents and Automated Workflows: Enabling AI systems to perform tasks, use tools, and execute multi-step workflows.
- Connecting AI Models with External Systems: Integrating AI models with databases, APIs, and existing business applications.
- Evaluating AI Performance: Assessing response accuracy, quality, reliability, latency, and operational costs.
- Implementing Security, Monitoring, and Deployment Practices: Ensuring AI applications are secure, monitored, reliable, and suitable for production environments.

## Example of AI Engineering

### AI-Powered HR Assistant

An AI-powered HR assistant retrieves company policies and HR documents, answers employee questions using relevant information.

This application combines an LLM, a RAG system, document retrieval, and application APIs to deliver relevant and context-aware responses.

## 1. Generative AI

Generative AI is a category of artificial intelligence that generates new content based on learned patterns and provided inputs.

### Key Capabilities

- Text generation and summarization.
- Code generation.
- Image, audio, and video generation.
- Question answering and content transformation.
- Conversational applications.

### Example

A language model generates a summary of a YouTube transcript or produces a Python function from a natural-language description.

### Common Technologies

- GPT
- Google Gemini
- Claude
- Mistral
- Ollama-supported language models

## 2. AI Agents

An AI agent is a system that uses an AI model to determine appropriate actions and interact with tools or external systems to accomplish a defined task.

Depending on its design, an agent can select tools, execute actions, observe results, and continue processing until a task is completed or a stopping condition is reached.

### Key Capabilities

- Tool calling and API integration.
- Retrieving information from databases.
- Executing task-specific actions.
- Processing tool results.
- Handling multi-step tasks.

### Example

An AI agent receives a request to identify pending project tasks, calls a project-management API, retrieves the task information, and presents the results.

### Common Technologies

- LangChain
- LangGraph
- LLM tool-calling APIs
- Python
- REST APIs

## 3. Agentic AI

Agentic AI refers to AI systems designed to pursue goals through planning, decision-making, tool usage, and multi-step execution, with varying degrees of autonomy.

Unlike a simple question-answering application, an agentic system can coordinate multiple actions, evaluate intermediate results, and adapt its approach based on the available information.

### Key Capabilities

- Goal-oriented planning.
- Multi-step reasoning and execution.
- Dynamic selection of tools and actions.
- Evaluating intermediate outcomes.
- Adapting workflows based on results.
- Coordinating multiple agents or specialized components.

### Example

An agentic AI system investigates a delayed software project by examining task statuses, deadlines, dependencies, and reports. It identifies possible causes and generates recommended corrective actions.

Execution of consequential actions should be governed by appropriate permissions, validation, and human approval.

## 4. AI Developer

An AI Developer is a software development professional who implements AI capabilities within applications and existing software systems.

The role generally focuses on application development, model integration, API implementation, and the creation of AI-powered features.

### Key Responsibilities

- Integrating AI models through APIs or local inference.
- Developing AI chatbot interfaces and backend services.
- Implementing prompts and structured outputs.
- Connecting AI applications to databases and external services.
- Developing application logic and error handling.
- Testing and maintaining AI-powered features.

### Example

An AI Developer implements a chatbot backend using Python, FastAPI, LangChain, and a language model.

## 6. Difference Between Generative AI, AI Agents, Agentic AI, and AI Developer

| Concept | Definition | Primary Focus | Example |
|---|---|---|---|
| Generative AI | AI technology that generates content | Content generation | Text summarization |
| AI Agent | An AI-powered system that uses tools to accomplish tasks | Tool usage and task execution | Retrieving project tasks through an API |
| Agentic AI | An approach to designing systems that pursue goals through multi-step actions | Planning and autonomy | Investigating project delays |
| AI Developer | A professional who develops software incorporating AI capabilities | Application implementation | Developing an AI chatbot backend |
| AI Engineer | A professional focused on engineering AI systems throughout their lifecycle | Integration, evaluation, reliability, and deployment | Developing and deploying a production RAG application |

## Where LLM Applications Fit in Modern Software

LLM (Large Language Model) applications integrate AI models into existing software systems to enable natural-language understanding, content generation, information retrieval, and intelligent automation.

| Component | Role |
|---|---|
| Frontend | Provides the user interface, such as a chatbot. |
| Backend | Handles business logic, authentication, and APIs. |
| LLM Application Layer | Manages prompts, context, RAG, and tool calling. |
| LLM | Generates responses and processes language. |
| Data and Tools | Provide information through databases, vector stores, and external APIs. |

Example: A project management application with an AI assistant that answers questions about project status, summarizes tasks, and retrieves project information.

**Conclusion:** LLM applications act as an intelligence layer within modern software, extending traditional applications with AI-powered capabilities.

# How LLMs Work — Simple Explanation

## 1. What is an LLM?

A Large Language Model (LLM) is an AI model trained on large amounts of text and other data to understand and generate language.

### Example

**Input:** What is Python?

**Output:** Python is a programming language.

An LLM generates responses by predicting suitable next tokens based on the input and the context.

## 2. Important LLM Concepts

### 1. Tokens

Tokens are small units of text that an LLM processes. A token can be a word, part of a word, punctuation, or another text unit.

**Example:** I love AI!

The model might split this into tokens such as I, love, AI, and !. The exact tokenization depends on the model.

**Remember:** Tokens are not always the same as words.

### 2. Context Window

The context window is the amount of input and generated content a model can handle in a single interaction, measured in tokens.

**Example:**

A document contains 100,000 tokens, but a model has a context window of 10,000 tokens. The entire document cannot fit into one request without reducing or splitting the content.

**Remember:** The context window is the model's working space for the current interaction.

### 3. Transformer (High Level)

A Transformer is a neural network architecture used by many modern LLMs. It helps the model understand relationships between tokens using a mechanism called attention.

**Example:**

Sentence: The dog chased the ball because it was excited.

Attention helps the model learn relationships between words and use surrounding context to interpret what it refers to.

**Remember:** Transformers help models process context and relationships between tokens.

### 4. Temperature

Temperature controls how varied or predictable the model's output tends to be.

- Low temperature: Usually more predictable responses.
- High temperature: Usually more varied responses.

**Example:**

Question: Suggest a name for a chatbot.

- Low temperature: HelpBot
- High temperature: Cosmic Companion

### 5. Top-p

Top-p, also called nucleus sampling, controls token selection by considering a group of likely next tokens whose combined probability reaches a specified threshold.

**Example:** If top-p is set to 0.9, the model generally considers the smallest set of likely tokens whose cumulative probability reaches 90%, then samples from that set.

- Low top-p: The model considers a smaller group of likely tokens. Responses tend to be more focused and predictable.
- High top-p: The model considers a larger group of possible tokens, including less likely options. Responses can be more varied.

**Remember:** Temperature affects randomness; top-p limits the pool of candidate tokens. Both influence output variety.

### 6. Max Tokens

Max tokens sets the maximum number of tokens the model can generate in a response.

**Example:**

Prompt: Explain AI in detail.

- Max tokens = 20: The response may be very short or cut off.
- Max tokens = 200: The response has more room for explanation.

**Remember:** Max tokens limits output length; it does not guarantee a complete answer.

### 7. Hallucination

A hallucination occurs when an LLM generates incorrect, fabricated, or unsupported information that sounds convincing.

**Example:**

Question: Who invented Python?

Correct answer: Guido van Rossum.

A hallucinating model might confidently name a different person.

**How to reduce hallucinations:**

- Use reliable source material.
- Use RAG to retrieve relevant information.
- Validate important facts.
- Require citations where appropriate.

### 8. Reasoning Models

Reasoning models are designed to handle complex tasks by performing more deliberate internal processing before producing an answer.

**Example:**

Question: A project has 12 tasks. Four are complete. How many remain?

Answer: 8 tasks remain.

Reasoning-oriented models are particularly useful for complex coding, planning, and multi-step problem-solving tasks. Not every task requires one.

### 9. Chat Models

Chat models are designed to process conversational messages and generate responses. They commonly support roles such as system, user, and assistant.

**Example:**

- User: What is RAG?
- Assistant: RAG retrieves relevant information and supplies it to an LLM to help generate an answer.

Chat models can also support tool calling, structured output, and reasoning, depending on the model.

**Remember:** “Chat model” describes an interaction format or capability, while “reasoning model” describes an emphasis on complex problem-solving. These categories can overlap.

## Exercise 1: Compare LLM Providers

### Provider Comparison Table

| Provider | Response Quality | Speed | Reasoning | Coding Ability | Context Length | Cost |
|---|---|---|---|---|---|---|
| OpenAI | Very high; structured responses | Moderate to fast | Excellent on reasoning-focused models | Excellent debugging and code generation | Around 128K tokens for the models listed in the document | Free or paid options depending on model and usage |
| Anthropic | High; natural and detailed responses | Moderate to fast | Strong analytical capabilities | Excellent for complex coding tasks | Around 200K tokens for the models listed | Paid, model-dependent pricing |
| Google Gemini | High; strong multimodal capabilities | Fast | Strong mathematical and scientific reasoning | Very good for full-stack development | 1M–2M tokens for the models listed | Free or paid options depending on model and usage |
| Groq | Good to high; direct responses | Very fast inference | Depends on the underlying model | Strong for code snippets and explanations | Model-dependent; around 8K–128K tokens in the listed examples | Free tier and paid usage options |
| Mistral AI | High; concise responses | Fast | Strong logic and multilingual capabilities | Strong specialized coding models | Model-dependent; around 32K–128K tokens in the listed examples | Free or paid options depending on model and usage |

**Note:** Exact context limits, pricing, speed, and capabilities vary by model and API configuration.

### Key Observation Notes

1. Response quality: Providers differ in response style, instruction following, factual accuracy, and consistency.
2. Speed: Groq emphasizes fast inference, while actual response times depend on the model, prompt length, output length, and service conditions.
3. Reasoning: Reasoning-focused models are generally useful for complex logic, planning, and multi-step problems.
4. Coding ability: Different models vary in code generation, debugging, refactoring, and understanding larger codebases.
5. Context length: Gemini offers models with large context windows; however, a larger context window does not automatically guarantee better retrieval or accuracy.
6. Cost: Smaller models can be economical for routine tasks, while more capable models may be suitable for complex tasks. Actual costs depend on current pricing and token usage.

**Deliverable:** A comparative overview of five LLM providers, highlighting differences in response quality, speed, reasoning, coding, context capacity, and cost.

## Exercise 2: Experiment with LLM Parameters

### 1. Parameter Observation Table

| Parameter | Tested Values | Observed Behavior | Suitable Use Cases |
|---|---|---|---|
| Temperature | 0.0 | More predictable and focused responses | Classification, structured output |
| Temperature | 0.3 | Focused responses with some variation | Technical documentation, coding |
| Temperature | 0.8–1.0 | More diverse and creative responses | Brainstorming, creative writing |
| Temperature | Above 1.2 | May produce less coherent or inconsistent responses | Generally avoid for reliable outputs |
| Top-p | 0.1 | Restricts token selection to a narrow, high-probability group | More constrained generation |
| Top-p | 0.9–0.95 | Allows a broader range of likely tokens | General conversational responses |
| Max tokens | 20 | Responses may be cut off before completion | Short output limits |
| Max tokens | 500+ | Allows longer responses when needed | Detailed explanations and summaries |

### 2. Short Deliverable Notes

- Temperature: Lower values tend to produce more predictable responses, while higher values allow more variation. Lower temperature does not guarantee factual accuracy.
- Top-p: Lower values restrict the available token choices; higher values allow a wider range of likely tokens.
- Max tokens: Limits the length of generated output. A limit that is too low may truncate sentences, code, or structured responses.
- Hallucination: An LLM may confidently produce fabricated information when a question contains a false premise or requires information unavailable to the model.
- Grounding: Reliable documents, retrieval systems, and verified external sources can help reduce unsupported answers, but do not eliminate all errors.

### 3. Hallucination Experiment Table

| Test Category | Example Prompt | Expected Observation |
|---|---|---|
| Non-existent technical term | Explain the Laravel QuantumMiddlewareOptimizer class. | The model may invent methods or clarify that the class cannot be verified. |
| Fake historical event | Describe the Treaty of Neo-Verona signed in 1842. | The model may accept the false premise or identify that the event is unverified. |
| Impossible physical constraint | How do lead-weighted helium balloons inside a submarine make it float? | The model may explain the physical contradiction or generate an incorrect explanation. |
| Latest news | What was the headline news event in Tokyo yesterday? | Without current information access, the model may provide outdated or fabricated details. |
| Current weather | What is the temperature in Paris right now? | Without a live weather source, the model cannot reliably establish the current temperature. |
