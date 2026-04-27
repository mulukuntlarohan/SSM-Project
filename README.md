#  Sequential-State Management (SSM) Framework
> **Preventing Semantic Drift in Multi-Turn Agentic Workflows**

## 📖 Project Overview
Large Language Models (LLMs) are powerful for iterative tasks, yet they suffer from a critical flaw: **Semantic Drift**. In long conversations, the model's context window becomes "noisy" due to recency bias, causing it to forget foundational constraints (such as programming language choices, technical standards, or stylistic rules) established at the start of the session.

The **SSM Framework** is an Agentic AI architecture designed to solve this problem. Instead of relying on the LLM's unreliable native memory, our system uses an **8-node LangGraph reasoning loop** to:
1. **Extract** project constraints into a structured, persistent **JSON State Map**.
2. **Detect Conflicts** in real-time when user intent contradicts established rules.
3. **Anchor** every response to a deterministic "Source of Truth" using automated prompt augmentation.
4. **Audit** results via a Reflexion node to ensure 100% adherence to the state map.

---

## 🛠️ Tech Stack
- **AI Engine:** Google Gemini 2.0 Flash (Native SDK)
- **Orchestration:** LangGraph (State-Machine Reasoning)
- **Data Validation:** Pydantic (Strongly-typed JSON schemas)
- **Memory Layer:** ChromaDB (Vector store for episodic history)
- **Observability:** LangSmith (Agentic Trace & Workflow Monitoring)
- **UI/Dashboard:** Streamlit (Research Interface)

---

## 🚀 Getting Started

### 1. Installation
First, clone the repository and install all required dependencies using the `requirements.txt` file:
```bash
git clone https://github.com/your-username/SSM-Project.git
cd SSM-Project
pip install -r requirements.txt
```
2. Environment Configuration
Create a .env file in the root directory of the project. You must provide your Gemini API key and LangSmith configuration for the agent to run correctly:
```bash
Env
# Google Gemini API Key (Get from Google AI Studio)
GEMINI_API_KEY=your_gemini_api_key_here
# LangSmith Configuration (For workflow tracing)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=SSM-Framework
```
3. Running the Application
Once the environment is set up, open your terminal and run the Streamlit dashboard:
```bash
streamlit run app/main.py
Note: If your file is named differently, use streamlit run app/app.py.
```
🕹️ Usage Instructions
Initialize the Project: Start the chat by providing your core technical rules (e.g., "I am building a fish detection system in Java using TensorFlow and No Comments").
Interactive Development: Continue chatting to build your logic.
Drift Detection: If you enter a prompt that contradicts your initial rules, a ⚠️ Drift Detected pop-up will appear. You can choose to "Accept" the new change or "Stick with Original" to keep the state map clean.
Live State Monitoring: Watch the sidebar to see the JSON State Map update in real-time as the Agent extracts information.
🔍 Observability (LangSmith)
To see the "Inner Workings" of the Agentic flow:
```bash
Open your LangSmith Dashboard.
Select the SSM-Framework project.
Inspect the Trace Graph to see how the Agent moves through the 8 nodes (Input -> Extraction -> Conflict Detection -> Generation -> Reflexion).
```
📊 Evaluation & Benchmarking
The project includes a scientific benchmarking script to compare SSM against vanilla LLM memory using the Attribute Retention Rate (ARR) metric.
To run the automated 30-turn stress test:
```bash
python benchmarks.py
```
