# 🧪 AI Test Case Generator

A **multi-agent agentic AI system** for automated test case generation built with
**LangGraph**, **A2A Protocol**, **GPT-4o**, and **Gradio**.

---

## Architecture

```
User Input (Gradio UI)
        │
        ▼
┌─────────────────────┐
│  DataIngestionAgent │  Reads POC_TestCase_Chatbot_Data.xlsx
└────────┬────────────┘
         │ A2A: data_context
         ▼
┌──────────────────────────────┐
│  RequirementAnalysisAgent    │  GPT-4o extracts entities/actions/constraints
└────────────┬─────────────────┘
             │ A2A: requirements
             ▼
┌──────────────────────────────┐
│  TestCaseGeneratorAgent      │  GPT-4o generates N structured test cases
└────────────┬─────────────────┘
             │ A2A: test_cases
             ▼
┌──────────────────────────────┐
│  TestCaseReviewerAgent       │  GPT-4o reviews quality (score 0–1)
└────────────┬─────────────────┘
             │  score < 0.70 AND retries < 2  →  back to Generator
             │  score >= 0.70
             ▼
┌──────────────────────────────┐
│  OutputFormatterAgent        │  DataFrame + Excel bytes
└────────────┬─────────────────┘
             ▼
     Gradio UI  (table · JSON · download · trace)
```

---

## Agents

| Agent | Role |
|---|---|
| **DataIngestionAgent** | Loads `POC_TestCase_Chatbot_Data.xlsx`, extracts schema & domain context |
| **RequirementAnalysisAgent** | Uses GPT-4o to parse the feature description into structured requirements |
| **TestCaseGeneratorAgent** | Uses GPT-4o to generate test cases matching the Excel column schema |
| **TestCaseReviewerAgent** | Uses GPT-4o to review completeness, clarity, coverage; assigns quality score |
| **OutputFormatterAgent** | Converts approved cases to DataFrame + downloadable Excel |

---

## A2A Protocol

Every inter-agent message is wrapped in an `A2AMessage` envelope:

```python
@dataclass
class A2AMessage:
    sender: str
    receiver: str
    message_type: str   # data_context | requirements | test_cases | review_result | output
    payload: dict
    trace_id: str       # shared UUID per pipeline run
    timestamp: str
```

The full trace is viewable in the **Agent Trace** tab of the UI.

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/shashankpandey9/test-case-generation.git
cd test-case-generation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 4. Run
python app.py
```

Open **http://localhost:7860** in your browser.

---

## File Structure

```
├── app.py                          # Gradio UI (4 tabs)
├── graph.py                        # LangGraph StateGraph + run_pipeline()
├── a2a_protocol.py                 # A2A message bus
├── data_loader.py                  # Excel reader
├── agents/
│   ├── data_ingestion_agent.py
│   ├── requirement_analysis_agent.py
│   ├── test_case_generator_agent.py
│   ├── test_case_reviewer_agent.py
│   └── output_formatter_agent.py
├── utils/
│   └── helpers.py                  # Excel export, DataFrame formatter
├── requirements.txt
├── .env.example
└── POC_TestCase_Chatbot_Data.xlsx  # Data source
```

---

## Tech Stack

- **LangGraph** — stateful multi-agent orchestration with conditional retry loop
- **A2A Protocol** — lightweight typed inter-agent messaging with full trace
- **GPT-4o** via LangChain — powers 3 of the 5 agents
- **Gradio** — multi-tab web UI
- **pandas + openpyxl** — data I/O and Excel export
