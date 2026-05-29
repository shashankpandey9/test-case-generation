"""
Gradio UI — multi-tab interface for the test-case generation pipeline.
Tabs: Generate | Data Preview | Pytest Code | Agent Trace | About
"""

from __future__ import annotations

import json
import os
import tempfile

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from data_loader import get_preview_dataframe
from graph import run_pipeline

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "POC_TestCase_Chatbot_Data.xlsx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_tmp(content: bytes | str, suffix: str) -> str:
    """Write content to a named tmp file and return the path."""
    mode = "wb" if isinstance(content, bytes) else "w"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode=mode)
    tmp.write(content)
    tmp.flush()
    return tmp.name


# ---------------------------------------------------------------------------
# Core generation callback
# ---------------------------------------------------------------------------

def generate_test_cases(feature_description, test_count, test_types, state_store):
    """Run the full 6-agent pipeline and return UI outputs."""
    if not feature_description.strip():
        empty = pd.DataFrame()
        return "⚠️ Please enter a feature description.", empty, "", None, "", None, state_store

    try:
        final_state = run_pipeline(
            user_input=feature_description,
            test_count=int(test_count),
            test_types=test_types,
            excel_path=EXCEL_PATH,
        )
    except Exception as exc:
        empty = pd.DataFrame()
        return f"❌ Pipeline error: {exc}", empty, "", None, "", None, state_store

    if final_state.get("error"):
        empty = pd.DataFrame()
        return f"❌ {final_state['error']}", empty, "", None, "", None, state_store

    output        = final_state.get("output", {})
    df            = output.get("dataframe", pd.DataFrame())
    excel_bytes   = output.get("excel_bytes", b"")
    quality_score = output.get("quality_score", 0.0)
    feedback      = output.get("overall_feedback", "")
    trace         = final_state.get("a2a_trace", [])
    pytest_code   = final_state.get("pytest_code", "# pytest code not generated")

    excel_path_tmp  = _save_tmp(excel_bytes, ".xlsx") if excel_bytes else None
    pytest_path_tmp = _save_tmp(pytest_code, ".py")
    json_text       = json.dumps(output.get("test_cases", []), indent=2)
    trace_text      = json.dumps(trace, indent=2)

    status_md = (
        f"✅ **{len(df)} test cases generated** — "
        f"Quality score: `{quality_score:.2f}` — {feedback}"
    )

    state_store = {
        "df": df,
        "trace": trace_text,
        "json": json_text,
        "pytest_code": pytest_code,
    }

    return status_md, df, json_text, excel_path_tmp, pytest_code, pytest_path_tmp, state_store


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="🧪 AI Test Case Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🧪 AI Test Case Generator\n"
            "Multi-agent pipeline powered by **LangGraph + A2A Protocol + GPT-4o**\n\n"
            "**6 Agents:** DataIngestion → RequirementAnalysis → TestCaseGenerator "
            "→ TestCaseReviewer → OutputFormatter → **TestCodeGenerator (pytest)**"
        )

        state_store = gr.State({})

        # ── Tab 1: Generate ────────────────────────────────────────────────
        with gr.Tab("🚀 Generate Test Cases"):
            with gr.Row():
                with gr.Column(scale=2):
                    feature_input = gr.Textbox(
                        label="Feature / Module Description",
                        placeholder="e.g. User login with email and password, including MFA support",
                        lines=4,
                    )
                with gr.Column(scale=1):
                    test_count = gr.Dropdown(
                        label="Number of Test Cases",
                        choices=[5, 10, 15, 20],
                        value=10,
                    )
                    test_types = gr.Dropdown(
                        label="Test Types",
                        choices=["All", "Positive only", "Negative only", "Edge cases"],
                        value="All",
                    )
                    generate_btn = gr.Button("🚀 Generate Test Cases", variant="primary")

            status_md = gr.Markdown("")
            result_df = gr.Dataframe(label="Generated Test Cases", wrap=True, interactive=False)

            with gr.Accordion("📋 JSON Preview", open=False):
                json_output = gr.Code(language="json", label="Test Cases JSON")

            with gr.Row():
                excel_download = gr.File(label="⬇️ Download Excel (.xlsx)", interactive=False)
                pytest_download = gr.File(label="⬇️ Download pytest File (.py)", interactive=False)

            pytest_hidden = gr.Textbox(visible=False)

            generate_btn.click(
                fn=generate_test_cases,
                inputs=[feature_input, test_count, test_types, state_store],
                outputs=[
                    status_md, result_df, json_output,
                    excel_download, pytest_hidden, pytest_download,
                    state_store,
                ],
            )

        # ── Tab 2: Data Preview ────────────────────────────────────────────
        with gr.Tab("📊 Data Source Preview"):
            gr.Markdown("### Source Excel: `POC_TestCase_Chatbot_Data.xlsx`")
            try:
                preview_df = get_preview_dataframe(EXCEL_PATH, n=20)
                gr.Markdown(
                    f"**Columns:** {', '.join(preview_df.columns.tolist())}  \n"
                    f"**Showing:** first 20 rows"
                )
                gr.Dataframe(value=preview_df, interactive=False, wrap=True)
            except Exception as exc:
                gr.Markdown(f"⚠️ Could not load preview: {exc}")

        # ── Tab 3: Pytest Code ─────────────────────────────────────────────
        with gr.Tab("🧬 Pytest Code"):
            gr.Markdown(
                "### Auto-generated pytest code\n"
                "Generated by **TestCodeGeneratorAgent** (Agent 6). "
                "Run a generation first, then click **Refresh**."
            )

            def show_pytest(store):
                return store.get("pytest_code", "# Run generation first to see pytest code here.")

            refresh_pytest_btn = gr.Button("🔄 Refresh Pytest Code")
            pytest_display = gr.Code(language="python", label="Generated pytest Code")
            refresh_pytest_btn.click(fn=show_pytest, inputs=[state_store], outputs=[pytest_display])

        # ── Tab 4: Agent Trace ─────────────────────────────────────────────
        with gr.Tab("🔍 Agent Trace"):
            gr.Markdown("### A2A Message Trace\nRun a generation first, then click Refresh.")

            def show_trace(store):
                return store.get("trace", "No trace available yet.")

            refresh_trace_btn = gr.Button("🔄 Refresh Trace")
            trace_display = gr.Code(language="json", label="Agent Communication Trace")
            refresh_trace_btn.click(fn=show_trace, inputs=[state_store], outputs=[trace_display])

        # ── Tab 5: About ───────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## Architecture (6 Agents)

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
│  RequirementAnalysisAgent    │  GPT-4o: extracts entities/actions/constraints
└────────────┬─────────────────┘
             │ A2A: requirements
             ▼
┌──────────────────────────────┐
│  TestCaseGeneratorAgent      │  GPT-4o: generates N structured test cases
└────────────┬─────────────────┘
             │ A2A: test_cases
             ▼
┌──────────────────────────────┐
│  TestCaseReviewerAgent       │  GPT-4o: reviews quality (score 0–1)
└─────��──────┬─────────────────┘
             │  score < 0.70 AND retries < 2  →  back to Generator
             │  score >= 0.70
             ▼
┌──────────────────────────────┐
│  OutputFormatterAgent        │  DataFrame + Excel bytes
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  TestCodeGeneratorAgent      │  GPT-4o: generates executable pytest code
└────────────┬─────────────────┘
             ▼
     Gradio UI (table · JSON · Excel download · pytest download · trace)
```

## Tech Stack
- **LangGraph** — stateful multi-agent orchestration with conditional retry loop
- **A2A Protocol** — typed message envelopes between agents with full trace
- **GPT-4o** via LangChain — powers 4 of the 6 agents
- **Gradio** — multi-tab web UI
- **pandas + openpyxl** — data I/O and Excel export
- **pytest** — generated test automation scripts
""")

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
