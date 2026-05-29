"""
Gradio UI — multi-tab interface for the test-case generation pipeline.
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

def _save_excel_tmp(excel_bytes: bytes) -> str:
    """Write bytes to a named tmp file and return the path (for gr.File)."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    tmp.write(excel_bytes)
    tmp.flush()
    return tmp.name


# ---------------------------------------------------------------------------
# Core generation callback
# ---------------------------------------------------------------------------

def generate_test_cases(feature_description, test_count, test_types, state_store):
    if not feature_description.strip():
        return "⚠️ Please enter a feature description.", pd.DataFrame(), "", None, "", state_store

    try:
        final_state = run_pipeline(
            user_input=feature_description,
            test_count=int(test_count),
            test_types=test_types,
            excel_path=EXCEL_PATH,
        )
    except Exception as exc:
        return f"❌ Pipeline error: {exc}", pd.DataFrame(), "", None, "", state_store

    if final_state.get("error"):
        return f"❌ {final_state['error']}", pd.DataFrame(), "", None, "", state_store

    output = final_state.get("output", {})
    df = output.get("dataframe", pd.DataFrame())
    excel_bytes = output.get("excel_bytes", b"")
    quality_score = output.get("quality_score", 0.0)
    overall_feedback = output.get("overall_feedback", "")
    trace = final_state.get("a2a_trace", [])

    download_path = _save_excel_tmp(excel_bytes) if excel_bytes else None
    json_text = json.dumps(output.get("test_cases", []), indent=2)
    trace_text = json.dumps(trace, indent=2)

    status_md = (
        f"✅ **{len(df)} test cases generated** — "
        f"Quality score: `{quality_score:.2f}` — {overall_feedback}"
    )

    state_store = {"df": df, "trace": trace_text, "json": json_text}
    return status_md, df, json_text, download_path, trace_text, state_store


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="🧪 AI Test Case Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🧪 AI Test Case Generator\n"
            "Multi-agent pipeline powered by **LangGraph + A2A Protocol + GPT-4o**"
        )

        state_store = gr.State({})

        # Tab 1 — Generate
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

            download_btn = gr.File(label="⬇️ Download as Excel (.xlsx)", interactive=False)
            trace_hidden = gr.Textbox(visible=False)

            generate_btn.click(
                fn=generate_test_cases,
                inputs=[feature_input, test_count, test_types, state_store],
                outputs=[status_md, result_df, json_output, download_btn, trace_hidden, state_store],
            )

        # Tab 2 — Data Preview
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

        # Tab 3 — Agent Trace
        with gr.Tab("🔍 Agent Trace"):
            gr.Markdown("### A2A Message Trace\nRun a generation first, then click Refresh.")

            def show_trace(store):
                return store.get("trace", "No trace available yet.")

            refresh_btn = gr.Button("🔄 Refresh Trace")
            trace_display = gr.Code(language="json", label="Agent Communication Trace")
            refresh_btn.click(fn=show_trace, inputs=[state_store], outputs=[trace_display])

        # Tab 4 — About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
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
│  RequirementAnalysisAgent    │  GPT-4o: extracts entities, actions, constraints
└────────────┬─────────────────┘
             │ A2A: requirements
             ▼
┌──────────────────────────────┐
│  TestCaseGeneratorAgent      │  GPT-4o: generates N test cases
└────────────┬─────────────────┘
             │ A2A: test_cases
             ▼
┌──────────────────────────────┐
│  TestCaseReviewerAgent       │  GPT-4o: reviews & scores quality
└────────────┬─────────────────┘
             │ quality_score >= 0.70? No → retry (max 2x) → Generator
             │ Yes
             ▼
┌──────────────────────────────┐
│  OutputFormatterAgent        │  Builds DataFrame + Excel bytes
└────────────┬─────────────────┘
             ▼
     Gradio UI (results · download · trace)
```

## Tech Stack
- **LangGraph** — stateful agent orchestration with conditional retry loop
- **A2A Protocol** — typed message envelopes between agents with full trace
- **GPT-4o** via LangChain — powers RequirementAnalysis, Generator, Reviewer
- **Gradio** — multi-tab web UI
- **pandas + openpyxl** — data I/O and Excel export
""")

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
