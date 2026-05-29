"""
LangGraph StateGraph — wires all 6 agents into a pipeline with a
conditional retry loop after the reviewer, plus pytest code generation.
"""

from __future__ import annotations

import uuid
import os
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from a2a_protocol import A2AProtocol
from agents import (
    DataIngestionAgent,
    RequirementAnalysisAgent,
    TestCaseGeneratorAgent,
    TestCaseReviewerAgent,
    OutputFormatterAgent,
    TestCodeGeneratorAgent,
)

load_dotenv()

QUALITY_THRESHOLD = 0.70
MAX_RETRIES = 2
DEFAULT_EXCEL_PATH = os.path.join(os.path.dirname(__file__), "POC_TestCase_Chatbot_Data.xlsx")


class AgentState(TypedDict, total=False):
    user_input: str
    test_count: int
    test_types: str
    data_context: Dict[str, Any]
    requirements: Dict[str, Any]
    test_cases: List[Dict[str, Any]]
    reviewed_cases: List[Dict[str, Any]]
    quality_score: float
    overall_feedback: str
    output: Dict[str, Any]
    pytest_code: str
    trace_id: str
    error: Optional[str]
    retry_count: int
    protocol: Any  # A2AProtocol instance


def build_graph(excel_path: str = DEFAULT_EXCEL_PATH) -> Any:
    """Build and compile the LangGraph pipeline."""

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    def node_data_ingestion(state: AgentState) -> AgentState:
        return DataIngestionAgent(protocol=state["protocol"], excel_path=excel_path).run(state)

    def node_requirement_analysis(state: AgentState) -> AgentState:
        return RequirementAnalysisAgent(protocol=state["protocol"], llm=llm).run(state)

    def node_test_case_generator(state: AgentState) -> AgentState:
        return TestCaseGeneratorAgent(protocol=state["protocol"], llm=llm).run(state)

    def node_test_case_reviewer(state: AgentState) -> AgentState:
        return TestCaseReviewerAgent(protocol=state["protocol"], llm=llm).run(state)

    def node_output_formatter(state: AgentState) -> AgentState:
        return OutputFormatterAgent(protocol=state["protocol"]).run(state)

    def node_test_code_generator(state: AgentState) -> AgentState:
        return TestCodeGeneratorAgent(protocol=state["protocol"], llm=llm).run(state)

    def should_retry(state: AgentState) -> str:
        if state.get("error"):
            return "format"
        score = state.get("quality_score", 0.0)
        retries = state.get("retry_count", 0)
        if score < QUALITY_THRESHOLD and retries < MAX_RETRIES:
            state["retry_count"] = retries + 1
            return "regenerate"
        return "format"

    workflow = StateGraph(AgentState)

    workflow.add_node("ingest", node_data_ingestion)
    workflow.add_node("analyse", node_requirement_analysis)
    workflow.add_node("generate", node_test_case_generator)
    workflow.add_node("review", node_test_case_reviewer)
    workflow.add_node("format", node_output_formatter)
    workflow.add_node("codegen", node_test_code_generator)

    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "analyse")
    workflow.add_edge("analyse", "generate")
    workflow.add_edge("generate", "review")
    workflow.add_conditional_edges(
        "review",
        should_retry,
        {"regenerate": "generate", "format": "format"},
    )
    workflow.add_edge("format", "codegen")
    workflow.add_edge("codegen", END)

    return workflow.compile()


def run_pipeline(
    user_input: str,
    test_count: int = 10,
    test_types: str = "All",
    excel_path: str = DEFAULT_EXCEL_PATH,
) -> AgentState:
    """Run the full pipeline and return the final state."""
    trace_id = str(uuid.uuid4())
    protocol = A2AProtocol(trace_id=trace_id)

    initial_state: AgentState = {
        "user_input": user_input,
        "test_count": test_count,
        "test_types": test_types,
        "trace_id": trace_id,
        "protocol": protocol,
        "retry_count": 0,
        "error": None,
    }

    graph = build_graph(excel_path=excel_path)
    final_state = graph.invoke(initial_state)
    final_state["a2a_trace"] = protocol.get_trace()
    return final_state
