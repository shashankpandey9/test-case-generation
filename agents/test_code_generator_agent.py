"""
Agent 6 — TestCodeGeneratorAgent
Takes reviewed test cases and generates executable pytest code.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from a2a_protocol import A2AMessage, A2AProtocol


SYSTEM_PROMPT = """
You are an expert Python test automation engineer.
Given a list of structured QA test cases, generate a complete, executable pytest file.

Rules:
- Each test case becomes one pytest function.
- Use the test case ID and scenario as the function name (snake_case).
- Add a docstring with: Test Case ID, scenario, type, priority.
- Use clear Arrange / Act / Assert comments inside each test.
- Add placeholder logic with TODO comments where real implementation is needed.
- Group positive, negative, and edge case tests using pytest marks:
  @pytest.mark.positive, @pytest.mark.negative, @pytest.mark.edge_case
- Add a module-level docstring describing the feature being tested.
- Import pytest at the top. Do NOT import any application code (use TODOs).
- Return ONLY the raw Python code. No markdown fences.
"""

HUMAN_PROMPT = """
Feature: {feature_name}

Test cases:
{test_cases}
"""


class TestCodeGeneratorAgent:
    NAME = "TestCodeGeneratorAgent"

    def __init__(self, protocol: A2AProtocol, llm: ChatOpenAI):
        self.protocol = protocol
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("error"):
            return state
        try:
            reviewed_cases = state.get("reviewed_cases", [])
            requirements = state.get("requirements", {})
            feature_name = requirements.get("feature_name", "Feature Under Test")

            # Format test cases as readable text for the prompt
            tc_lines = []
            for i, tc in enumerate(reviewed_cases, 1):
                tc_lines.append(f"Test Case {i}:")
                for k, v in tc.items():
                    if k not in ("review_comment", "approved"):
                        tc_lines.append(f"  {k}: {v}")
                tc_lines.append("")
            tc_text = "\n".join(tc_lines)

            chain = self.prompt | self.llm
            response = chain.invoke(
                {
                    "feature_name": feature_name,
                    "test_cases": tc_text,
                }
            )
            pytest_code = response.content.strip()
            # Strip accidental markdown fences
            if pytest_code.startswith("```"):
                pytest_code = pytest_code.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            self.protocol.send(
                A2AMessage(
                    sender=self.NAME,
                    receiver="UI",
                    message_type="pytest_code",
                    payload={"pytest_code": pytest_code, "feature_name": feature_name},
                )
            )
            state["pytest_code"] = pytest_code
            state["error"] = None
        except Exception as exc:
            state["error"] = f"[{self.NAME}] {exc}"
        return state
