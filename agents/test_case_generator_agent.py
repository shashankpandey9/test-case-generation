"""
Agent 3 — TestCaseGeneratorAgent
Generates comprehensive test cases (positive, negative, edge, boundary)
in the same column schema as the source Excel file.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from a2a_protocol import A2AMessage, A2AProtocol


SYSTEM_PROMPT = """
You are an expert QA engineer. Generate detailed test cases based on the
provided requirements analysis.

You MUST return a JSON array of test case objects. Each object must use
exactly these column keys (matching the source Excel):
{columns}

If a column is not applicable, use an empty string.

Guidelines:
- Cover positive, negative, edge case, and boundary scenarios as requested.
- Each test case must be unique and unambiguous.
- "Test Steps" should be a numbered list as a single string.
- "Expected Result" must be clear and verifiable.
"""

HUMAN_PROMPT = """
Requirements analysis:
{requirements}

Generate exactly {test_count} test cases.
Focus on test types: {test_types}
"""


class TestCaseGeneratorAgent:
    NAME = "TestCaseGeneratorAgent"

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
            msg = self.protocol.receive(self.NAME)
            payload = msg.payload if msg else {}
            requirements = payload.get("requirements", state.get("requirements", {}))
            data_context = payload.get("data_context", state.get("data_context", {}))
            test_count = payload.get("test_count", state.get("test_count", 10))
            test_types = payload.get("test_types", state.get("test_types", "All"))
            columns = data_context.get("columns", [])

            chain = self.prompt | self.llm
            response = chain.invoke(
                {
                    "columns": ", ".join(columns),
                    "requirements": json.dumps(requirements, indent=2),
                    "test_count": test_count,
                    "test_types": test_types,
                }
            )
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            test_cases = json.loads(raw)
            if not isinstance(test_cases, list):
                test_cases = test_cases.get("test_cases", [])

            self.protocol.send(
                A2AMessage(
                    sender=self.NAME,
                    receiver="TestCaseReviewerAgent",
                    message_type="test_cases",
                    payload={
                        "test_cases": test_cases,
                        "columns": columns,
                        "requirements": requirements,
                    },
                )
            )
            state["test_cases"] = test_cases
            state["error"] = None
        except Exception as exc:
            state["error"] = f"[{self.NAME}] {exc}"
        return state
