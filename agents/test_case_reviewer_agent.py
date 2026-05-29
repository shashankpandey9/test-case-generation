"""
Agent 4 — TestCaseReviewerAgent
Reviews generated test cases for quality, completeness, and coverage.
Assigns a quality_score (0.0–1.0) and flags issues.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from a2a_protocol import A2AMessage, A2AProtocol

QUALITY_THRESHOLD = 0.70

SYSTEM_PROMPT = """
You are a senior QA reviewer. Review the provided test cases for:
- Completeness: all required fields filled
- Clarity: steps and expected results are unambiguous
- Coverage: positive, negative, edge, and boundary cases present
- Duplicates: no two test cases test the same thing

Return ONLY valid JSON with:
{
  "quality_score": <float 0.0-1.0>,
  "overall_feedback": "<string>",
  "reviewed_cases": [
    { ...original fields..., "review_comment": "OK" or "<issue>", "approved": true/false }
  ]
}
"""

HUMAN_PROMPT = """
Test cases to review:
{test_cases}

Original requirements:
{requirements}
"""


class TestCaseReviewerAgent:
    NAME = "TestCaseReviewerAgent"

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
            test_cases = payload.get("test_cases", state.get("test_cases", []))
            requirements = payload.get("requirements", state.get("requirements", {}))
            columns = payload.get("columns", state.get("data_context", {}).get("columns", []))

            chain = self.prompt | self.llm
            response = chain.invoke(
                {
                    "test_cases": json.dumps(test_cases, indent=2),
                    "requirements": json.dumps(requirements, indent=2),
                }
            )
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            review_result = json.loads(raw)

            quality_score = float(review_result.get("quality_score", 0.0))
            reviewed_cases = review_result.get("reviewed_cases", test_cases)

            self.protocol.send(
                A2AMessage(
                    sender=self.NAME,
                    receiver="OutputFormatterAgent",
                    message_type="review_result",
                    payload={
                        "reviewed_cases": reviewed_cases,
                        "quality_score": quality_score,
                        "overall_feedback": review_result.get("overall_feedback", ""),
                        "columns": columns,
                    },
                )
            )
            state["reviewed_cases"] = reviewed_cases
            state["quality_score"] = quality_score
            state["overall_feedback"] = review_result.get("overall_feedback", "")
            state["error"] = None
        except Exception as exc:
            state["error"] = f"[{self.NAME}] {exc}"
        return state
