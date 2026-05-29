"""
Agent 2 — RequirementAnalysisAgent
Uses GPT-4o to analyse the user's input and extract structured
requirements, then sends them to TestCaseGeneratorAgent via A2A.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from a2a_protocol import A2AMessage, A2AProtocol


SYSTEM_PROMPT = """
You are a senior QA analyst. Your task is to analyse a feature description and
extract structured requirements for test case generation.

Data source context:
{domain_summary}

Existing columns in the test-case sheet: {columns}

Return ONLY valid JSON with the following keys:
- "feature_name": short name of the feature
- "entities": list of key entities/objects involved
- "user_actions": list of actions a user can perform
- "expected_behaviors": list of expected outcomes
- "constraints": list of business rules or constraints
- "test_types_needed": list from [positive, negative, edge_case, boundary]
"""

HUMAN_PROMPT = """
Feature description: {user_input}
Requested test types: {test_types}
Number of test cases requested: {test_count}
"""


class RequirementAnalysisAgent:
    NAME = "RequirementAnalysisAgent"

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
            data_context = msg.payload if msg else state.get("data_context", {})

            chain = self.prompt | self.llm
            response = chain.invoke(
                {
                    "domain_summary": data_context.get("domain_summary", ""),
                    "columns": ", ".join(data_context.get("columns", [])),
                    "user_input": state["user_input"],
                    "test_types": state.get("test_types", "All"),
                    "test_count": state.get("test_count", 10),
                }
            )
            raw = response.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            requirements = json.loads(raw)

            self.protocol.send(
                A2AMessage(
                    sender=self.NAME,
                    receiver="TestCaseGeneratorAgent",
                    message_type="requirements",
                    payload={
                        "requirements": requirements,
                        "data_context": data_context,
                        "test_count": state.get("test_count", 10),
                        "test_types": state.get("test_types", "All"),
                    },
                )
            )
            state["requirements"] = requirements
            state["error"] = None
        except Exception as exc:
            state["error"] = f"[{self.NAME}] {exc}"
        return state
