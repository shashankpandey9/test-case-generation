"""
Agent 5 — OutputFormatterAgent
Formats reviewed test cases into DataFrame, JSON, and Excel bytes.
"""

from __future__ import annotations

from typing import Any, Dict

from a2a_protocol import A2AMessage, A2AProtocol
from utils.helpers import export_to_excel_bytes, format_test_cases_as_df


class OutputFormatterAgent:
    NAME = "OutputFormatterAgent"

    def __init__(self, protocol: A2AProtocol):
        self.protocol = protocol

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("error"):
            return state
        try:
            msg = self.protocol.receive(self.NAME)
            payload = msg.payload if msg else {}
            reviewed_cases = payload.get("reviewed_cases", state.get("reviewed_cases", []))

            df = format_test_cases_as_df(reviewed_cases)
            excel_bytes = export_to_excel_bytes(reviewed_cases)

            output = {
                "dataframe": df,
                "excel_bytes": excel_bytes,
                "test_cases": reviewed_cases,
                "total_generated": len(reviewed_cases),
                "quality_score": state.get("quality_score", 0.0),
                "overall_feedback": state.get("overall_feedback", ""),
            }

            self.protocol.send(
                A2AMessage(
                    sender=self.NAME,
                    receiver="UI",
                    message_type="output",
                    payload={
                        "total_generated": output["total_generated"],
                        "quality_score": output["quality_score"],
                        "overall_feedback": output["overall_feedback"],
                    },
                )
            )
            state["output"] = output
            state["error"] = None
        except Exception as exc:
            state["error"] = f"[{self.NAME}] {exc}"
        return state
