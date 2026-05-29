"""
Agent 1 — DataIngestionAgent
Loads and parses the Excel data source, then broadcasts the schema
and domain context via A2A to RequirementAnalysisAgent.
"""

from __future__ import annotations

from typing import Any, Dict

from a2a_protocol import A2AMessage, A2AProtocol
from data_loader import load_excel_data


class DataIngestionAgent:
    NAME = "DataIngestionAgent"

    def __init__(self, protocol: A2AProtocol, excel_path: str):
        self.protocol = protocol
        self.excel_path = excel_path

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data = load_excel_data(self.excel_path)
            data_context = {
                "columns": data["columns"],
                "sample_rows": data["sample_rows"],
                "total_rows": data["total_rows"],
                "sheet_names": data["sheet_names"],
                "domain_summary": data["domain_summary"],
            }
            self.protocol.send(
                A2AMessage(
                    sender=self.NAME,
                    receiver="RequirementAnalysisAgent",
                    message_type="data_context",
                    payload=data_context,
                )
            )
            state["data_context"] = data_context
            state["error"] = None
        except Exception as exc:
            state["error"] = f"[{self.NAME}] {exc}"
        return state
