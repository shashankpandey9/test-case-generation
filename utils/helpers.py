"""
Shared utility helpers.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List

import pandas as pd


def format_test_cases_as_df(test_cases: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of test-case dicts to a tidy DataFrame."""
    if not test_cases:
        return pd.DataFrame()
    return pd.DataFrame(test_cases).fillna("")


def export_to_excel_bytes(test_cases: List[Dict[str, Any]]) -> bytes:
    """
    Serialise test cases to an in-memory .xlsx file and return raw bytes
    suitable for a Gradio File download component.
    """
    df = format_test_cases_as_df(test_cases)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Generated Test Cases")
    buf.seek(0)
    return buf.read()
