"""
Data Loader — reads POC_TestCase_Chatbot_Data.xlsx and returns a
structured summary for use by the pipeline agents.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd


DEFAULT_EXCEL_PATH = os.path.join(os.path.dirname(__file__), "POC_TestCase_Chatbot_Data.xlsx")


def load_excel_data(filepath: str = DEFAULT_EXCEL_PATH) -> Dict[str, Any]:
    """
    Load the Excel data source and return a summary dict.

    Returns
    -------
    {
        "columns":       list[str],
        "sample_rows":   list[dict],   # first 5 rows as dicts
        "total_rows":    int,
        "sheet_names":   list[str],
        "domain_summary": str,
        "dataframe":     pd.DataFrame
    }
    """
    xl = pd.ExcelFile(filepath)
    sheet_names: List[str] = xl.sheet_names

    df: pd.DataFrame = xl.parse(sheet_names[0])
    df.columns = [str(c).strip() for c in df.columns]

    columns: List[str] = df.columns.tolist()
    sample_rows: List[dict] = df.head(5).fillna("").to_dict(orient="records")
    total_rows: int = len(df)
    domain_summary: str = (
        f"The test-case spreadsheet has {total_rows} rows and the following columns: "
        + ", ".join(columns)
        + "."
    )

    return {
        "columns": columns,
        "sample_rows": sample_rows,
        "total_rows": total_rows,
        "sheet_names": sheet_names,
        "domain_summary": domain_summary,
        "dataframe": df,
    }


def get_preview_dataframe(filepath: str = DEFAULT_EXCEL_PATH, n: int = 20) -> pd.DataFrame:
    """Return first *n* rows for display in the Gradio Data Preview tab."""
    data = load_excel_data(filepath)
    return data["dataframe"].head(n).fillna("")
