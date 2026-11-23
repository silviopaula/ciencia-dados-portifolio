"""
Global and session state management for App Forecasts Python Backend.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd


class AppState:
    def __init__(self) -> None:
        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.source_filename: Optional[str] = None
        self.detected_type: Optional[str] = None
        self.official_date_col: Optional[str] = None
        self.date_processing_complete: bool = False
        self.operation_logs: List[str] = []
        self.seasonality_report: str = ""

        # Cached modeling results
        self.caret_results: Optional[Dict[str, Any]] = None
        self.arima_results: Optional[Dict[str, Any]] = None
        self.autots_results: Optional[Dict[str, Any]] = None
        self.autots_growth_results: Optional[Dict[str, Any]] = None
        self.custom_models_config: Optional[Dict[str, Dict[str, Any]]] = None

    def reset_import_state(self) -> None:
        self.raw_df = None
        self.processed_df = None
        self.source_filename = None
        self.detected_type = None
        self.official_date_col = None
        self.date_processing_complete = False
        self.operation_logs = []
        self.seasonality_report = ""

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.operation_logs.append(f"{ts} - {message}")

    def get_active_df(self) -> Optional[pd.DataFrame]:
        if self.processed_df is not None:
            return self.processed_df
        return self.raw_df


STATE = AppState()
