"""Convertor: transform optimizer_logic results.json into Python source files + impact report."""

from .json_to_python import write_python_files
from .report import generate_report

__all__ = ["write_python_files", "generate_report"]
