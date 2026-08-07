"""
Configuration for the RAG pipeline.
All settings/paths in one place — nothing hardcoded elsewhere in the code.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "data.pdf"