"""Ensure the backend package root is importable when running pytest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
