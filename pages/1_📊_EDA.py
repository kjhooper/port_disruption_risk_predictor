"""
Port EDA page — delegates to eda.py in the repo root.
Streamlit multi-page shim: keeps a single source of truth.
"""
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

exec((_root / "eda.py").read_text())  # noqa: S102
