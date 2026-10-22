"""Makes src/ importable from test files without each one repeating sys.path setup."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
