"""Make the project's flat source modules available to unittest discovery."""
import sys
from pathlib import Path

SOURCE = str(Path(__file__).resolve().parents[1] / 'src')
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)
