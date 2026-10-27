import sys
from pathlib import Path

# Make the Flash-Searcher-main project root importable regardless of the
# directory pytest is invoked from, so `import MemEvolve...` resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
