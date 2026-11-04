"""Run the full WattWise pipeline from the project root without installing the package.

    python run_pipeline.py                 # all stages
    python run_pipeline.py --stage raw     # Task 1 only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from wattwise.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
