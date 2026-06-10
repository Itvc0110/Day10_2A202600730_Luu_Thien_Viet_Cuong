from __future__ import annotations

import sys
from pathlib import Path

# Thêm thư mục src vào Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipelines.phase1 import main

if __name__ == "__main__":
    main()