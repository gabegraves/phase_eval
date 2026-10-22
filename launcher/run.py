"""The launcher environment's entry: put the checkout's src first on the path, then launch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phaseeval.launcher import main  # noqa: E402

sys.exit(main())
