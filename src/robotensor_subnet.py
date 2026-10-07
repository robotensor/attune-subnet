"""`robotensor_subnet` is `robotensor`.

The package took the name `robotensor` when it became something a miner installs from an index
(`pip install robotensor`). A script that still imports the old name keeps working and is told
once; the name goes when the first release after 0.2 does.
"""

from __future__ import annotations

import sys
import warnings

import robotensor

warnings.warn(
    "robotensor_subnet is now robotensor: import robotensor instead",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = robotensor
