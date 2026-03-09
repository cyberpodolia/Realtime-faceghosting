"""Quick smoke check for runtime_cuda native IDW backend."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    from facefx.runtime_cuda.roi import Roi
    from facefx.runtime_cuda.warp import build_dense_remap_idw

    roi = Roi(x=20, y=12, w=96, h=72)
    dst = np.array(
        [
            [30.0, 20.0],
            [100.0, 20.0],
            [65.0, 48.0],
            [40.0, 76.0],
            [90.0, 76.0],
        ],
        dtype=np.float32,
    )
    src = dst + np.array([2.5, -1.0], dtype=np.float32)
    _map_x, _map_y, stats = build_dense_remap_idw(src, dst, roi)
    payload = {
        "builder_backend": stats.builder_backend,
        "build_ms": stats.build_ms,
        "width": stats.width,
        "height": stats.height,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
