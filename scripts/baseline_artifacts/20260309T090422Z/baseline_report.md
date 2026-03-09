# Baseline Report

- run_id: `20260309T090422Z`
- command: `C:\work\repo4-Realtime-faceghosting\.venv\Scripts\python.exe -m facefx.main --camera 1 --device cuda --scale 1.0 --topology frozen --region all --color-match-every 1 --shading on --profile`
- avg_fps: `3.49`
- p95_frame_ms: `342.88`
- bottlenecks:
  - `warp`: `208.78 ms`
  - `mask`: `19.24 ms`
  - `landmarks`: `16.72 ms`

Screenshots:
- `scripts\baseline_artifacts\20260309T090422Z\screenshot_01_frame_0060.png`
- `scripts\baseline_artifacts\20260309T090422Z\screenshot_02_frame_0120.png`
- `scripts\baseline_artifacts\20260309T090422Z\screenshot_03_frame_0180.png`
