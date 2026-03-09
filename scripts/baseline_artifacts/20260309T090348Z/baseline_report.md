# Baseline Report

- run_id: `20260309T090348Z`
- command: `C:\work\repo4-Realtime-faceghosting\.venv\Scripts\python.exe -m facefx.main --camera 0 --device cuda --scale 1.0 --topology frozen --region all --color-match-every 1 --shading on --profile`
- avg_fps: `21.47`
- p95_frame_ms: `66.88`
- bottlenecks:
  - `capture`: `20.42 ms`
  - `display`: `14.00 ms`
  - `landmarks`: `10.54 ms`

Screenshots:
- `scripts\baseline_artifacts\20260309T090348Z\screenshot_01_frame_0060.png`
- `scripts\baseline_artifacts\20260309T090348Z\screenshot_02_frame_0120.png`
- `scripts\baseline_artifacts\20260309T090348Z\screenshot_03_frame_0180.png`
