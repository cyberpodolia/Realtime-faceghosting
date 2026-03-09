# Runtime CUDA v1 Contract

Status: frozen for initial implementation
Date: 2026-03-09
Scope root: `facefx/runtime_cuda/`

## Purpose

Define the runtime-only contract for the new CUDA-oriented pipeline so implementation can proceed in small work items without changing decisions midstream.

## Runtime Scope

The runtime path is a separate package rooted at `facefx/runtime_cuda/`.
It is focused on live face replacement execution only and must not depend on legacy mesh internals in `facefx/src/`.

## Fixed v1 Decisions

These decisions are locked for v1 and are not optional toggles:

- Backend: OpenCV CUDA runtime path (`cv2.cuda`) for remap/composite stages.
- ROI strategy: adaptive face ROI derived from live landmarks.
- Landmark cadence: MediaPipe landmark solve every 3 frames, with hold/smoothing between updates.
- IO mode: single camera input and single output stream.
- Effect locality: processing remains confined to adaptive ROI (not full-frame warp).

## Performance Targets

Test target:

- Resolution: `1280x720`
- GPU: `GTX 1660 Ti`
- Required minimum: `20 FPS`
- Desired target: `30 FPS`

All performance claims must include measured average FPS and p95 frame time from profiling runs.

## Visual Review Requirements

Manual visual review is mandatory for runtime changes.

- Capture representative screenshots from runtime output.
- Verify no obvious face boundary tearing, eye/mouth overpaint, or severe color mismatch.
- Keep screenshots as review evidence in work item artifacts.

## Required v1 Features

The following must be enabled in v1 runtime behavior:

- Color match: ON
- Shading: ON
- Legacy pipeline path: preserved and runnable

If performance regresses, report it with evidence. Do not silently disable required features.

## Invariants

- Do not modify default legacy behavior in `facefx/main.py` while building runtime_cuda internals.
- Do not remove the legacy mesh-based path.
- Keep new runtime code isolated under `facefx/runtime_cuda/` and runtime-focused tests/docs.
- No network-dependent tooling or downloads are required for this contract.

## Out of Scope for This Contract

- Rewriting legacy `facefx/src/*` modules.
- Multi-camera orchestration.
- Multiple output streams or distributed runtime architecture.
- Dynamic policy switching that changes the locked v1 defaults above.

