# FaceFX (MVP scaffold)

Windows Python FaceMesh patch-warp MVP (scaffold + core helpers).

## Current status

- Patch images are scanned once at startup. Only images with a detectable face are used.
- If a patch has no detected face, it is skipped; if none qualify, a noise fallback is used.
- Runtime warping uses patch landmarks -> live face landmarks (not bbox resize).

## Architecture

```
facefx/
  main.py                 # entry point (pipeline wiring)
  requirements.txt        # runtime deps (mediapipe pinned)
  requirements-dev.txt    # gates (ruff/pytest)
  pyproject.toml          # ruff config
  patches/                # patch PNG/JPG assets (user-provided)
    .gitkeep
  src/
    __init__.py           # exports
    facemesh.py           # MediaPipe FaceMesh wrapper (live + static)
    regions.py            # region masks
    triangulation.py      # Delaunay triangulation
    warp.py               # piecewise affine warp
    blend.py              # feather blend/composite
    patchbank.py          # patch loader + landmark cache
    ui.py                 # minimal UI helpers
  tests/
    test_imports.py        # import smoke test
```

## Requirements

Install runtime deps (note mediapipe pin):

```
python -m pip install -r facefx/requirements.txt
```

Dev deps (gates):

```
python -m pip install -r facefx/requirements-dev.txt
```

If you see `AttributeError: module 'mediapipe' has no attribute 'solutions'`,
reinstall with the pinned version in `facefx/requirements.txt`.

## Run

Preferred (from repo root):

```
python -m facefx.main
```

Also works (from inside `facefx/`):

```
python .\main.py
```

Second camera:

```
python -m facefx.main --camera 1
```

OBS: capture the app window.

## Notes

- Put patch PNG/JPG files into `facefx/patches/`.
- Use frontal faces for best alignment.
- If no patch has a detectable face, a procedural noise patch is used.
