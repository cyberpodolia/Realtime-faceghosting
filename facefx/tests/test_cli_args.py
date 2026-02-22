"""CLI parser coverage for performance flags."""

from facefx.main import _build_parser


def test_cli_defaults():
    args = _build_parser().parse_args([])
    assert args.scale == 1.0
    assert args.refine_landmarks == "on"
    assert args.topology == "frozen"
    assert args.region == "all"
    assert args.color_match_every == 1
    assert args.shading == "on"
    assert args.profile is False
    assert args.dry_run is False


def test_cli_perf_flags_parse():
    args = _build_parser().parse_args(
        [
            "--profile",
            "--dry-run",
            "--scale",
            "0.5",
            "--refine-landmarks",
            "off",
            "--topology",
            "mediapipe",
            "--region",
            "eyes",
            "--color-match-every",
            "3",
            "--shading",
            "off",
        ]
    )
    assert args.profile is True
    assert args.dry_run is True
    assert args.scale == 0.5
    assert args.refine_landmarks == "off"
    assert args.topology == "mediapipe"
    assert args.region == "eyes"
    assert args.color_match_every == 3
    assert args.shading == "off"
