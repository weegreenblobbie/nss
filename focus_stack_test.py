"""Tests of focus_stack.py's command line: a target and frames, or a Lightroom request, never both and never neither."""
import pytest

from focus_stack import parse_args


def test_target_and_frames_with_the_default_mode() -> None:
    """A target and frames, no mode: focus, no Lightroom request."""
    args = parse_args(["target.tif", "one.tif", "two.tif"])
    assert (args.target, args.frames, args.mode, args.lightroom) == ("target.tif", ["one.tif", "two.tif"], "focus", None)


def test_clean_mode_and_work_dim() -> None:
    """--mode clean and --work-dim are read wherever they come."""
    args = parse_args(["--mode", "clean", "target.tif", "one.tif", "--work-dim", "2000"])
    assert (args.mode, args.work_dim, args.frames) == ("clean", 2000, ["one.tif"])


def test_a_lightroom_request_replaces_the_target_and_frames() -> None:
    """--lightroom alone is enough."""
    args = parse_args(["--lightroom", "request.txt"])
    assert args.lightroom == "request.txt" and args.target is None and args.frames == []


@pytest.mark.parametrize("argv", [[], ["target.tif"], ["--lightroom", "request.txt", "target.tif", "one.tif"], ["--mode", "stretch", "a.tif", "b.tif"]])
def test_bad_combinations_are_refused(argv: list[str]) -> None:
    """Nothing, a target without frames, a request together with pictures, and an unknown mode all exit with a usage error."""
    with pytest.raises(SystemExit) as error:
        parse_args(argv)
    assert error.value.code == 2
