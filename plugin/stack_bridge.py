"""
Hand-over files between the Lightroom Classic plug-in's "Stack" menu items (nss.lrdevplugin/Stack.lua) and `python focus_stack.py --lightroom`.

The plug-in exports the selected photos as 16-bit Adobe RGB TIFFs and writes a request file of `key=value` lines:

    mode=clean|clean2|focus  what to stack for (nss/focus_stack/README.md)
    target=<tiff>         the first photo: the picture that is corrected
    count=N               how many frames follow, then frame.1=<tiff> ... frame.N=<tiff>
    output=<path>         full path of the TIFF to write when the result is saved with the target as it was sent (photo 0), then
                          output.1=<path> ... output.N=<path> for when the user makes frame i the target in the window: the result is
                          written next to, and stacked above, whichever photo ended up as the target
    result=<path>         path of the result file
    work_dim=<pixels>     optional: long side of the preview you paint on (default DEFAULT_WORK_DIM)

focus_stack.py answers in the result file and exits with EXIT_CODES[status]:

    status=saved       output=<path>        the corrected picture was written to output at full size; the plug-in imports it
                       target=<i>           which photo was the target: 0 the one sent as target=, i the frame i (so output is output.i)
    status=cancelled                        the window was closed or Cancel pressed without saving; no output was written
    status=error       message=<one line>   the frames could not be aligned or the picture could not be written
"""
from dataclasses import dataclass

from plugin.handover import read_values

SAVED, CANCELLED, ERROR = "saved", "cancelled", "error"
EXIT_CODES = {SAVED: 0, CANCELLED: 1, ERROR: 2}
MODES = ("clean", "clean2", "focus")
DEFAULT_WORK_DIM = 3000  # the working size of nss.focus_stack.stack.REFERENCE_DIM, repeated so the plug-in side needs no nss imports


@dataclass
class Request:
    """A request file's contents."""
    mode: str
    target: str
    frames: list[str]
    output: str
    result: str
    work_dim: int
    outputs: list[str]  # output for photo 0 (the target as sent), then one for each frame


def whole_number(values: dict[str, str], key: str, default: int | None = None) -> int:
    """The positive whole number in values[key]; `default` when the key is absent and a default is given; ValueError otherwise."""
    if key not in values and default is not None:
        return default
    try:
        number = int(values[key])
    except (KeyError, ValueError):
        raise ValueError(f"request needs {key}= as a whole number, got {values.get(key)!r}") from None
    if number < 1:
        raise ValueError(f"request needs {key}= of at least 1, got {number}")
    return number


def read_request(path: str) -> Request:
    """Parses a request file; ValueError naming what is missing or wrong."""
    v = read_values(path)
    missing = [key for key in ("mode", "target", "count", "output", "result") if not v.get(key)]
    if missing:
        raise ValueError(f"{path}: request needs {', '.join(key + '=' for key in missing)}")
    if v["mode"] not in MODES:
        raise ValueError(f"{path}: mode= must be one of {', '.join(MODES)}, got {v['mode']!r}")
    count = whole_number(v, "count")
    absent = [f"frame.{i}" for i in range(1, count + 1) if not v.get(f"frame.{i}")]
    if absent:
        raise ValueError(f"{path}: request needs {', '.join(key + '=' for key in absent)}")
    absent = [f"output.{i}" for i in range(1, count + 1) if not v.get(f"output.{i}")]
    if absent:
        raise ValueError(f"{path}: request needs {', '.join(key + '=' for key in absent)}")
    return Request(
        v["mode"], v["target"], [v[f"frame.{i}"] for i in range(1, count + 1)], v["output"], v["result"],
        whole_number(v, "work_dim", DEFAULT_WORK_DIM), [v["output"]] + [v[f"output.{i}"] for i in range(1, count + 1)],
    )


def result_lines(status: str, output: str = "", message: str = "", target: int = 0) -> list[str]:
    """The result file's lines: status=, then output= and target= for saved and message= (made one line) for error."""
    if status not in EXIT_CODES:
        raise ValueError(f"unknown status {status!r}")
    lines = [f"status={status}"]
    if status == SAVED:
        lines += [f"output={output}", f"target={target}"]
    if status == ERROR:
        lines.append(f"message={' '.join(message.split())}")
    return lines


def write_result(path: str, status: str, output: str = "", message: str = "", target: int = 0) -> None:
    """Writes result_lines(status, output, message, target) to path."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(result_lines(status, output, message, target)) + "\n")
