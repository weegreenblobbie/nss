"""
Hand-over files between the Lightroom Classic plug-in's "ICM: Intentional Camera Movement" (nss.lrdevplugin/ICM.lua) and `python -m nss.icm --lightroom`.

The plug-in writes a request file of `key=value` lines:

    tiff=<the exported picture to open>    output=<full path of the TIFF to write>    result=<path of the result file>

nss.icm answers in the result file and exits with EXIT_CODES[status]:

    status=saved       output=<path>        the ICM image was written to output; the plug-in imports it
    status=cancelled                        the window was closed without Save to Lightroom; no output was written
    status=error       message=<one line>   rendering or saving failed (the window was then closed)
"""
from dataclasses import dataclass

from plugin.handover import read_values

SAVED, CANCELLED, ERROR = "saved", "cancelled", "error"
EXIT_CODES = {SAVED: 0, CANCELLED: 1, ERROR: 2}


@dataclass
class Request:
    """A request file's contents."""
    tiff: str
    output: str
    result: str


def read_request(path: str) -> Request:
    """Parses a request file; ValueError when tiff=, output= or result= is missing."""
    v = read_values(path)
    missing = [key for key in ("tiff", "output", "result") if not v.get(key)]
    if missing:
        raise ValueError(f"{path}: request needs {', '.join(key + '=' for key in missing)}")
    return Request(v["tiff"], v["output"], v["result"])


def result_lines(status: str, output: str = "", message: str = "") -> list[str]:
    """The result file's lines: status=, then output= for saved and message= (made one line) for error."""
    if status not in EXIT_CODES:
        raise ValueError(f"unknown status {status!r}")
    lines = [f"status={status}"]
    if status == SAVED:
        lines.append(f"output={output}")
    if status == ERROR:
        lines.append(f"message={' '.join(message.split())}")
    return lines


def write_result(path: str, status: str, output: str = "", message: str = "") -> None:
    """Writes result_lines(status, output, message) to path."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(result_lines(status, output, message)) + "\n")
