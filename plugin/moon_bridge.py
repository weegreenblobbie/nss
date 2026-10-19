"""
Hand-over files between the Lightroom Classic plug-in's moon menu items (nss.lrdevplugin/Moon.lua) and moon.py.  Two kinds of request,
`key=value` lines, i = 1 .. count:

mode=measure: find the moon in exported 16-bit TIFFs (center moon measures one chunk of photos per request, verify all of them)

    result=<path of the result file>    size=<detect_moon grid size, pixels>    count=N
    min_confidence=<0..1>               a circle with a smaller share of the edge points on it is notfound (default DEFAULT_MIN_CONFIDENCE)
    progress=<path>                     optional: rewritten with "<done> <count>" after each photo
    cancel=<path>                       optional: once this file exists no further photo is started (those left: status canceled)
    workers=<n>                         processes measuring photos at the same time (default 1); about 1 GB of memory each
    photo.i.id, photo.i.image           the photo's local id and its exported 16-bit TIFF
    photo.i.left, .top, .right, .bottom the photo's crop fractions as Lightroom stores them (CropLeft ...; 0 0 1 1 for verify)
    photo.i.orientation                 photo:getRawMetadata('orientation'): AB (as shot), BC, CD, DA and the mirrors BA, CB, DC, AD
    photo.i.angle                       the crop angle (CropAngle, degrees; default 0)

  result, per photo i:
    photo.i.status=ok|notfound|error|canceled    photo.i.message=<one line>    photo.i.confidence (when a circle was fitted)
    photo.i.cx, .cy, .radius            the moon in image pixels (x right, y down, pixel centres at index + 0.5)
    photo.i.width, .height              the image's size in pixels
    photo.i.crop_width, .crop_height    the size in image pixels of the request's crop
    photo.i.dx, .dy                     moon centre minus the centre of the request's crop, image pixels

mode=single|batch: the new crops from measurements (no images; batch: one common pixel size for all the photos)

    result=<path of the result file>    count=N
    photo.i.id, .cx, .cy, .width, .height, .crop_width, .crop_height, .orientation, .angle    as a measure result gave them

  result, per photo i:
    photo.i.status=ok    photo.i.left, .top, .right, .bottom    the new crop, in Lightroom's crop fractions (with an angle, the turned
                                                                crop's top-left and bottom-right corners, as Lightroom stores them)
"""
from dataclasses import dataclass, field

MODES = ("measure", "single", "batch")
DEFAULT_MIN_CONFIDENCE = 0.9      # correct fits on orig-moon.tif scored 0.958 .. 1.0 over grid sizes 61 .. 961
RESULT_KEYS = ("status", "confidence", "cx", "cy", "radius", "width", "height", "crop_width", "crop_height", "dx", "dy",
               "left", "top", "right", "bottom", "message")
MEASURED_KEYS = ("cx", "cy", "width", "height", "crop_width", "crop_height", "angle")     # numbers a single / batch request carries per photo


@dataclass
class Photo:
    """One photo of a measure request."""
    id: str
    image: str
    crop: tuple[float, float, float, float]      # left, top, right, bottom as Lightroom stores them
    orientation: str
    angle: float = 0.0                           # CropAngle, degrees


@dataclass
class Request:
    """A request file's contents: photos for measure, measured (one dict per photo, MEASURED_KEYS + id, orientation) for single / batch."""
    result: str
    mode: str
    size: int = 121
    photos: list[Photo] = field(default_factory=list)
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    progress: str = ""
    cancel: str = ""
    workers: int = 1
    measured: list[dict] = field(default_factory=list)


def _read_lines(path: str) -> dict[str, str]:
    """The key=value lines of a file (the value may contain '=')."""
    values = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    return values


def read_request(path: str) -> Request:
    """Parses a request file; ValueError when result=, a valid mode= or a photo's required keys are missing."""
    v = _read_lines(path)
    if "result" not in v or v.get("mode") not in MODES:
        raise ValueError(f"{path}: request needs result= and mode= one of {MODES}")
    request = Request(v["result"], v["mode"], int(v.get("size", "121")), min_confidence=float(v.get("min_confidence", DEFAULT_MIN_CONFIDENCE)),
                      progress=v.get("progress", ""), cancel=v.get("cancel", ""), workers=int(v.get("workers", "1")))
    for i in range(1, int(v.get("count", "0")) + 1):
        p = f"photo.{i}."
        if request.mode == "measure":
            if p + "id" not in v or p + "image" not in v:
                raise ValueError(f"{path}: photo {i} needs id= and image=")
            crop = tuple(float(v.get(p + side, default)) for side, default in (("left", 0), ("top", 0), ("right", 1), ("bottom", 1)))
            request.photos.append(Photo(v[p + "id"], v[p + "image"], crop, v.get(p + "orientation", "AB"), float(v.get(p + "angle", 0.0))))
        else:
            missing = [key for key in ("id",) + MEASURED_KEYS if p + key not in v]
            if missing:
                raise ValueError(f"{path}: photo {i} needs {', '.join(missing)}")
            measured = {key: float(v[p + key]) for key in MEASURED_KEYS}
            measured.update(id=v[p + "id"], orientation=v.get(p + "orientation", "AB"), status="ok")
            request.measured.append(measured)
    return request


def result_lines(results: list[dict]) -> list[str]:
    """photo.i.<key>=value lines of the results (i = 1-based position); keys missing from a result are left out, floats get 6 decimals."""
    out = []
    for i, result in enumerate(results, start=1):
        for key in RESULT_KEYS:
            if key in result:
                value = result[key]
                text = f"{value:.6f}" if isinstance(value, float) else " ".join(str(value).split())       # one line
                out.append(f"photo.{i}.{key}={text}")
    return out


def write_result(path: str, results: list[dict]) -> None:
    """Writes the result file; count= is the number of results."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join([f"count={len(results)}"] + result_lines(results)) + "\n")
