"""Minimal ICC matrix/TRC profile support: describe a profile and convert RGB -> sRGB from the
embedded profile itself (Adobe RGB, ProPhoto, sRGB ...). No assumed gammas or matrices.

Only matrix/TRC (rXYZ/gXYZ/bXYZ + rTRC/gTRC/bTRC) profiles are supported - which covers everything
Lightroom exports for sRGB / Adobe RGB / ProPhoto. Values are NOT clipped unless asked, so
out-of-gamut colors come back as <0 or >1 instead of being destroyed.
"""
import struct
import numpy as np

# sRGB primaries adapted (Bradford) from D65 to the ICC PCS white (D50 = 0.9642, 1.0, 0.8249), so the
# target is consistent with the colorants stored in ICC profiles. Column convention: XYZ = M @ rgb_linear.
_SRGB_D65 = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                      [-0.7502, 1.7135, 0.0367],
                      [0.0389, -0.0685, 1.0296]])


def _bradford_adapt(src_white, dst_white):
    scale = (_BRADFORD @ dst_white) / (_BRADFORD @ src_white)
    return np.linalg.inv(_BRADFORD) @ np.diag(scale) @ _BRADFORD


_SRGB_D50 = _bradford_adapt(np.array([0.95047, 1.0, 1.08883]), np.array([0.9642, 1.0, 0.8249])) @ _SRGB_D65
_XYZ_D50_TO_SRGB = np.linalg.inv(_SRGB_D50)


def _tags(icc: bytes) -> dict:
    n = struct.unpack(">I", icc[128:132])[0]
    out = {}
    for i in range(n):
        sig, off, size = struct.unpack(">4sII", icc[132 + 12 * i: 144 + 12 * i])
        out[sig.decode("latin1")] = icc[off: off + size]
    return out


def icc_description(icc: bytes) -> str | None:
    d = _tags(icc).get("desc")
    if d is None:
        return None
    if d[:4] == b"desc":
        n = struct.unpack(">I", d[8:12])[0]
        return d[12: 12 + n].split(b"\0")[0].decode("latin1")
    if d[:4] == b"mluc":
        length, off = struct.unpack(">II", d[20:28])
        return d[off: off + length].decode("utf-16-be")
    return None


def _xyz(tag: bytes) -> np.ndarray:
    return np.array(struct.unpack(">3i", tag[8:20])) / 65536.0


def _curve(tag: bytes):
    """Returns f(encoded) -> linear for a 'curv' or 'para' tag."""
    typ = tag[:4]
    if typ == b"curv":
        n = struct.unpack(">I", tag[8:12])[0]
        if n == 0:
            return lambda v: v
        if n == 1:
            g = struct.unpack(">H", tag[12:14])[0] / 256.0
            return lambda v: np.sign(v) * np.abs(v) ** g
        table = np.frombuffer(tag[12: 12 + 2 * n], dtype=">u2").astype(np.float64) / 65535.0
        xs = np.linspace(0.0, 1.0, n)
        return lambda v: np.interp(np.clip(v, 0.0, 1.0), xs, table)
    if typ == b"para":
        ftype = struct.unpack(">H", tag[8:10])[0]
        count = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}[ftype]
        p = np.array(struct.unpack(f">{count}i", tag[12: 12 + 4 * count])) / 65536.0
        g = p[0]
        def f(x):
            x = np.asarray(x, np.float64)
            if ftype == 0:
                return np.sign(x) * np.abs(x) ** g
            a, b = p[1], p[2]
            if ftype == 1:
                return np.where(x >= -b / a, np.maximum(a * x + b, 0.0) ** g, 0.0)
            c = p[3]
            if ftype == 2:
                return np.where(x >= -b / a, np.maximum(a * x + b, 0.0) ** g + c, c)
            d = p[4]
            if ftype == 3:
                return np.where(x >= d, np.maximum(a * x + b, 0.0) ** g, c * x)
            e, ff = p[5], p[6]
            return np.where(x >= d, np.maximum(a * x + b, 0.0) ** g + e, c * x + ff)
        return f
    raise ValueError(f"unsupported TRC type {typ!r}")


def srgb_encode(lin):
    a = np.abs(lin)
    return np.sign(lin) * np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)


def icc_to_srgb(rgb, icc: bytes, clip: bool = False) -> np.ndarray:
    """rgb: float array (..., 3), 0..1 in the profile's encoding -> sRGB-encoded float64."""
    t = _tags(icc)
    if "rXYZ" not in t or "rTRC" not in t:
        raise ValueError(f"profile {icc_description(icc)!r} is not a matrix/TRC profile")
    rgb = np.asarray(rgb, np.float64)
    lin = np.stack([_curve(t[f"{c}TRC"])(rgb[..., i]) for i, c in enumerate("rgb")], axis=-1)
    M = np.stack([_xyz(t["rXYZ"]), _xyz(t["gXYZ"]), _xyz(t["bXYZ"])], axis=1)
    out = srgb_encode((lin @ M.T) @ _XYZ_D50_TO_SRGB.T)
    return np.clip(out, 0.0, 1.0) if clip else out


def gamma_icc_bytes(description: str, primaries: list[tuple[float, float, float]], gamma: float) -> bytes:
    """A minimal matrix/TRC ICC profile: desc tag, red / green / blue colorants (D50 XYZ each) and one pure-gamma curve for all three channels."""
    def xyz(v): return b"XYZ \0\0\0\0" + struct.pack(">3i", *[int(round(x * 65536)) for x in v])
    curv = b"curv\0\0\0\0" + struct.pack(">IH", 1, int(round(gamma * 256))) + b"\0\0"
    dsc = b"desc\0\0\0\0" + struct.pack(">I", len(description) + 1) + description.encode() + b"\0" + b"\0" * 60
    tags = [("desc", dsc), ("rXYZ", xyz(primaries[0])), ("gXYZ", xyz(primaries[1])), ("bXYZ", xyz(primaries[2])),
            ("rTRC", curv), ("gTRC", curv), ("bTRC", curv)]
    off, table, body = 128 + 4 + 12 * len(tags), b"", b""
    for s, d in tags:
        table += s.encode() + struct.pack(">II", off + len(body), len(d)); body += d + b"\0" * ((-len(d)) % 4)
    return b"\0" * 128 + struct.pack(">I", len(tags)) + table + body


def srgb_icc_bytes() -> bytes:
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


# ---------------------------------------------------------------------------------------------
# Native-space tools: work in the file's own (large) color space, no sRGB involved.
# "encoded" = file values, "linear" = TRC-decoded in the profile's primaries, "xyz" = CIE XYZ (D50 PCS).
# ---------------------------------------------------------------------------------------------
import functools

_D50 = np.array([0.9642, 1.0, 0.8249])


@functools.lru_cache(maxsize=8)
def _profile(icc: bytes):
    t = _tags(icc)
    grid = np.linspace(0.0, 1.0, 2 ** 18 + 1)
    curves = [_curve(t[f"{c}TRC"]) for c in "rgb"]
    M = np.stack([_xyz(t["rXYZ"]), _xyz(t["gXYZ"]), _xyz(t["bXYZ"])], axis=1)
    return dict(curves=curves, grid=grid, vals=[c(grid) for c in curves], M=M, Minv=np.linalg.inv(M))


def icc_to_linear(rgb, icc: bytes):
    p = _profile(icc)
    rgb = np.asarray(rgb, np.float64)
    return np.stack([p["curves"][i](rgb[..., i]) for i in range(3)], axis=-1)


ENCODE_GRID = 2 ** 14   # table size of icc_from_linear: round trips within 0.5 of a 16-bit level for Adobe RGB, ProPhoto and sRGB


@functools.lru_cache(maxsize=8)
def _encode_tables(icc: bytes) -> np.ndarray:
    """(3, ENCODE_GRID) encoded value of each channel at linear = u^2, u uniform over 0..1 (square-root spacing keeps the steep shadow end fine)."""
    p = _profile(icc)
    u = np.linspace(0.0, 1.0, ENCODE_GRID)
    return np.stack([np.interp(u * u, p["vals"][i], p["grid"]) for i in range(3)])


def icc_from_linear(lin: np.ndarray, icc: bytes) -> np.ndarray:
    """Encodes linear RGB (..., 3) with the profile's TRCs by direct table lookup (no search).  Linear values are clamped to 0..1."""
    tables = _encode_tables(icc)
    x = np.sqrt(np.clip(np.asarray(lin, np.float64), 0.0, 1.0)) * (ENCODE_GRID - 1)
    i0 = np.minimum(x.astype(np.int32), ENCODE_GRID - 2)
    f = x - i0
    return np.stack([tables[i][i0[..., i]] * (1.0 - f[..., i]) + tables[i][i0[..., i] + 1] * f[..., i] for i in range(3)], axis=-1)


def icc_to_xyz(rgb, icc: bytes):
    return icc_to_linear(rgb, icc) @ _profile(icc)["M"].T


def icc_from_xyz(xyz, icc: bytes):
    return icc_from_linear(np.asarray(xyz, np.float64) @ _profile(icc)["Minv"].T, icc)


_EPS, _KAPPA = 216.0 / 24389.0, 24389.0 / 27.0


def xyz_to_lab(xyz):
    t = np.asarray(xyz, np.float64) / _D50
    f = np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_to_xyz(lab):
    lab = np.asarray(lab, np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx, fz = fy + lab[..., 1] / 500.0, fy - lab[..., 2] / 200.0
    inv = lambda f: np.where(f ** 3 > _EPS, f ** 3, (116.0 * f - 16.0) / _KAPPA)
    return np.stack([inv(fx), np.where(lab[..., 0] > _KAPPA * _EPS, fy ** 3, lab[..., 0] / _KAPPA), inv(fz)], axis=-1) * _D50
