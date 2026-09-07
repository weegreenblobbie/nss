"""Tests of nss.icc: icc_from_linear is the inverse of icc_to_linear."""
import numpy as np
import pytest

from nss.icc import _profile, gamma_icc_bytes, icc_description, icc_from_linear, icc_to_linear, srgb_icc_bytes

ADOBE = gamma_icc_bytes("Adobe RGB (1998)", [(0.60974, 0.31111, 0.01947), (0.20528, 0.62567, 0.06087), (0.14919, 0.06322, 0.74457)], 563 / 256)
PROPHOTO = gamma_icc_bytes("ProPhoto RGB", [(0.7977, 0.2880, 0.0), (0.1352, 0.7119, 0.0), (0.0313, 0.0001, 0.8249)], 1.8)


@pytest.mark.parametrize("icc", [ADOBE, PROPHOTO, srgb_icc_bytes()], ids=["adobe", "prophoto", "srgb"])
def test_round_trip_within_half_a_16_bit_level(icc: bytes) -> None:
    """Decoding then encoding returns the (13, 12, 3) input within 0.5 / 65535, shadows included."""
    enc = np.random.default_rng(4).random((13, 12, 3)) ** 3
    enc[0, 0] = (0.0, 1.0, 1e-5)
    lin = icc_to_linear(enc, icc)
    assert np.abs(lin - enc).max() > 0.05                     # the curve is not the identity
    out = icc_from_linear(lin, icc)
    assert out.shape == (13, 12, 3)
    assert np.abs(out - enc).max() < 0.5 / 65535


def test_clamps_outside_zero_to_one() -> None:
    """Linear values below 0 or above 1 encode as 0 and 1; a mid value follows the profile's gamma."""
    out = icc_from_linear(np.array([[-0.2, 1.3, 0.5]]), ADOBE)
    assert out[0, 0] == 0.0 and out[0, 1] == 1.0
    assert out[0, 2] == pytest.approx(0.5 ** (256 / 563), abs=1e-5)


def test_gamma_icc_bytes_description_colorants_and_gamma() -> None:
    """The built profile carries its description, its three colorants as the matrix columns and a pure-gamma TRC (gamma stored in 1/256 steps)."""
    primaries = [(0.7977, 0.2880, 0.0), (0.1352, 0.7119, 0.0), (0.0313, 0.0001, 0.8249)]
    icc = gamma_icc_bytes("ProPhoto RGB", primaries, 1.8)
    assert icc_description(icc) == "ProPhoto RGB"
    assert np.allclose(_profile(icc)["M"], np.array(primaries).T, atol=1.0 / 65536)
    enc = np.random.default_rng(6).random((4, 2, 3))
    assert np.allclose(icc_to_linear(enc, icc), enc ** (round(1.8 * 256) / 256), atol=1e-12)
    assert len(icc) % 4 == 0 and int.from_bytes(icc[128:132], "big") == 7       # seven tags, 4-byte aligned
