import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from scipy import signal

import ying

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("img_name", "max_mae"),
    [
        ("01.jpg", 6.0),
        ("02.jpg", 3.5),
        ("03.jpg", 6.0),
        ("04.jpg", 3.5),
    ],
)
def test_ying_matches_reference(img_name, max_mae):
    img = iio.imread(REPO_ROOT / "testdata" / img_name)
    ref = iio.imread(REPO_ROOT / "result" / "ying" / img_name)

    out = ying.Ying_2017_CAIP(img)
    assert out.shape == ref.shape
    assert out.dtype == np.uint8
    mae = np.abs(out.astype(np.float64) - ref.astype(np.float64)).mean()
    assert mae < max_mae


@pytest.mark.parametrize("boundary", ["neumann", "periodic"])
def test_ying_pcg_matches_direct_solver(boundary):
    img = iio.imread(REPO_ROOT / "testdata" / "01.jpg")
    out_pcg = ying.Ying_2017_CAIP(img, solver="pcg", boundary=boundary)
    out_direct = ying.Ying_2017_CAIP(img, solver="direct", boundary=boundary)

    diff = np.abs(out_pcg.astype(np.int16) - out_direct.astype(np.int16))
    assert diff.max() <= 2
    assert diff.mean() < 0.1


def test_ying_neumann_eliminates_border_wrap_halo():
    # Bright top half (sky) over dark bottom half (ground)
    sky_ground = np.full((64, 64), 0.1, dtype=np.float64)
    sky_ground[:32, :] = 1.0

    s_neumann = ying.tsmooth(sky_ground, lamda=0.5, sigma=5, boundary="neumann")
    s_periodic = ying.tsmooth(sky_ground, lamda=0.5, sigma=5, boundary="periodic")

    # Periodic bleeds bottom darkness into top row (creating a border halo dip);
    # Neumann stays monotonic with no wrap-around bleeding.
    assert s_neumann[0, 32] >= s_neumann[15, 32] - 1e-3
    assert s_neumann[-1, 32] <= s_neumann[48, 32] + 1e-3
    assert s_periodic[0, 32] < s_periodic[15, 32] - 0.005
    assert s_neumann[0, 32] > s_periodic[0, 32] + 0.015
    assert s_neumann[-1, 32] < s_periodic[-1, 32] - 0.015


def test_ying_tunable_parameters():
    img = iio.imread(REPO_ROOT / "testdata" / "01.jpg")
    out = ying.Ying_2017_CAIP(img, lamda=0.3, sigma=3, k_max=3.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_ying_uint8_lut_matches_float_path():
    img = iio.imread(REPO_ROOT / "testdata" / "01.jpg")
    out_u8 = ying.Ying_2017_CAIP(img)
    out_f32 = ying.Ying_2017_CAIP(img.astype(np.float32))
    diff = np.abs(out_u8.astype(np.int16) - out_f32.astype(np.int16))
    assert diff.max() <= 1
    assert diff.mean() < 0.01


def test_ying_max_smooth_size():
    img = iio.imread(REPO_ROOT / "testdata" / "03.jpg")
    out_uncapped = ying.Ying_2017_CAIP(img, max_smooth_size=None)
    out_capped = ying.Ying_2017_CAIP(img, max_smooth_size=240)
    assert out_capped.shape == img.shape
    diff = np.abs(out_capped.astype(np.float64) - out_uncapped.astype(np.float64))
    assert diff.mean() < 3.5


@pytest.mark.parametrize("kernel_size", [1, 3, 4, 5, 7])
def test_box_sum_1d_matches_convolve2d(kernel_size):
    rng = np.random.default_rng(7)
    arr = rng.normal(size=(23, 31))

    ref_h = signal.convolve2d(arr, np.ones((1, kernel_size)), mode="same")
    ref_v = signal.convolve2d(arr, np.ones((kernel_size, 1)), mode="same")
    np.testing.assert_allclose(ying._box_sum_1d(arr, kernel_size, axis=1), ref_h, atol=1e-12)
    np.testing.assert_allclose(ying._box_sum_1d(arr, kernel_size, axis=0), ref_v, atol=1e-12)


def test_solve_linear_equation_does_not_mutate_weights():
    rng = np.random.default_rng(0)
    IN = rng.random((16, 20))
    wx, wy = ying.computeTextureWeights(IN, sigma=3, sharpness=0.001)
    wx_orig, wy_orig = wx.copy(), wy.copy()

    for solver in ("pcg", "direct"):
        out = ying.solveLinearEquation(IN, wx, wy, lamda=0.5, solver=solver)
        assert out.shape == IN.shape
        np.testing.assert_array_equal(wx, wx_orig)
        np.testing.assert_array_equal(wy, wy_orig)


def test_entropy_properties():
    assert ying.entropy(np.array([], dtype=np.float64)) == 0.0
    assert ying.entropy(np.full((16, 16), 0.5)) == 0.0
    uniform = (np.arange(256, dtype=np.float64) + 0.1) / 255.0
    assert ying.entropy(uniform) == pytest.approx(8.0, abs=1e-6)


def test_ying_grayscale_rgba_and_edge_cases():
    img = iio.imread(REPO_ROOT / "testdata" / "01.jpg")

    gray = img[:, :, 0]
    out_gray = ying.Ying_2017_CAIP(gray)
    assert out_gray.shape == gray.shape
    assert out_gray.dtype == np.uint8

    alpha = np.full(img.shape[:2] + (1,), 200, dtype=np.uint8)
    rgba = np.concatenate([img, alpha], axis=2)
    out_rgba = ying.Ying_2017_CAIP(rgba)
    assert out_rgba.shape == rgba.shape
    assert np.all(out_rgba[:, :, 3] == 200)

    for val in (0, 128, 255):
        flat = np.full((32, 32, 3), val, dtype=np.uint8)
        out_flat = ying.Ying_2017_CAIP(flat)
        assert out_flat.shape == flat.shape
        assert out_flat.dtype == np.uint8

    for shape in ((1, 1, 3), (2, 2, 3)):
        tiny = np.full(shape, 42, dtype=np.uint8)
        assert ying.Ying_2017_CAIP(tiny).shape == shape


def test_ying_invalid_inputs():
    with pytest.raises(ValueError):
        ying.Ying_2017_CAIP(np.zeros((10,), dtype=np.uint8))
    with pytest.raises(ValueError):
        ying.Ying_2017_CAIP(np.zeros((16, 16, 3), dtype=np.uint8), solver="unknown")
    with pytest.raises(ValueError):
        ying.Ying_2017_CAIP(np.zeros((16, 16, 3), dtype=np.uint8), boundary="unknown")


def test_ying_cli(tmp_path):
    out_path = tmp_path / "ying_out.jpg"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "ying.py"),
            str(REPO_ROOT / "testdata" / "01.jpg"),
            "--solver",
            "pcg",
            "-o",
            str(out_path),
            "--no-show",
        ],
        check=True,
    )
    assert out_path.exists() and out_path.stat().st_size > 0
