import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from scipy import signal

import dhe

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("img_name", "max_mae"),
    [
        ("01.jpg", 10.0),
        ("02.jpg", 4.0),
        ("03.jpg", 6.5),
        ("04.jpg", 3.5),
    ],
)
def test_dhe_matches_reference(img_name, max_mae):
    img = iio.imread(REPO_ROOT / "testdata" / img_name)
    ref = iio.imread(REPO_ROOT / "result" / "dhe" / img_name)

    out = dhe.dhe(img)
    assert out.shape == ref.shape
    assert out.dtype == np.uint8
    mae = np.abs(out.astype(np.float64) - ref.astype(np.float64)).mean()
    assert mae < max_mae


def test_dhe_helpers_match_naive():
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(24, 30), dtype=np.int64)

    # _box_sum_5x5 vs scipy.signal.convolve2d valid mode
    expected_box = signal.convolve2d(arr, np.ones((5, 5), dtype=np.int64), mode="valid")
    np.testing.assert_array_equal(dhe._box_sum_5x5(arr), expected_box)

    # _sobel_mag vs scipy.signal.convolve2d on 2px-padded interior
    fh = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    fv = fh.T
    gx = signal.convolve2d(arr.astype(np.float64), np.rot90(fh, 2), mode="same")[2:-2, 2:-2]
    gy = signal.convolve2d(arr.astype(np.float64), np.rot90(fv, 2), mode="same")[2:-2, 2:-2]
    expected_sobel = np.hypot(gx, gy).astype(np.uint32)
    np.testing.assert_array_equal(dhe._sobel_mag(arr), expected_sobel)


def test_build_is_hist_contract():
    img = iio.imread(REPO_ROOT / "testdata" / "01.jpg")
    hist_i, hist_s = dhe.build_is_hist(img)
    assert hist_i.shape == (256, 1)
    assert hist_s.shape == (256, 1)
    assert hist_i[0, 0] == 0.0
    assert hist_s[0, 0] == 0.0
    assert np.all(hist_i >= 0) and np.all(hist_s >= 0)


def test_dhe_grayscale_rgba_and_edge_cases():
    img = iio.imread(REPO_ROOT / "testdata" / "01.jpg")

    gray = img[:, :, 0]
    out_gray = dhe.dhe(gray)
    assert out_gray.shape == gray.shape
    assert out_gray.dtype == np.uint8

    alpha = np.full(img.shape[:2] + (1,), 77, dtype=np.uint8)
    rgba = np.concatenate([img, alpha], axis=2)
    out_rgba = dhe.dhe(rgba, alpha=0.3)
    assert out_rgba.shape == rgba.shape
    assert np.all(out_rgba[:, :, 3] == 77)

    for val in (0, 128, 255):
        flat = np.full((20, 20, 3), val, dtype=np.uint8)
        out_flat = dhe.dhe(flat)
        assert out_flat.shape == flat.shape
        assert out_flat.dtype == np.uint8

    tiny = np.array([[[10, 20, 30]]], dtype=np.uint8)
    assert dhe.dhe(tiny).shape == tiny.shape


def test_dhe_invalid_inputs():
    with pytest.raises(ValueError):
        dhe.dhe(np.zeros((10,), dtype=np.uint8))
    with pytest.raises(ValueError):
        dhe.build_is_hist(np.zeros((10, 10, 2), dtype=np.uint8))


def test_dhe_cli(tmp_path):
    out_path = tmp_path / "dhe_out.jpg"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "dhe.py"),
            str(REPO_ROOT / "testdata" / "01.jpg"),
            "--alpha",
            "0.4",
            "-o",
            str(out_path),
            "--no-show",
        ],
        check=True,
    )
    assert out_path.exists() and out_path.stat().st_size > 0
