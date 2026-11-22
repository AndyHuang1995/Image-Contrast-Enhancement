import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

import he

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("img_name", "max_mae"),
    [
        ("01.jpg", 13.0),
        ("02.jpg", 4.0),
        ("03.jpg", 6.5),
        ("04.jpg", 4.0),
    ],
)
def test_he_matches_reference(img_name, max_mae):
    img = iio.imread(REPO_ROOT / "testdata" / img_name)
    ref = iio.imread(REPO_ROOT / "result" / "he" / img_name)

    out = he.he(img)
    assert out.shape == ref.shape
    assert out.dtype == np.uint8
    mae = np.abs(out.astype(np.float64) - ref.astype(np.float64)).mean()
    assert mae < max_mae


def test_he_ycrcb_mode():
    img = iio.imread(REPO_ROOT / "testdata" / "01.jpg")
    out = he.he(img, mode="ycrcb")
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_he_grayscale_rgba_and_edge_cases():
    img = iio.imread(REPO_ROOT / "testdata" / "01.jpg")

    gray = img[:, :, 0]
    out_gray = he.he(gray)
    assert out_gray.shape == gray.shape
    assert out_gray.dtype == np.uint8

    alpha = np.full(img.shape[:2] + (1,), 123, dtype=np.uint8)
    rgba = np.concatenate([img, alpha], axis=2)
    for mode in ("rgb", "ycrcb"):
        out_rgba = he.he(rgba, mode=mode)
        assert out_rgba.shape == rgba.shape
        assert np.all(out_rgba[:, :, 3] == 123)

    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    out_flat = he.he(flat)
    assert out_flat.shape == flat.shape
    assert out_flat.dtype == np.uint8

    float_img = img.astype(np.float32) / 255.0
    out_float = he.he(float_img)
    assert out_float.shape == img.shape
    assert out_float.dtype == np.uint8


def test_he_invalid_inputs():
    with pytest.raises(ValueError):
        he.he(np.zeros((10,), dtype=np.uint8))
    with pytest.raises(ValueError):
        he.he(np.zeros((10, 10, 3), dtype=np.uint8), mode="invalid")


def test_he_cli(tmp_path):
    out_path = tmp_path / "he_out.jpg"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "he.py"),
            str(REPO_ROOT / "testdata" / "01.jpg"),
            "--mode",
            "ycrcb",
            "-o",
            str(out_path),
            "--no-show",
        ],
        check=True,
    )
    assert out_path.exists() and out_path.stat().st_size > 0
