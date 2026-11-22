import argparse
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np


def _box_sum_5x5(arr):
    """Compute exact valid 5x5 window sums via separable 1D slicing."""
    h_sum = arr[:, :-4] + arr[:, 1:-3] + arr[:, 2:-2] + arr[:, 3:-1] + arr[:, 4:]
    return h_sum[:-4] + h_sum[1:-3] + h_sum[2:-2] + h_sum[3:-1] + h_sum[4:]


def _sobel_mag(arr):
    """Compute Sobel gradient magnitude on the interior [2:-2, 2:-2] of a 2px-padded array."""
    gx = (
        (arr[1:-3, 3:-1] - arr[1:-3, 1:-3])
        + 2 * (arr[2:-2, 3:-1] - arr[2:-2, 1:-3])
        + (arr[3:-1, 3:-1] - arr[3:-1, 1:-3])
    )
    gy = (
        (arr[3:-1, 1:-3] - arr[1:-3, 1:-3])
        + 2 * (arr[3:-1, 2:-2] - arr[1:-3, 2:-2])
        + (arr[3:-1, 3:-1] - arr[1:-3, 3:-1])
    )
    return np.hypot(gx, gy).astype(np.uint32)


def _build_is_hist_from_si(s_u8, i_u8):
    S = np.pad(s_u8, 2, mode="edge").astype(np.int64)
    I = np.pad(i_u8, 2, mode="edge").astype(np.int64)

    di = _sobel_mag(I)
    ds = _sobel_mag(S)

    sum_I = _box_sum_5x5(I)
    sum_S = _box_sum_5x5(S)
    sum_II = _box_sum_5x5(I * I)
    sum_SS = _box_sum_5x5(S * S)
    sum_IS = _box_sum_5x5(I * S)

    num = 25 * sum_IS - sum_I * sum_S
    var_I = 25 * sum_II - sum_I * sum_I
    var_S = 25 * sum_SS - sum_S * sum_S
    denom = np.sqrt(var_I.astype(np.float64) * var_S.astype(np.float64))
    rho = np.divide(np.abs(num), denom, out=np.zeros_like(denom), where=denom > 0)

    rd = (rho * ds).astype(np.uint32)

    hist_i = np.zeros((256, 1), dtype=np.float64)
    hist_s = np.zeros((256, 1), dtype=np.float64)
    flat_i = i_u8.ravel()
    hist_i[1:, 0] = np.bincount(flat_i, weights=di.ravel(), minlength=256)[:255]
    hist_s[1:, 0] = np.bincount(flat_i, weights=rd.ravel(), minlength=256)[:255]
    return hist_i, hist_s


def build_is_hist(img):
    """Build intensity and saturation gradient-weighted histograms."""
    if img.ndim == 2:
        i_u8 = np.clip(img, 0, 255).astype(np.uint8)
        s_u8 = np.zeros_like(i_u8)
    elif img.ndim == 3 and img.shape[2] >= 3:
        hsv = cv2.cvtColor(np.asarray(img[:, :, :3], dtype=np.float32), cv2.COLOR_RGB2HSV)
        s_u8 = np.clip(hsv[:, :, 1] * 255.0, 0, 255).astype(np.uint8)
        i_u8 = np.clip(hsv[:, :, 2], 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return _build_is_hist_from_si(s_u8, i_u8)


def dhe(img, alpha=0.5):
    """Apply Dynamic Histogram Equalization (DHE) using HSV intensity-saturation correlation."""
    if img.ndim == 2:
        i_u8 = np.clip(img, 0, 255).astype(np.uint8)
        s_u8 = np.zeros_like(i_u8)
        hist_i, hist_s = _build_is_hist_from_si(s_u8, i_u8)
        hist_c = alpha * hist_s + (1.0 - alpha) * hist_i
        hist_sum = hist_c.sum()
        if hist_sum == 0:
            return i_u8.copy()
        lut = np.empty(256, dtype=np.uint8)
        lut[:255] = np.clip(
            hist_c.cumsum()[1:] * (255.0 / hist_sum), 0, 255
        ).astype(np.uint8)
        lut[255] = 255
        return lut[i_u8]

    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape: {img.shape}")

    has_alpha = img.shape[2] == 4
    rgb = img[:, :, :3]
    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.float32), cv2.COLOR_RGB2HSV)
    s_u8 = np.clip(hsv[:, :, 1] * 255.0, 0, 255).astype(np.uint8)
    i_u8 = np.clip(hsv[:, :, 2], 0, 255).astype(np.uint8)

    hist_i, hist_s = _build_is_hist_from_si(s_u8, i_u8)
    hist_c = alpha * hist_s + (1.0 - alpha) * hist_i
    hist_sum = hist_c.sum()
    if hist_sum == 0:
        return np.clip(img, 0, 255).astype(np.uint8)

    hist_cum = hist_c.cumsum(axis=0)
    lut = np.empty(256, dtype=np.float32)
    lut[:255] = (hist_cum[1:, 0] * (255.0 / hist_sum)).astype(np.float32)
    lut[255] = 255.0

    hsv[:, :, 2] = lut[i_u8]
    result = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 255).astype(np.uint8)
    if has_alpha:
        return np.concatenate((result, img[:, :, 3:4].astype(np.uint8)), axis=2)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamic Histogram Equalization (DHE) for image contrast enhancement."
    )
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weighting parameter between saturation and intensity histograms (default: 0.5)",
    )
    parser.add_argument(
        "-o", "--output", help="Optional path to save the enhanced image"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a matplotlib preview window",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    img = iio.imread(args.input_image)
    result = dhe(img, alpha=args.alpha)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(output_path, result)

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.imshow(result, cmap="gray" if result.ndim == 2 else None)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
