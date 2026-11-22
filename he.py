import argparse
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np


def _equalize_channel(channel):
    """Equalize a single 2D channel using a fast bincount LUT for uint8."""
    if channel.dtype == np.uint8:
        hist = np.bincount(channel.ravel(), minlength=256)
        cdf = hist.cumsum()
        total = cdf[-1]
        if total == 0:
            return channel.copy()
        lut = np.clip(cdf * (255.0 / total), 0, 255).astype(np.uint8)
        return lut[channel]

    hist, bin_edges = np.histogram(channel, bins=256)
    cdf = hist.cumsum()
    total = cdf[-1]
    if total == 0:
        return np.zeros_like(channel, dtype=np.uint8)
    cdf_norm = cdf * (255.0 / total)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    out = np.interp(channel.ravel(), bin_centers, cdf_norm).reshape(channel.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def he(img, mode="rgb"):
    """Apply Histogram Equalization (HE).

    Args:
        img: 2D grayscale or 3D RGB/RGBA image array.
        mode: Color handling mode for 3D images:
            - "rgb": equalize each RGB channel independently (default).
            - "ycrcb": equalize luminance (Y) channel only to preserve color.
    """
    if img.ndim == 2:
        return _equalize_channel(img)

    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape: {img.shape}")

    has_alpha = img.shape[2] == 4
    rgb = img[:, :, :3]

    if mode == "rgb":
        out_rgb = np.empty(rgb.shape, dtype=np.uint8)
        for ch in range(3):
            out_rgb[:, :, ch] = _equalize_channel(rgb[:, :, ch])
    elif mode == "ycrcb":
        rgb_u8 = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)
        ycrcb = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = _equalize_channel(ycrcb[:, :, 0])
        out_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    else:
        raise ValueError(f"Unsupported mode: {mode!r}. Expected 'rgb' or 'ycrcb'.")

    if has_alpha:
        return np.concatenate((out_rgb, img[:, :, 3:4].astype(np.uint8)), axis=2)
    return out_rgb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Histogram Equalization (HE) for image contrast enhancement."
    )
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument(
        "--mode",
        choices=["rgb", "ycrcb"],
        default="rgb",
        help="Color equalization mode for RGB images (default: rgb)",
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
    result = he(img, mode=args.mode)

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
