import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure as ex


def he(img):
    if img.ndim == 2:  # Grayscale
        out_img = ex.equalize_hist(img) * 255.0
    elif img.ndim == 3:  # RGB / RGBA
        out_img = np.zeros_like(img, dtype=np.float64)
        num_channels = min(img.shape[2], 3)
        for channel in range(num_channels):
            out_img[:, :, channel] = ex.equalize_hist(img[:, :, channel]) * 255.0
        if img.shape[2] == 4:
            out_img[:, :, 3] = img[:, :, 3]
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    return np.clip(out_img, 0, 255).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Histogram Equalization (HE) for image contrast enhancement."
    )
    parser.add_argument("input_image", help="Path to the input image")
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
    result = he(img)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(output_path, result)

    if not args.no_show:
        plt.imshow(result, cmap="gray" if result.ndim == 2 else None)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
