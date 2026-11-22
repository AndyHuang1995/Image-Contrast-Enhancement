import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def build_is_hist(img):
    hei = img.shape[0]
    wid = img.shape[1]
    ch = img.shape[2]
    Img = np.zeros((hei + 4, wid + 4, ch))
    for i in range(ch):
        Img[:, :, i] = np.pad(img[:, :, i], (2, 2), "edge")
    hsv = matplotlib.colors.rgb_to_hsv(Img)
    hsv[:, :, 0] = hsv[:, :, 0] * 255
    hsv[:, :, 1] = hsv[:, :, 1] * 255
    hsv = np.clip(hsv, 0, 255).astype(np.uint8).astype(np.float64)
    fh = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    fv = fh.conj().T

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    I = hsv[:, :, 2]

    dIh = signal.convolve2d(I, np.rot90(fh, 2), mode="same")
    dIv = signal.convolve2d(I, np.rot90(fv, 2), mode="same")
    dIh[dIh == 0] = 0.00001
    dIv[dIv == 0] = 0.00001
    dI = np.sqrt(dIh**2 + dIv**2).astype(np.uint32)
    di = dI[2 : hei + 2, 2 : wid + 2]

    dSh = signal.convolve2d(S, np.rot90(fh, 2), mode="same")
    dSv = signal.convolve2d(S, np.rot90(fv, 2), mode="same")
    dSh[dSh == 0] = 0.00001
    dSv[dSv == 0] = 0.00001
    dS = np.sqrt(dSh**2 + dSv**2).astype(np.uint32)
    ds = dS[2 : hei + 2, 2 : wid + 2]

    i = I[2 : hei + 2, 2 : wid + 2].astype(np.uint8)

    Rho = np.zeros((hei + 4, wid + 4))
    with np.errstate(divide="ignore", invalid="ignore"):
        for p in range(2, hei + 2):
            for q in range(2, wid + 2):
                tmpi = I[p - 2 : p + 3, q - 2 : q + 3]
                tmps = S[p - 2 : p + 3, q - 2 : q + 3]
                corre = np.corrcoef(tmpi.flatten("F"), tmps.flatten("F"))
                Rho[p, q] = corre[0, 1]

    rho = np.abs(Rho[2 : hei + 2, 2 : wid + 2])
    rho[np.isnan(rho)] = 0
    rd = (rho * ds).astype(np.uint32)
    Hist_I = np.zeros((256, 1))
    Hist_S = np.zeros((256, 1))

    for n in range(0, 255):
        temp = np.zeros(di.shape)
        temp[i == n] = di[i == n]
        Hist_I[n + 1, 0] = np.sum(temp.flatten("F"))
        temp = np.zeros(di.shape)
        temp[i == n] = rd[i == n]
        Hist_S[n + 1, 0] = np.sum(temp.flatten("F"))

    return Hist_I, Hist_S


def dhe(img, alpha=0.5):
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    hist_i, hist_s = build_is_hist(img)
    hist_c = alpha * hist_s + (1 - alpha) * hist_i
    hist_sum = np.sum(hist_c)
    hist_cum = hist_c.cumsum(axis=0)

    hsv = matplotlib.colors.rgb_to_hsv(np.asarray(img, dtype=np.float64))
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    i = hsv[:, :, 2].astype(np.uint8)

    c = hist_cum / hist_sum
    s_r = c * 255
    i_s = np.zeros(i.shape)
    for n in range(0, 255):
        i_s[i == n] = s_r[n + 1, 0] / 255.0
    i_s[i == 255] = 1
    hsi_o = np.stack((h, s, i_s), axis=2)
    result = matplotlib.colors.hsv_to_rgb(hsi_o)

    result = np.clip(result * 255, 0, 255)
    return result.astype(np.uint8)


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
        plt.imshow(result)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
