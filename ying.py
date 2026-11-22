import argparse
from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import optimize, signal, sparse
from scipy.sparse.linalg import spsolve


def imresize_bicubic(arr, size):
    """Resize a 2D array with bicubic interpolation (replaces removed scipy.misc.imresize)."""
    pil_img = Image.fromarray(np.asarray(arr, dtype=np.float32))
    if isinstance(size, (int, float)):
        new_h = int(arr.shape[0] * size)
        new_w = int(arr.shape[1] * size)
    else:
        new_h, new_w = int(size[0]), int(size[1])
    resized = pil_img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    return np.array(resized, dtype=np.float32)


def computeTextureWeights(fin, sigma, sharpness):
    kernel_size = int(sigma)
    dt0_v = np.vstack((np.diff(fin, n=1, axis=0), fin[0, :] - fin[-1, :]))
    dt0_h = np.vstack(
        (np.diff(fin, n=1, axis=1).conj().T, fin[:, 0].conj().T - fin[:, -1].conj().T)
    ).conj().T

    gauker_h = signal.convolve2d(dt0_h, np.ones((1, kernel_size)), mode="same")
    gauker_v = signal.convolve2d(dt0_v, np.ones((kernel_size, 1)), mode="same")

    W_h = 1 / (np.abs(gauker_h) * np.abs(dt0_h) + sharpness)
    W_v = 1 / (np.abs(gauker_v) * np.abs(dt0_v) + sharpness)

    return W_h, W_v


def solveLinearEquation(IN, wx, wy, lamda):
    [r, c] = IN.shape
    k = r * c
    dx = -lamda * wx.flatten("F")
    dy = -lamda * wy.flatten("F")
    tempx = np.roll(wx, 1, axis=1)
    tempy = np.roll(wy, 1, axis=0)
    dxa = -lamda * tempx.flatten("F")
    dya = -lamda * tempy.flatten("F")
    tmp = wx[:, -1]
    tempx = np.concatenate((tmp[:, None], np.zeros((r, c - 1))), axis=1)
    tmp = wy[-1, :]
    tempy = np.concatenate((tmp[None, :], np.zeros((r - 1, c))), axis=0)
    dxd1 = -lamda * tempx.flatten("F")
    dyd1 = -lamda * tempy.flatten("F")

    wx[:, -1] = 0
    wy[-1, :] = 0
    dxd2 = -lamda * wx.flatten("F")
    dyd2 = -lamda * wy.flatten("F")

    Ax = sparse.spdiags(
        np.concatenate((dxd1[:, None], dxd2[:, None]), axis=1).T,
        np.array([-k + r, -r]),
        k,
        k,
    )
    Ay = sparse.spdiags(
        np.concatenate((dyd1[None, :], dyd2[None, :]), axis=0),
        np.array([-r + 1, -1]),
        k,
        k,
    )
    D = 1 - (dx + dy + dxa + dya)
    A = ((Ax + Ay) + (Ax + Ay).conj().T + sparse.spdiags(D, 0, k, k)).T.tocsc()

    tin = IN[:, :]
    tout = spsolve(A, tin.flatten("F"))
    OUT = np.reshape(tout, (r, c), order="F")

    return OUT


def tsmooth(img, lamda=0.01, sigma=3, sharpness=0.001):
    I = cv2.normalize(img.astype("float64"), None, 0.0, 1.0, cv2.NORM_MINMAX)
    x = np.copy(I)
    wx, wy = computeTextureWeights(x, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lamda)
    return S


def rgb2gm(I):
    if I.shape[2] == 3:
        I = cv2.normalize(I.astype("float64"), None, 0.0, 1.0, cv2.NORM_MINMAX)
        I = np.abs((I[:, :, 0] * I[:, :, 1] * I[:, :, 2])) ** (1 / 3)

    return I


def applyK(I, k, a=-0.3293, b=1.1258):
    f = lambda x: np.exp((1 - x**a) * b)
    beta = f(k)
    gamma = k**a
    J = (I**gamma) * beta
    return J


def entropy(X):
    tmp = np.clip(X * 255, 0, 255).astype(np.uint8)
    _, counts = np.unique(tmp, return_counts=True)
    pk = np.asarray(counts, dtype=np.float64)
    pk = pk / np.sum(pk, axis=0)
    S = -np.sum(pk * np.log2(pk), axis=0)
    return S


def maxEntropyEnhance(I, isBad, a=-0.3293, b=1.1258):
    # Estimate k
    tmp = cv2.resize(I, (50, 50), interpolation=cv2.INTER_AREA)
    tmp[tmp < 0] = 0
    tmp = tmp.real
    Y = rgb2gm(tmp)

    isBad = imresize_bicubic(isBad.astype(np.float32), (50, 50))
    isBad[isBad < 0.5] = 0
    isBad[isBad >= 0.5] = 1
    Y = Y[isBad == 1]

    if Y.size == 0:
        return I

    f = lambda k: -entropy(applyK(Y, k, a, b))
    opt_k = optimize.fminbound(f, 1, 7)

    # Apply k
    J = applyK(I, opt_k, a, b) - 0.01
    return J


def Ying_2017_CAIP(img, mu=0.5, a=-0.3293, b=1.1258):
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    lamda = 0.5
    sigma = 5
    I = cv2.normalize(img.astype("float64"), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Weight matrix estimation
    t_b = np.max(I, axis=2)
    t_our = cv2.resize(
        tsmooth(imresize_bicubic(t_b, 0.5), lamda, sigma),
        (t_b.shape[1], t_b.shape[0]),
        interpolation=cv2.INTER_AREA,
    )

    # Apply camera model with k (exposure ratio)
    isBad = t_our < 0.5
    J = maxEntropyEnhance(I, isBad, a=a, b=b)

    # W: Weight Matrix
    t = np.zeros((t_our.shape[0], t_our.shape[1], I.shape[2]))
    for i in range(I.shape[2]):
        t[:, :, i] = t_our
    W = t**mu

    I2 = I * W
    J2 = J * (1 - W)

    result = np.clip((I2 + J2) * 255, 0, 255)
    return result.astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exposure Fusion Framework (Ying et al., CAIP 2017) for image contrast enhancement."
    )
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument(
        "--mu",
        type=float,
        default=0.5,
        help="Weight matrix exponent parameter (default: 0.5)",
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
    result = Ying_2017_CAIP(img, mu=args.mu)

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
