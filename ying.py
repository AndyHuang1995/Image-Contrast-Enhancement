import argparse
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from PIL import Image
from scipy import optimize, sparse
from scipy.sparse.linalg import spsolve


def imresize_bicubic(arr, size):
    """Resize a 2D array with bicubic interpolation (replaces removed scipy.misc.imresize)."""
    pil_img = Image.fromarray(np.asarray(arr, dtype=np.float32))
    if isinstance(size, (int, float)):
        new_h = max(1, int(arr.shape[0] * size))
        new_w = max(1, int(arr.shape[1] * size))
    else:
        new_h, new_w = int(size[0]), int(size[1])
    resized = pil_img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    return np.array(resized, dtype=np.float32)


def _box_sum_1d(arr, kernel_size, axis):
    """Zero-padded 1D box sum matching scipy.signal.convolve2d(..., mode='same')."""
    if kernel_size <= 1:
        return arr.copy()
    pad_before = kernel_size // 2
    pad_after = kernel_size - 1 - pad_before
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (pad_before + 1, pad_after)
    padded = np.pad(arr, pad_width, mode="constant")
    cs = np.cumsum(padded, axis=axis)
    if axis == 1:
        return cs[:, kernel_size:] - cs[:, :-kernel_size]
    return cs[kernel_size:, :] - cs[:-kernel_size, :]


def computeTextureWeights(fin, sigma, sharpness, boundary="neumann"):
    """Compute horizontal and vertical relative total variation texture weights."""
    if boundary not in ("neumann", "periodic"):
        raise ValueError(
            f"Unsupported boundary: {boundary!r}. Expected 'neumann' or 'periodic'."
        )
    kernel_size = max(1, int(sigma))
    dt0_v = np.empty_like(fin)
    dt0_v[:-1, :] = fin[1:, :] - fin[:-1, :]
    dt0_v[-1, :] = 0.0 if boundary == "neumann" else fin[0, :] - fin[-1, :]

    dt0_h = np.empty_like(fin)
    dt0_h[:, :-1] = fin[:, 1:] - fin[:, :-1]
    dt0_h[:, -1] = 0.0 if boundary == "neumann" else fin[:, 0] - fin[:, -1]

    gauker_h = _box_sum_1d(dt0_h, kernel_size, axis=1)
    gauker_v = _box_sum_1d(dt0_v, kernel_size, axis=0)

    W_h = 1.0 / (np.abs(gauker_h) * np.abs(dt0_h) + sharpness)
    W_v = 1.0 / (np.abs(gauker_v) * np.abs(dt0_v) + sharpness)
    return W_h, W_v


def _solve_pcg(IN, wx, wy, lamda, rtol=5e-3, maxiter=500, boundary="neumann"):
    """Solve (I + lamda * L) x = IN via matrix-free float32 Jacobi-PCG."""
    wx_m = (-lamda * wx).astype(np.float32)
    wy_m = (-lamda * wy).astype(np.float32)

    if boundary == "neumann":
        wx_m[:, -1] = 0.0
        wy_m[-1, :] = 0.0
        diag = np.ones(IN.shape, dtype=np.float32)
        diag -= wx_m
        diag -= wy_m
        diag[:, 1:] -= wx_m[:, :-1]
        diag[1:, :] -= wy_m[:-1, :]
    else:
        wx_l = np.roll(wx_m, 1, axis=1)
        wy_u = np.roll(wy_m, 1, axis=0)
        diag = (1.0 - (wx_m + wy_m + wx_l + wy_u)).astype(np.float32)

    inv_diag = 1.0 / diag
    b = np.asarray(IN, dtype=np.float32)
    b_flat = b.ravel()
    b_norm_sq = float(np.dot(b_flat, b_flat))
    if b_norm_sq == 0.0:
        return np.zeros(IN.shape, dtype=np.float64)

    x = b.copy()
    Ap = np.empty_like(b)

    def apply_A(u, out):
        np.multiply(diag, u, out=out)
        out[:, :-1] += wx_m[:, :-1] * u[:, 1:]
        out[:, 1:] += wx_m[:, :-1] * u[:, :-1]
        out[:-1, :] += wy_m[:-1, :] * u[1:, :]
        out[1:, :] += wy_m[:-1, :] * u[:-1, :]
        if boundary == "periodic":
            out[:, -1] += wx_m[:, -1] * u[:, 0]
            out[:, 0] += wx_l[:, 0] * u[:, -1]
            out[-1, :] += wy_m[-1, :] * u[0, :]
            out[0, :] += wy_u[0, :] * u[-1, :]
        return out

    r_vec = b - apply_A(x, Ap)
    r_flat = r_vec.ravel()
    tol_sq = (rtol * rtol) * b_norm_sq
    if float(np.dot(r_flat, r_flat)) <= tol_sq:
        return x.astype(np.float64)

    z = inv_diag * r_vec
    p = z.copy()
    rz = float(np.dot(r_flat, z.ravel()))

    for _ in range(maxiter):
        apply_A(p, Ap)
        pAp = float(np.dot(p.ravel(), Ap.ravel()))
        if pAp <= 0.0:
            break
        alpha = np.float32(rz / pAp)
        x += alpha * p
        r_vec -= alpha * Ap
        if float(np.dot(r_flat, r_flat)) <= tol_sq:
            break
        np.multiply(inv_diag, r_vec, out=z)
        rz_new = float(np.dot(r_flat, z.ravel()))
        if rz == 0.0:
            break
        p *= np.float32(rz_new / rz)
        p += z
        rz = rz_new

    return x.astype(np.float64)


def _solve_direct(IN, wx, wy, lamda, boundary="neumann"):
    """Exact sparse LU solve for (I + lamda * L) x = IN."""
    r, c = IN.shape
    k = r * c
    wx_c = wx.copy()
    wy_c = wy.copy()

    if boundary == "neumann":
        wx_c[:, -1] = 0.0
        wy_c[-1, :] = 0.0
        dx = -lamda * wx_c.flatten("F")
        dy = -lamda * wy_c.flatten("F")
        dxa = -lamda * np.roll(wx_c, 1, axis=1).flatten("F")
        dya = -lamda * np.roll(wy_c, 1, axis=0).flatten("F")
        Ax = sparse.spdiags(dx, -r, k, k)
        Ay = sparse.spdiags(dy, -1, k, k)
    else:
        dx = -lamda * wx_c.flatten("F")
        dy = -lamda * wy_c.flatten("F")
        dxa = -lamda * np.roll(wx_c, 1, axis=1).flatten("F")
        dya = -lamda * np.roll(wy_c, 1, axis=0).flatten("F")
        tempx = np.concatenate((wx_c[:, -1:], np.zeros((r, c - 1))), axis=1)
        tempy = np.concatenate((wy_c[-1:, :], np.zeros((r - 1, c))), axis=0)
        dxd1 = -lamda * tempx.flatten("F")
        dyd1 = -lamda * tempy.flatten("F")
        wx_c[:, -1] = 0.0
        wy_c[-1, :] = 0.0
        dxd2 = -lamda * wx_c.flatten("F")
        dyd2 = -lamda * wy_c.flatten("F")
        Ax = sparse.spdiags(np.vstack((dxd1, dxd2)), np.array([-k + r, -r]), k, k)
        Ay = sparse.spdiags(np.vstack((dyd1, dyd2)), np.array([-r + 1, -1]), k, k)

    D = 1.0 - (dx + dy + dxa + dya)
    A = ((Ax + Ay) + (Ax + Ay).T + sparse.spdiags(D, 0, k, k)).T.tocsc()

    tout = spsolve(A, IN.flatten("F"))
    return np.reshape(tout, (r, c), order="F")


def solveLinearEquation(
    IN, wx, wy, lamda, solver="pcg", rtol=5e-3, maxiter=500, boundary="neumann"
):
    """Solve the weighted least squares smoothing linear system."""
    if boundary not in ("neumann", "periodic"):
        raise ValueError(
            f"Unsupported boundary: {boundary!r}. Expected 'neumann' or 'periodic'."
        )
    if solver == "pcg":
        return _solve_pcg(
            IN, wx, wy, lamda, rtol=rtol, maxiter=maxiter, boundary=boundary
        )
    if solver == "direct":
        return _solve_direct(IN, wx, wy, lamda, boundary=boundary)
    raise ValueError(f"Unsupported solver: {solver!r}. Expected 'pcg' or 'direct'.")


def tsmooth(
    img, lamda=0.01, sigma=3, sharpness=0.001, solver="pcg", boundary="neumann"
):
    """Smooth illumination map via relative total variation."""
    I = cv2.normalize(img.astype(np.float64), None, 0.0, 1.0, cv2.NORM_MINMAX)
    wx, wy = computeTextureWeights(I, sigma, sharpness, boundary=boundary)
    return solveLinearEquation(I, wx, wy, lamda, solver=solver, boundary=boundary)


def rgb2gm(I):
    """Compute per-pixel geometric mean across RGB channels."""
    if I.ndim == 3 and I.shape[2] == 3:
        I = cv2.normalize(I.astype(np.float64), None, 0.0, 1.0, cv2.NORM_MINMAX)
        return np.abs(I[:, :, 0] * I[:, :, 1] * I[:, :, 2]) ** (1.0 / 3.0)
    return I


def applyK(I, k, a=-0.3293, b=1.1258):
    """Apply BTF (Brightness Transform Function) with exposure ratio k."""
    beta = np.exp((1.0 - k**a) * b)
    gamma = k**a
    return (I**gamma) * beta


def entropy(X):
    """Compute Shannon entropy (in bits) of a normalized [0, 1] intensity array."""
    tmp = np.clip(X * 255.0, 0, 255).astype(np.uint8)
    counts = np.bincount(tmp.ravel(), minlength=256)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    pk = counts / counts.sum()
    return float(-np.sum(pk * np.log2(pk)))


def _estimate_exposure_ratio(I, isBad, a=-0.3293, b=1.1258, k_max=7.0):
    """Estimate optimal exposure ratio k maximizing entropy in under-exposed regions."""
    tmp = cv2.resize(I, (50, 50), interpolation=cv2.INTER_AREA)
    tmp = np.maximum(tmp.real, 0.0)
    Y = rgb2gm(tmp)

    isBad_small = imresize_bicubic(isBad.astype(np.float32), (50, 50))
    Y = Y[isBad_small >= 0.5]
    if Y.size == 0:
        return None

    return float(
        optimize.fminbound(lambda k: -entropy(applyK(Y, k, a, b)), 1.0, float(k_max))
    )


def maxEntropyEnhance(I, isBad, a=-0.3293, b=1.1258, k_max=7.0):
    """Enhance under-exposed regions by maximizing entropy over exposure ratio k."""
    opt_k = _estimate_exposure_ratio(I, isBad, a=a, b=b, k_max=k_max)
    if opt_k is None:
        return I
    return applyK(I, opt_k, a, b) - 0.01


def Ying_2017_CAIP(
    img,
    mu=0.5,
    a=-0.3293,
    b=1.1258,
    lamda=0.5,
    sigma=5,
    k_max=7.0,
    solver="pcg",
    boundary="neumann",
    max_smooth_size=480,
):
    """Enhance image contrast using Ying et al. (CAIP 2017) Exposure Fusion Framework."""
    if img.ndim == 2:
        has_alpha = False
        src = img
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        has_alpha = img.shape[2] == 4
        src = img[:, :, :3]
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # Fast 256-entry LUT path for uint8 inputs avoids full-image float pow in applyK
    norm_lut = None
    if src.dtype == np.uint8:
        vmin = int(src.min())
        vmax = int(src.max())
        if vmax > vmin:
            norm_lut = np.clip(
                (np.arange(256, dtype=np.float32) - vmin) / (vmax - vmin), 0.0, 1.0
            )
        else:
            norm_lut = np.zeros(256, dtype=np.float32)
        I = norm_lut[src]
        t_b = norm_lut[np.max(src, axis=2)] if src.ndim == 3 else I
    else:
        I = cv2.normalize(src.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
        t_b = np.max(I, axis=2) if I.ndim == 3 else I

    # Weight matrix estimation with optional cap on smoothing grid size
    scale = 0.5
    if max_smooth_size is not None and max_smooth_size > 0:
        scale = min(scale, max_smooth_size / max(t_b.shape))
    t_smooth = tsmooth(
        imresize_bicubic(t_b, scale), lamda, sigma, solver=solver, boundary=boundary
    )
    t_our = cv2.resize(
        t_smooth.astype(np.float32),
        (t_b.shape[1], t_b.shape[0]),
        interpolation=cv2.INTER_AREA,
    )

    # Apply camera model with optimal exposure ratio k
    isBad = t_our < 0.5
    if norm_lut is not None:
        opt_k = _estimate_exposure_ratio(I, isBad, a=a, b=b, k_max=k_max)
        if opt_k is None:
            J = I
        else:
            j_lut = (applyK(norm_lut, opt_k, a, b) - 0.01).astype(np.float32)
            J = j_lut[src]
    else:
        J = maxEntropyEnhance(I, isBad, a=a, b=b, k_max=k_max)

    # Exposure fusion weight matrix
    W = np.maximum(t_our, 0.0) ** mu
    if I.ndim == 3:
        W = W[:, :, None]

    result = np.clip((I * W + J * (1.0 - W)) * 255.0, 0, 255).astype(np.uint8)
    if has_alpha:
        return np.concatenate((result, img[:, :, 3:4].astype(np.uint8)), axis=2)
    return result


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
        "--lamda",
        type=float,
        default=0.5,
        help="Illumination smoothing regularization weight (default: 0.5)",
    )
    parser.add_argument(
        "--sigma",
        type=int,
        default=5,
        help="Texture window size for illumination smoothing (default: 5)",
    )
    parser.add_argument(
        "--k-max",
        type=float,
        default=7.0,
        help="Upper bound for optimal exposure ratio k search (default: 7.0)",
    )
    parser.add_argument(
        "--solver",
        choices=["pcg", "direct"],
        default="pcg",
        help="Linear solver for illumination smoothing (default: pcg)",
    )
    parser.add_argument(
        "--boundary",
        choices=["neumann", "periodic"],
        default="neumann",
        help="Boundary condition for illumination smoothing (default: neumann)",
    )
    parser.add_argument(
        "--max-smooth-size",
        type=int,
        default=480,
        help="Max long-edge size for illumination smoothing grid; 0 disables cap (default: 480)",
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
    max_smooth_size = args.max_smooth_size if args.max_smooth_size > 0 else None
    result = Ying_2017_CAIP(
        img,
        mu=args.mu,
        lamda=args.lamda,
        sigma=args.sigma,
        k_max=args.k_max,
        solver=args.solver,
        boundary=args.boundary,
        max_smooth_size=max_smooth_size,
    )

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
