#!/usr/bin/env python3
"""
Quickly try a few deblurring options without extra dependencies:
- Unsharp mask (fast)
- Wiener deconvolution using an estimated linear (motion) PSF
- Grid-search lengths/angles and score by Laplacian variance

Saves outputs next to the input image for inspection.
"""
import argparse
from pathlib import Path
import math
import cv2
import numpy as np


def unsharp(img, sigma=1.0, amount=1.0):
    img_f = img.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = np.clip(img_f + amount * (img_f - blur), 0.0, 1.0)
    return (sharp * 255.0).astype(np.uint8)


def motion_kernel(length, angle_deg):
    # create a horizontal line kernel then rotate
    L = max(3, int(length))
    # kernel size: make odd and a bit larger than length
    k = int(L) if int(L) % 2 == 1 else int(L) + 1
    size = k
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    # draw a horizontal line of ones of the specified length centered
    half = L // 2
    x0 = max(0, center - half)
    x1 = min(size, center + half + (1 if L % 2 else 0))
    kernel[center, x0:x1] = 1.0
    # rotate kernel by angle
    M = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (size, size), flags=cv2.INTER_CUBIC)
    s = kernel.sum()
    if s != 0:
        kernel /= s
    return kernel


def pad_psf_to_img(psf, shape):
    h, w = shape
    psf_padded = np.zeros((h, w), dtype=np.float32)
    kh, kw = psf.shape
    # place PSF at top-left corner shifted so center at (0,0) in freq domain
    # put center at (0,0) by placing psf center at (0,0) location
    cy, cx = kh // 2, kw // 2
    # target coords
    y0 = -cy
    x0 = -cx
    # we will place psf starting at (y0 mod h, x0 mod w)
    y0_mod = y0 % h
    x0_mod = x0 % w
    # copy
    y1 = min(h, y0_mod + kh)
    x1 = min(w, x0_mod + kw)
    psf_padded[y0_mod:y1, x0_mod:x1] = psf[0:(y1-y0_mod), 0:(x1-x0_mod)]
    return psf_padded


def wiener_deconv(img, psf, K=1e-3):
    # img: uint8 HxWxC or HxW
    # psf: small kernel, will be padded to img size
    if img.ndim == 3:
        channels = []
        for c in range(3):
            channels.append(wiener_deconv(img[..., c], psf, K))
        return np.stack(channels, axis=2)
    img_f = img.astype(np.float32)
    h, w = img_f.shape
    psf_padded = pad_psf_to_img(psf, (h, w))
    # FFTs
    G = np.fft.fft2(img_f)
    H = np.fft.fft2(psf_padded)
    H_conj = np.conj(H)
    denom = (H * H_conj) + K
    F_est = (H_conj / denom) * G
    f_est = np.fft.ifft2(F_est)
    f_est = np.real(f_est)
    # clamp
    f_est = np.clip(f_est, 0, 255)
    return f_est.astype(np.uint8)


def score_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    return var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', required=True)
    ap.add_argument('--out-dir', '-o', default=None)
    ap.add_argument('--wiener-K', type=float, default=5e-3)
    ap.add_argument('--lengths', type=str, default='5,9,15,25,35')
    ap.add_argument('--angles', type=str, default='0,30,60,90,120,150')
    ap.add_argument('--top-n', type=int, default=3)
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists():
        print('Input missing:', p)
        return
    out_dir = Path(args.out_dir) if args.out_dir else p.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(p))
    if img is None:
        print('Failed to read image')
        return

    # baseline unsharp
    us = unsharp(img, sigma=1.5, amount=1.2)
    us_path = out_dir / (p.stem + '_unsharp.jpg')
    cv2.imwrite(str(us_path), us)
    print('Wrote', us_path)

    lengths = [int(x) for x in args.lengths.split(',') if x.strip()]
    angles = [float(x) for x in args.angles.split(',') if x.strip()]

    results = []
    for L in lengths:
        for a in angles:
            psf = motion_kernel(L, a)
            try:
                out = wiener_deconv(img, psf, K=args.wiener_K)
            except Exception as e:
                print('Failed deconv', L, a, e)
                continue
            sc = score_sharpness(out)
            name = f"{p.stem}_wiener_L{L}_A{int(a)}_K{args.wiener_K}.jpg"
            path = out_dir / name
            cv2.imwrite(str(path), out)
            results.append((sc, path, L, a))
            print(f'Wrote {path} score={sc:.2f}')

    # sort by score desc
    results.sort(reverse=True, key=lambda x: x[0])
    top = results[: args.top_n]
    print('Top results:')
    for sc, path, L, a in top:
        print(f'  {path.name} score={sc:.2f} L={L} angle={a}')
    if top:
        best_path = top[0][1]
        best_copy = out_dir / (p.stem + '_best_deblur.jpg')
        cv2.imwrite(str(best_copy), cv2.imread(str(best_path)))
        print('Saved best ->', best_copy)

if __name__ == '__main__':
    main()
