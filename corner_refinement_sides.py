#!/usr/bin/env python3
"""Gradient-based corner refinement utilities for side-face images.

This mirrors `corner_refinement.py` but exposes an optional `force_head`
per-box flag in `refine_corners` so callers can force a head-style corner
placement (using the existing helper) without changing the global logic used
for top-face images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import math

from corner_completion import assign_quadrant, box_center


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CornerCandidate:
    quadrant: str
    point: Tuple[int, int]
    confidence: float
    method: str
    box: Tuple[int, int, int, int]


@dataclass(frozen=True)
class CornerRefinerParams:
    side_margin_frac: float = 0.12
    center_window_frac: float = 0.45
    center_near_px: int = 4
    center_thr_alpha: float = 0.5
    min_side_px: int = 6
    sobel_ksize: int = 3
    perc: float = 90.0
    perc_alpha: float = 0.45
    std_mult: float = 1.0
    refine_half_w: int = 6
    refine_half_h: int = 6
    refine_search_h: int = 8
    refine_search_w: int = 8
    min_accept_score: float = 10.0
    container_head_area_thresh: float = 2000.0
    container_head_offset: int = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _dynamic_threshold(arr: np.ndarray,
                       pct: float,
                       pct_alpha: float,
                       std_mult: float) -> float:
    flat = np.asarray(arr, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0
    med = float(np.median(flat))
    std = float(np.std(flat))
    perc_v = float(np.percentile(flat, pct))
    return max(perc_v * pct_alpha, med + std_mult * std)


def _weighted_centroid_1d(vals: np.ndarray, coords: Sequence[int]) -> float:
    weights = np.asarray(vals, dtype=np.float64)
    if weights.sum() <= 0:
        return float(coords[len(coords) // 2])
    return float((weights * np.asarray(coords, dtype=np.float64)).sum() / weights.sum())


def _estimate_global_bounds(img: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nonblack = gray > 12
    r0 = int(max(0, min(h - 1, round(h * 0.18))))
    r1 = int(max(r0 + 1, min(h, round(h * 0.88))))
    col_frac = nonblack[r0:r1, :].mean(axis=0) if r1 > r0 else nonblack.mean(axis=0)
    try:
        xl = int(next(i for i, v in enumerate(col_frac) if v >= 0.98))
        xr = int(len(col_frac) - 1 - next(i for i, v in enumerate(col_frac[::-1]) if v >= 0.98))
    except StopIteration:
        xl, xr = int(0.1 * w), int(0.9 * w)
    y_top = int(_clamp(h * 0.18, 0, h - 1))
    y_bottom = int(_clamp(h * 0.88, 0, h - 1))
    return xl, xr, y_top, y_bottom


def _refine_row_from_col(gy: np.ndarray,
                         bw: int,
                         bh: int,
                         col: int,
                         approx_row: int,
                         params: CornerRefinerParams,
                         search_h: int) -> Tuple[int, float]:
    c0 = int(round(max(0, col - params.refine_half_w)))
    c1 = int(round(min(bw, col + params.refine_half_w + 1)))
    strip = gy[:, c0:c1]
    if strip.size == 0:
        return approx_row, 0.0
    r0 = int(max(0, approx_row - search_h))
    r1 = int(min(bh, approx_row + search_h + 1))
    local = strip[r0:r1, :]
    if local.size == 0:
        return approx_row, 0.0
    col_sum = local.sum(axis=1)
    idx = int(np.argmax(col_sum))
    val = float(col_sum[idx])
    start = max(0, idx - 2)
    end = min(col_sum.size, idx + 3)
    win = col_sum[start:end]
    coords = list(range(r0 + start, r0 + start + win.size))
    sub_r = int(round(_weighted_centroid_1d(win, coords)))
    return sub_r, val


def _refine_col_from_row(gx: np.ndarray,
                         bw: int,
                         bh: int,
                         row: int,
                         approx_col: int,
                         params: CornerRefinerParams,
                         search_w: int) -> Tuple[float, float]:
    r0 = int(round(max(0, row - params.refine_half_h)))
    r1 = int(round(min(bh, row + params.refine_half_h + 1)))
    strip = gx[r0:r1, :]
    if strip.size == 0:
        return approx_col, 0.0
    col_sum = strip.sum(axis=0)
    idx = int(np.argmax(col_sum))
    val = float(col_sum[idx])
    win_c0 = max(0, idx - 2)
    win_c1 = min(col_sum.size, idx + 3)
    win = col_sum[win_c0:win_c1]
    coords = list(range(win_c0, win_c0 + win.size))
    sub_c = _weighted_centroid_1d(win, coords)
    return sub_c, val


def _handle_container_head(box: Tuple[int, int, int, int],
                           quadrant: str,
                           params: CornerRefinerParams) -> Optional[CornerCandidate]:
    x1, y1, x2, y2 = box
    offset = params.container_head_offset
    if quadrant == "TR":
        return CornerCandidate("TR", (x2, y1 + offset), 1.0, "container_head_TR_extend", box)
    if quadrant == "BR":
        return CornerCandidate("BR", (x2, y2 - offset), 1.0, "container_head_BR_extend", box)
    if quadrant == "TL":
        return CornerCandidate("TL", (x1, y1 + offset), 1.0, "container_head_TL_extend", box)
    if quadrant == "BL":
        return CornerCandidate("BL", (x1, y2 - offset), 1.0, "container_head_BL_extend", box)
    return None


def _find_corner_in_box(img: np.ndarray,
                        box: Tuple[int, int, int, int],
                        params: CornerRefinerParams,
                        quadrant: str,
                        force_head: bool = False) -> CornerCandidate:
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    bh, bw = crop.shape[:2]

    # If caller requested a forced "head" placement for this box, use the
    # helper to compute the head-style corner and return immediately. This
    # mirrors the previous behavior that was triggered by bbox area but
    # allows callers to opt-in per-box without affecting top-face logic.
    if force_head:
        head = _handle_container_head(box, quadrant, params)
        if head is not None:
            return head

    if bh <= 2 or bw <= 2:
        cx, cy = box_center(box)
        pt = (int(round(cx)), int(round(cy)))
        return CornerCandidate(quadrant, pt, 0.0, "bbox_fallback", box)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    k = params.sobel_ksize
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k))

    cx_rel = bw // 2
    cy_rel = bh // 2
    r = max(1, int(params.center_near_px))
    c0 = max(0, cx_rel - r)
    c1 = min(bw, cx_rel + r + 1)
    r0 = max(0, cy_rel - r)
    r1 = min(bh, cy_rel + r + 1)

    whole_col = gx.sum(axis=0)
    whole_row = gy.sum(axis=1)
    thr_c_whole = _dynamic_threshold(whole_col, params.perc, params.perc_alpha, params.std_mult)
    thr_r_whole = _dynamic_threshold(whole_row, params.perc, params.perc_alpha, params.std_mult)

    # NOTE: Previously we returned early when the center window had strong
    # gradients ("center_local"). That can pull regular (non-head) corners
    # slightly inward. We now defer this check until after side-edge search
    # and global fallbacks. This keeps regular corners hugging the silhouette.

    side_px = max(int(params.side_margin_frac * bw), params.min_side_px)
    side_px = min(side_px, max(1, bw // 2))
    center_r0 = int(round(bh * (0.5 - params.center_window_frac / 2.0)))
    center_r1 = int(round(bh * (0.5 + params.center_window_frac / 2.0)))
    center_r0 = int(_clamp(center_r0, 0, bh - 1))
    center_r1 = max(center_r0 + 1, int(_clamp(center_r1, 0, bh)))

    left_gx = gx[center_r0:center_r1, 0:side_px] if side_px > 0 else np.zeros((0, 0))
    right_gx = gx[center_r0:center_r1, max(0, bw - side_px):bw] if side_px > 0 else np.zeros((0, 0))

    thr_left = _dynamic_threshold(left_gx, params.perc, params.perc_alpha, params.std_mult)
    thr_right = _dynamic_threshold(right_gx, params.perc, params.perc_alpha, params.std_mult)

    best_left = None
    if left_gx.size > 0:
        idx = np.unravel_index(np.argmax(left_gx), left_gx.shape)
        val = float(left_gx[idx])
        if val >= thr_left and val > 0:
            row = center_r0 + idx[0]
            col = idx[1]
            best_left = (col, row, val)

    best_right = None
    if right_gx.size > 0:
        idx = np.unravel_index(np.argmax(right_gx), right_gx.shape)
        val = float(right_gx[idx])
        if val >= thr_right and val > 0:
            row = center_r0 + idx[0]
            col = max(0, bw - side_px) + idx[1]
            best_right = (col, row, val)

    chosen_point: Optional[Tuple[float, float]] = None
    chosen_score: float = 0.0
    chosen_method: Optional[str] = None

    # Prefer the vertical side that matches the quadrant (left for TL/BL,
    # right for TR/BR). We'll apply a small bias to the preferred side.
    prefer_left = quadrant in ("TL", "BL")
    bias_factor = 1.07  # ~7% nudge towards the expected side

    if best_left is not None:
        col_l, row_l, val_l = best_left
        row_refined, gy_val = _refine_row_from_col(gy, bw, bh, col_l, row_l, params, params.refine_search_h)
        col_refined, gx_val = _refine_col_from_row(gx, bw, bh, row_refined, col_l, params, params.refine_search_w)
        x_abs = x1 + col_refined
        y_abs = y1 + row_refined
        score = (val_l + gy_val + gx_val) / 3.0
        if prefer_left:
            score *= bias_factor
        chosen_point = (x_abs, y_abs)
        chosen_score = score
        chosen_method = "gradient_left"

    if best_right is not None:
        col_r, row_r, val_r = best_right
        row_refined, gy_val = _refine_row_from_col(gy, bw, bh, col_r, row_r, params, params.refine_search_h)
        col_refined, gx_val = _refine_col_from_row(gx, bw, bh, row_refined, col_r, params, params.refine_search_w)
        x_abs = x1 + col_refined
        y_abs = y1 + row_refined
        score = (val_r + gy_val + gx_val) / 3.0
        if not prefer_left:
            score *= bias_factor
        if chosen_point is None or score > chosen_score:
            chosen_point = (x_abs, y_abs)
            chosen_score = score
            chosen_method = "gradient_right"

    # Optional row boost: if a significantly stronger horizontal edge (gy)
    # exists near the bbox, snap the row to that edge and re-refine the
    # column on that row. This avoids keeping a slightly too-high/low y
    # chosen purely from the side-edge search.
    if chosen_point is not None:
        row_sum_total = gy.sum(axis=1)
        if row_sum_total.size:
            r_idx = int(np.argmax(row_sum_total))
            thr_r_boost = _dynamic_threshold(row_sum_total, params.perc, params.perc_alpha, params.std_mult)
            cur_r = int(round(chosen_point[1] - y1))
            if 0 <= cur_r < bh and row_sum_total[r_idx] >= thr_r_boost:
                cur_strength = float(row_sum_total[cur_r])
                # Require a modest improvement to avoid oscillation
                if row_sum_total[r_idx] > cur_strength * 1.05:
                    col_seed = int(round(chosen_point[0] - x1))
                    col_refined2, gx_val2 = _refine_col_from_row(gx, bw, bh, r_idx, col_seed, params, params.refine_search_w)
                    chosen_point = (x1 + col_refined2, y1 + r_idx)
                    # Update score conservatively using boosted row strength
                    chosen_score = max(chosen_score, float((gx_val2 + row_sum_total[r_idx]) / 2.0))
                    chosen_method = (chosen_method or "gradient_combo") + "+row_boost"

    # If side-edge search did not yield a confident point, try a
    # quadrant-centered local search to intersect strong vertical/horizontal
    # responses near the bbox center.
    if chosen_point is None:
        lc0 = max(0, cx_rel - params.center_near_px)
        lc1 = min(bw, cx_rel + params.center_near_px + 1)
        lr0 = max(0, cy_rel - params.center_near_px)
        lr1 = min(bh, cy_rel + params.center_near_px + 1)
        if lc1 > lc0 and lr1 > lr0:
            local_col_sum = gx[lr0:lr1, lc0:lc1].sum(axis=0)
            local_row_sum = gy[lr0:lr1, lc0:lc1].sum(axis=1)
            if local_col_sum.size and local_row_sum.size:
                lc_idx = int(np.argmax(local_col_sum)) + lc0
                lr_idx = int(np.argmax(local_row_sum)) + lr0
                thr_c = thr_c_whole * params.center_thr_alpha
                thr_r = thr_r_whole * params.center_thr_alpha
                if gx[:, lc_idx].sum() >= thr_c and gy[lr_idx, :].sum() >= thr_r:
                    col_refined, _ = _refine_col_from_row(gx, bw, bh, lr_idx, lc_idx, params, params.refine_search_w)
                    row_refined, _ = _refine_row_from_col(gy, bw, bh, lc_idx, lr_idx, params, params.refine_search_h)
                    chosen_point = (x1 + col_refined, y1 + row_refined)
                    chosen_score = float((gx[:, lc_idx].sum() + gy[lr_idx, :].sum()) / 2.0)
                    chosen_method = "center_near_search"

    # Try a global max of vertical/horizontal energy within the box.
    if chosen_point is None or chosen_score < params.min_accept_score:
        col_sum_total = gx.sum(axis=0)
        row_sum_total = gy.sum(axis=1)
        if col_sum_total.size and row_sum_total.size:
            c_idx = int(np.argmax(col_sum_total))
            r_idx = int(np.argmax(row_sum_total))
            thr_c = _dynamic_threshold(col_sum_total, params.perc, params.perc_alpha, params.std_mult)
            thr_r = _dynamic_threshold(row_sum_total, params.perc, params.perc_alpha, params.std_mult)
            if col_sum_total[c_idx] >= thr_c and row_sum_total[r_idx] >= thr_r:
                col_refined, _ = _refine_col_from_row(gx, bw, bh, r_idx, c_idx, params, params.refine_search_w)
                row_refined, _ = _refine_row_from_col(gy, bw, bh, c_idx, r_idx, params, params.refine_search_h)
                chosen_point = (x1 + col_refined, y1 + row_refined)
                chosen_score = float((col_sum_total[c_idx] + row_sum_total[r_idx]) / 2.0)
                chosen_method = "gradient_global"

    # Finally, consider the center-local response as a last resort.
    if chosen_point is None:
        center_gx = gx[r0:r1, c0:c1]
        center_gy = gy[r0:r1, c0:c1]
        sum_cx_center = float(center_gx.sum()) if center_gx.size else 0.0
        sum_gy_center = float(center_gy.sum()) if center_gy.size else 0.0
        center_thr_c = thr_c_whole * params.center_thr_alpha
        center_thr_r = thr_r_whole * params.center_thr_alpha
        if sum_cx_center >= center_thr_c and sum_gy_center >= center_thr_r:
            cols = np.arange(c0, c1)
            rows = np.arange(r0, r1)
            col_weights = center_gx.sum(axis=0)
            row_weights = center_gy.sum(axis=1)
            sub_c = _weighted_centroid_1d(col_weights, cols) if cols.size else cx_rel
            sub_r = _weighted_centroid_1d(row_weights, rows) if rows.size else cy_rel
            x_abs = x1 + sub_c
            y_abs = y1 + sub_r
            conf = float(1.0 - math.exp(-0.0005 * ((sum_cx_center + sum_gy_center) / 2.0)))
            chosen_point = (x_abs, y_abs)
            chosen_score = max(sum_cx_center, sum_gy_center)
            chosen_method = "center_local"

    if chosen_point is None:
        cx_box, cy_box = box_center(box)
        pt = (int(round(cx_box)), int(round(cy_box)))
        return CornerCandidate(quadrant, pt, 0.0, "bbox_fallback", box)

    conf = float(1.0 - math.exp(-0.0005 * max(chosen_score, 0.0))) if chosen_score > 0 else 0.0

    # Subpixel snap using cornerSubPix to reduce small inward drift.
    # We refine on the local crop using the predicted point as the seed.
    px_f = float(chosen_point[0] - x1)
    py_f = float(chosen_point[1] - y1)
    # Ensure seed lies inside the crop (cornerSubPix requires valid interior point)
    if 1 <= px_f < (bw - 1) and 1 <= py_f < (bh - 1):
        seed = np.array([[[px_f, py_f]]], dtype=np.float32)
        try:
            cv2.cornerSubPix(
                gray,
                seed,
                winSize=(3, 3),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01),
            )
            sub_x = float(seed[0, 0, 0])
            sub_y = float(seed[0, 0, 1])
            chosen_point = (x1 + sub_x, y1 + sub_y)
            if chosen_method is not None:
                chosen_method = f"{chosen_method}+subpix"
        except Exception:
            pass

    pt = (int(round(chosen_point[0])), int(round(chosen_point[1])))
    return CornerCandidate(quadrant, pt, conf, chosen_method or "gradient_combo", box)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def refine_corners(img: np.ndarray,
                   boxes: Iterable[Sequence[float]],
                   params: Optional[CornerRefinerParams] = None,
                   quadrants: Optional[Iterable[str]] = None,
                   force_head: Optional[Iterable[bool]] = None) -> List[CornerCandidate]:
    """Refine corners for a list of bounding boxes.

    Args:
        img: Source image (BGR).
        boxes: Iterable of boxes [x1, y1, x2, y2].
        params: Optional override parameters.
        quadrants: Optional iterable of quadrant labels matching ``boxes``.
        force_head: Optional iterable of booleans matching ``boxes``. When
            True for a box, a head-style corner will be returned immediately
            using the helper routine (useful for side-face images where the
            detector already indicates head-like boxes).

    Returns:
        List of :class:`CornerCandidate`, in the same order as ``boxes``.
    """

    boxes_list = [tuple(int(round(v)) for v in box) for box in boxes]
    if not boxes_list:
        return []

    params = params or CornerRefinerParams()
    h, w = img.shape[:2]

    if quadrants is not None:
        quad_list = list(quadrants)
    else:
        quad_list = [assign_quadrant(*box_center(box), w, h) for box in boxes_list]

    force_list: Optional[List[bool]] = None
    if force_head is not None:
        force_list = list(force_head)
        if len(force_list) != len(boxes_list):
            raise ValueError("force_head iterable must match number of boxes")

    candidates: List[CornerCandidate] = []
    for i, (box, quad) in enumerate(zip(boxes_list, quad_list)):
        quad_valid = quad if quad in {"TL", "TR", "BR", "BL"} else assign_quadrant(*box_center(box), w, h)
        fh = False
        if force_list is not None:
            fh = bool(force_list[i])
        cand = _find_corner_in_box(img, box, params, quad_valid, force_head=fh)
        candidates.append(cand)
    return candidates


__all__ = [
    "CornerCandidate",
    "CornerRefinerParams",
    "refine_corners",
]
