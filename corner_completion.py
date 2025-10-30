#!/usr/bin/env python3
"""Utilities for completing missing container corners.

This module wraps the 3-corner vector completion logic used in the corner
annotation pipeline. It exposes a small helper class that can be reused by
other scripts without duplicating the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

Quadrant = str
Box = List[float]


def box_center(box: Iterable[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def assign_quadrant(cx: float, cy: float, width: int, height: int) -> Quadrant:
    """Classify a point into TL / TR / BR / BL by comparing to image midlines."""
    qx = "L" if cx < 0.5 * width else "R"
    qy = "T" if cy < 0.5 * height else "B"
    return f"{qy}{qx}"  # TL, TR, BR, BL


def _clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> List[int]:
    x1_i = max(0, int(round(x1)))
    y1_i = max(0, int(round(y1)))
    x2_i = min(width - 1, int(round(x2)))
    y2_i = min(height - 1, int(round(y2)))
    if x2_i <= x1_i:
        x2_i = min(width - 1, x1_i + 1)
    if y2_i <= y1_i:
        y2_i = min(height - 1, y1_i + 1)
    return [x1_i, y1_i, x2_i, y2_i]


def _make_box_from_center(cx: float, cy: float, bw: float, bh: float, width: int, height: int) -> List[int]:
    half_w = bw / 2.0
    half_h = bh / 2.0
    return _clamp_box(cx - half_w, cy - half_h, cx + half_w, cy + half_h, width, height)


def _pick_size_for_missing(wh_by_quad: Dict[Quadrant, Tuple[float, float]],
                           missing_q: Quadrant,
                           fallback_w: float,
                           fallback_h: float) -> Tuple[float, float]:
    """Choose width/height for the inferred corner based on neighbors."""
    if missing_q == "TL":
        bw = wh_by_quad.get("TR", (fallback_w,))[0]
        bh = wh_by_quad.get("BL", (fallback_h,))[1]
    elif missing_q == "TR":
        bw = wh_by_quad.get("TL", (fallback_w,))[0]
        bh = wh_by_quad.get("BR", (fallback_h,))[1]
    elif missing_q == "BR":
        bw = wh_by_quad.get("BL", (fallback_w,))[0]
        bh = wh_by_quad.get("TR", (fallback_h,))[1]
    else:  # "BL"
        bw = wh_by_quad.get("BR", (fallback_w,))[0]
        bh = wh_by_quad.get("TL", (fallback_h,))[1]
    return bw or fallback_w, bh or fallback_h


@dataclass
class CornerCompletionResult:
    quadrant: Quadrant
    box: List[int]
    inferred_class: Optional[str] = None  # 'gu_cor' or 'edge_cor' inferred from same-column corner


class VectorCornerCompleter:
    """Vector-based corner completion for the 3-corner detection case."""

    _FORMULAE: Dict[Quadrant, Tuple[Quadrant, Quadrant, Quadrant]] = {
        "TL": ("TR", "BL", "BR"),
        "TR": ("TL", "BR", "BL"),
        "BR": ("TR", "BL", "TL"),
        "BL": ("TL", "BR", "TR"),
    }

    @classmethod
    def infer_missing_three(cls,
                            boxes: List[Box],
                            image_size: Tuple[int, int],
                            classes: Optional[List[str]] = None) -> Optional[CornerCompletionResult]:
        """Infer the missing corner when exactly three quadrants are detected.

        Args:
            boxes: list of three bounding boxes [x1, y1, x2, y2].
            image_size: (width, height) of the image.
            classes: optional list of class names ('gu_cor' or 'edge_cor') for each box.

        Returns:
            CornerCompletionResult if a missing corner can be inferred, otherwise None.
        """
        if len(boxes) != 3:
            return None

        width, height = image_size
        centers = [np.asarray(box_center(b), dtype=np.float64) for b in boxes]
        whs = [
            (abs(b[2] - b[0]), abs(b[3] - b[1]))
            for b in boxes
        ]

        # Fallback sizes use medians to avoid extreme aspect ratios.
        fallback_w = float(np.median([w for w, _ in whs]) if whs else max(8, 0.05 * width))
        fallback_h = float(np.median([h for _, h in whs]) if whs else max(8, 0.05 * height))

        quads = [assign_quadrant(cx, cy, width, height) for cx, cy in centers]
        quad_to_center = {q: c for q, c in zip(quads, centers)}
        quad_to_wh = {q: (abs(b[2] - b[0]), abs(b[3] - b[1])) for q, b in zip(quads, boxes)}
        
        # Map quadrant to class if provided
        quad_to_class = {}
        if classes is not None and len(classes) == len(quads):
            quad_to_class = {q: c for q, c in zip(quads, classes)}

        missing = list({"TL", "TR", "BR", "BL"} - set(quads))
        if not missing:
            return None
        missing_q = missing[0]

        needed = cls._FORMULAE.get(missing_q)
        if needed is None:
            return None
        a_key, b_key, ref_key = needed

        if a_key not in quad_to_center or b_key not in quad_to_center or ref_key not in quad_to_center:
            return None

        center = quad_to_center[a_key] + quad_to_center[b_key] - quad_to_center[ref_key]
        cx = float(np.clip(center[0], 0.0, width - 1.0))
        cy = float(np.clip(center[1], 0.0, height - 1.0))

        bw, bh = _pick_size_for_missing(quad_to_wh, missing_q, fallback_w, fallback_h)
        box = _make_box_from_center(cx, cy, bw, bh, width, height)
        
        # Infer class from same-column corner:
        # TL/BL are left column, TR/BR are right column
        inferred_class = None
        if quad_to_class:
            if missing_q == "TL" and "BL" in quad_to_class:
                inferred_class = quad_to_class["BL"]  # same left column
            elif missing_q == "TR" and "BR" in quad_to_class:
                inferred_class = quad_to_class["BR"]  # same right column
            elif missing_q == "BR" and "TR" in quad_to_class:
                inferred_class = quad_to_class["TR"]  # same right column
            elif missing_q == "BL" and "TL" in quad_to_class:
                inferred_class = quad_to_class["TL"]  # same left column
        
        return CornerCompletionResult(quadrant=missing_q, box=box, inferred_class=inferred_class)


__all__ = [
    "box_center",
    "assign_quadrant",
    "CornerCompletionResult",
    "VectorCornerCompleter",
]
