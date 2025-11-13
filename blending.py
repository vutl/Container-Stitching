#!/usr/bin/env python3
"""Blending utilities for panorama stitching.

This module provides various blending strategies for combining overlapping
images in panoramic stitching pipelines, including:
- Feather blending: distance-based smooth alpha blending
- Seam carving: minimal-error seam-based blending
- None: simple overwrite (no blending)

Usage:
    blender = ImageBlender(mode='seam', seam_width=8)
    result = blender.blend(old_img, new_img, mask_old, mask_new)
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class BlendMode(str, Enum):
    """Supported blending modes."""
    FEATHER = 'feather'
    SEAM = 'seam'
    NONE = 'none'


@dataclass
class BlendResult:
    """Result of a blending operation."""
    blended: np.ndarray
    seam_mask: Optional[np.ndarray] = None  # For visualization/debugging


class ImageBlender:
    """Unified interface for image blending operations.
    
    This class encapsulates different blending strategies used in panoramic
    stitching, similar to how corner detection is organized in separate modules.
    
    Attributes:
        mode: Blending mode ('feather', 'seam', or 'none')
        seam_width: Width of the seam blend region (for 'seam' mode)
        feather_sigma: Gaussian blur sigma for feather blending
        min_seam_size: Minimum overlap size to use seam blending (fallback to feather)
    """
    
    def __init__(self,
                 mode: str = 'feather',
                 seam_width: int = 1,
                 feather_sigma: float = 6.0,
                 min_seam_size: int = 8):
        """Initialize blender with specified parameters.
        
        Args:
            mode: 'feather', 'seam', or 'none'
            seam_width: Width for seam blending smoothing
            feather_sigma: Sigma for Gaussian blur in feather mode
            min_seam_size: Minimum overlap dimensions for seam mode
        """
        self.mode = BlendMode(mode)
        self.seam_width = max(1, int(seam_width))
        self.feather_sigma = float(feather_sigma)
        self.min_seam_size = max(1, int(min_seam_size))
    
    def blend(self,
              old_img: np.ndarray,
              new_img: np.ndarray,
              mask_old: np.ndarray,
              mask_new: np.ndarray,
              overlap_bbox: Optional[Tuple[int, int, int, int]] = None) -> BlendResult:
        """Blend two images in their overlap region.
        
        Args:
            old_img: Existing mosaic/base image
            new_img: New image to blend in
            mask_old: Binary mask for old_img valid regions
            mask_new: Binary mask for new_img valid regions
            overlap_bbox: Optional (y0, y1, x0, x1) to restrict blending region
        
        Returns:
            BlendResult containing blended image and optional seam mask
        """
        if self.mode == BlendMode.NONE:
            return self._blend_none(old_img, new_img, mask_old, mask_new)
        elif self.mode == BlendMode.SEAM:
            return self._blend_seam(old_img, new_img, mask_old, mask_new, overlap_bbox)
        else:  # FEATHER
            return self._blend_feather(old_img, new_img, mask_old, mask_new, overlap_bbox)
    
    def _blend_none(self,
                    old_img: np.ndarray,
                    new_img: np.ndarray,
                    mask_old: np.ndarray,
                    mask_new: np.ndarray) -> BlendResult:
        """Simple overwrite: new image overwrites old in overlap regions."""
        result = old_img.copy()
        overlap = mask_new & mask_old
        only_new = mask_new & (~mask_old)
        
        if np.any(only_new):
            result[only_new] = new_img[only_new]
        if np.any(overlap):
            result[overlap] = new_img[overlap]
        
        return BlendResult(blended=result)
    
    def _blend_feather(self,
                       old_img: np.ndarray,
                       new_img: np.ndarray,
                       mask_old: np.ndarray,
                       mask_new: np.ndarray,
                       overlap_bbox: Optional[Tuple[int, int, int, int]] = None) -> BlendResult:
        """Distance-based feather blending using distance transforms."""
        result = old_img.copy()
        overlap = mask_new & mask_old
        only_new = mask_new & (~mask_old)
        
        if np.any(only_new):
            result[only_new] = new_img[only_new]
        
        if np.any(overlap):
            if overlap_bbox is not None:
                # Localized feather blending within bbox
                y0, y1, x0, x1 = overlap_bbox
                result[y0:y1, x0:x1] = self._feather_region(
                    old_img[y0:y1, x0:x1],
                    new_img[y0:y1, x0:x1],
                    mask_old[y0:y1, x0:x1].astype(np.uint8),
                    mask_new[y0:y1, x0:x1].astype(np.uint8),
                    overlap[y0:y1, x0:x1]
                )
            else:
                # Full-frame feather blending
                mask_old_u8 = mask_old.astype(np.uint8)
                mask_new_u8 = mask_new.astype(np.uint8)
                dist_old = cv2.distanceTransform(mask_old_u8, cv2.DIST_L2, 5).astype(np.float32)
                dist_new = cv2.distanceTransform(mask_new_u8, cv2.DIST_L2, 5).astype(np.float32)
                dist_old = cv2.GaussianBlur(dist_old, (0, 0), sigmaX=self.feather_sigma, sigmaY=self.feather_sigma)
                dist_new = cv2.GaussianBlur(dist_new, (0, 0), sigmaX=self.feather_sigma, sigmaY=self.feather_sigma)
                
                denom = dist_old + dist_new + 1e-6
                w_new = dist_new / denom
                w_old = 1.0 - w_new
                
                blended = (old_img.astype(np.float32) * w_old[..., None] +
                          new_img.astype(np.float32) * w_new[..., None])
                result[overlap] = np.clip(blended, 0, 255).astype(np.uint8)[overlap]
        
        return BlendResult(blended=result)
    
    def _feather_region(self,
                        old_sub: np.ndarray,
                        new_sub: np.ndarray,
                        mask_old_sub: np.ndarray,
                        mask_new_sub: np.ndarray,
                        overlap_sub: np.ndarray) -> np.ndarray:
        """Apply feather blending to a subregion."""
        result = old_sub.copy()
        
        dist_old = cv2.distanceTransform(mask_old_sub, cv2.DIST_L2, 5).astype(np.float32)
        dist_new = cv2.distanceTransform(mask_new_sub, cv2.DIST_L2, 5).astype(np.float32)
        dist_old = cv2.GaussianBlur(dist_old, (0, 0), sigmaX=self.feather_sigma, sigmaY=self.feather_sigma)
        dist_new = cv2.GaussianBlur(dist_new, (0, 0), sigmaX=self.feather_sigma, sigmaY=self.feather_sigma)
        
        denom = dist_old + dist_new + 1e-6
        w_new = dist_new / denom
        w_old = 1.0 - w_new
        
        blended = (old_sub.astype(np.float32) * w_old[..., None] +
                  new_sub.astype(np.float32) * w_new[..., None])
        result[overlap_sub] = np.clip(blended, 0, 255).astype(np.uint8)[overlap_sub]
        
        return result
    
    def _blend_seam(self,
                    old_img: np.ndarray,
                    new_img: np.ndarray,
                    mask_old: np.ndarray,
                    mask_new: np.ndarray,
                    overlap_bbox: Optional[Tuple[int, int, int, int]] = None) -> BlendResult:
        """Minimal-error seam carving based blending."""
        result = old_img.copy()
        overlap = mask_new & mask_old
        only_new = mask_new & (~mask_old)
        
        if np.any(only_new):
            result[only_new] = new_img[only_new]
        
        seam_mask_full = None
        
        if np.any(overlap):
            # Find overlap bounding box if not provided
            if overlap_bbox is None:
                ys, xs = np.where(overlap)
                if len(ys) == 0:
                    return BlendResult(blended=result)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
            else:
                y0, y1, x0, x1 = overlap_bbox
            
            subA = old_img[y0:y1, x0:x1]
            subB = new_img[y0:y1, x0:x1]
            mask_sub = overlap[y0:y1, x0:x1]
            
            # Check if region is too small for seam carving
            if (x1 - x0) < self.min_seam_size or (y1 - y0) < self.min_seam_size:
                # Fallback to feather blending
                result[y0:y1, x0:x1] = self._feather_region(
                    subA, subB,
                    mask_old[y0:y1, x0:x1].astype(np.uint8),
                    mask_new[y0:y1, x0:x1].astype(np.uint8),
                    mask_sub
                )
            else:
                # Compute vertical minimal-error seam
                seam_mask = self._compute_vertical_seam(subA, subB).astype(np.uint8)
                
                # Create full-size seam mask for debugging
                seam_mask_full = np.zeros(old_img.shape[:2], dtype=np.uint8)
                seam_mask_full[y0:y1, x0:x1] = (seam_mask * 255).astype(np.uint8)
                
                # Keep only inside overlap
                seam_mask = (seam_mask & mask_sub.astype(np.uint8))
                
                # Create smoothed alpha mask
                fmask = seam_mask.astype(np.float32)
                if self.seam_width > 1:
                    kx = self.seam_width * 2 + 1
                    # Minimal horizontal blur to soften hard seam edge
                    fmask = cv2.GaussianBlur(fmask, (kx, 1), sigmaX=self.seam_width * 0.5)
                    # Normalize to [0,1]
                    maxv = float(fmask.max()) if fmask.max() > 0 else 1.0
                    fmask = fmask / maxv
                else:
                    # Binary -> 0/1
                    fmask = (fmask > 0).astype(np.float32)
                
                # Blend using soft alpha mask
                alpha = fmask[..., None]
                subA_f = subA.astype(np.float32)
                subB_f = subB.astype(np.float32)
                blended = (subA_f * (1.0 - alpha) + subB_f * alpha).astype(np.uint8)
                
                sub_out = result[y0:y1, x0:x1].copy()
                sub_out[mask_sub] = blended[mask_sub]
                result[y0:y1, x0:x1] = sub_out
        
        return BlendResult(blended=result, seam_mask=seam_mask_full)
    
    @staticmethod
    def _compute_vertical_seam(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
        """Compute vertical minimal-error seam mask.
        
        Uses dynamic programming to find the path through the overlap region
        that minimizes color differences between the two images.
        
        Args:
            img_a: First image (BGR)
            img_b: Second image (BGR)
        
        Returns:
            Binary mask (uint8) with 1 where img_b should be used
        """
        # Use grayscale absolute difference as cost
        ag = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        bg = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(ag, bg).astype(np.float32) + 1.0  # avoid zeros
        h, w = diff.shape
        
        if w <= 2 or h <= 2:
            return np.ones((h, w), dtype=np.uint8)
        
        # Dynamic programming for minimal cost path
        dp = np.zeros_like(diff)
        back = np.zeros((h, w), dtype=np.int16)
        dp[0, :] = diff[0, :]
        
        for r in range(1, h):
            prev = dp[r - 1]
            for c in range(w):
                # Consider three predecessors: c-1, c, c+1
                c0 = prev[c]
                c1 = prev[c - 1] if c - 1 >= 0 else 1e9
                c2 = prev[c + 1] if c + 1 < w else 1e9
                
                if c1 <= c0 and c1 <= c2:
                    dp[r, c] = diff[r, c] + c1
                    back[r, c] = c - 1
                elif c0 <= c1 and c0 <= c2:
                    dp[r, c] = diff[r, c] + c0
                    back[r, c] = c
                else:
                    dp[r, c] = diff[r, c] + c2
                    back[r, c] = c + 1
        
        # Backtrack to find seam path
        seam = np.zeros(h, dtype=np.int32)
        seam[h - 1] = int(np.argmin(dp[h - 1]))
        for r in range(h - 2, -1, -1):
            seam[r] = int(back[r + 1, seam[r + 1]])
        
        # Build mask: use img_b to the right of seam
        mask = np.zeros((h, w), dtype=np.uint8)
        for r in range(h):
            c = seam[r]
            if c + 1 < w:
                mask[r, c + 1:] = 1
        
        return mask
    
    def save_debug_images(self,
                         result: BlendResult,
                         old_img: np.ndarray,
                         new_img: np.ndarray,
                         debug_dir: Path,
                         step_name: str = 'blend'):
        """Save debug visualization images.
        
        Args:
            result: BlendResult from blend operation
            old_img: Original old image
            new_img: Original new image
            debug_dir: Directory to save debug images
            step_name: Prefix for debug filenames
        """
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Save input images
            cv2.imwrite(str(debug_dir / f"{step_name}_old.png"), old_img)
            cv2.imwrite(str(debug_dir / f"{step_name}_new.png"), new_img)
            
            # Save blended result
            cv2.imwrite(str(debug_dir / f"{step_name}_blended.png"), result.blended)
            
            # Save seam mask if available
            if result.seam_mask is not None:
                cv2.imwrite(str(debug_dir / f"{step_name}_seam_mask.png"), result.seam_mask)
        except Exception as e:
            print(f"Warning: Failed to save debug images: {e}")


__all__ = [
    'BlendMode',
    'BlendResult',
    'ImageBlender',
]
