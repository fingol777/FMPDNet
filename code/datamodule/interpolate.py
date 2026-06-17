import numpy as np
import torch
from typing import Tuple

def compute_lane_width(
        left_even_pts, right_even_pts
) -> float:
    if left_even_pts.shape != right_even_pts.shape:
         raise ValueError(
            f"Shape of left_even_pts {left_even_pts.shape} did not match right_even_pts {right_even_pts.shape}"
        )
    lane_width = float(np.mean(np.linalg.norm(left_even_pts - right_even_pts, axis=1)))
    return lane_width

def compute_midpoint_line_argoverse1(
    left_ln_boundary: np.ndarray,
    right_ln_boundary: np.ndarray,
    num_interp_pts: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the midpoint line and lane width from left and right lane boundaries for Argoverse 1.
    This function will:
    1. Interpolate points on both boundaries.
    2. Compute the centerline by averaging the left and right boundaries.
    3. Compute lane width by measuring the Euclidean distance between corresponding points on the left and right boundaries.

    Args:
        left_ln_boundary: Numpy array of shape (M,2) or (M,3), representing the left boundary of the lane
        right_ln_boundary: Numpy array of shape (N,2) or (N,3), representing the right boundary of the lane
        num_interp_pts: The number of points to interpolate along the lane.

    Returns:
        Tuple containing:
            - centerline_pts: Numpy array of shape (num_interp_pts, 2), representing the computed centerline.
            - lane_width: Numpy array of shape (num_interp_pts,), representing the lane width at each interpolated point.
    """
    
    # Check if the input arrays are 2D or 3D
    if left_ln_boundary.ndim != 2 or right_ln_boundary.ndim != 2:
        raise ValueError("Both left and right lane boundaries must be 2D or 3D arrays.")
    
    # If the boundaries have only 1 point, handle as a special case
    if len(left_ln_boundary) == 1:
        return compute_mid_pivot_arc(single_pt=left_ln_boundary, arc_pts=right_ln_boundary)
    
    if len(right_ln_boundary) == 1:
        return compute_mid_pivot_arc(single_pt=right_ln_boundary, arc_pts=left_ln_boundary)

    # Interpolate both the left and right boundaries
    left_interp = interp_arc(num_interp_pts, left_ln_boundary)
    right_interp = interp_arc(num_interp_pts, right_ln_boundary)

    # Compute the centerline as the midpoint between the left and right boundaries
    centerline_pts = (left_interp + right_interp) / 2.0

    # Compute lane width (Euclidean distance between the left and right boundary at each point)
    lane_width = compute_lane_width(left_interp, right_interp)

    return centerline_pts, lane_width

def interp_arc(t: int, points: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate equally spaced points along a polyline, either in 2d or 3d.
    
    Args:
        t: Number of points to interpolate.
        points: Numpy array of shape (N, 2) or (N, 3), representing 2D or 3D coordinates of the polyline.
    
    Returns:
        Numpy array of shape (t, 2), interpolated points along the polyline.
    """
    if points.ndim != 2:
        raise ValueError("Input array must be (N, 2) or (N, 3) in shape.")
    
    n, _ = points.shape

    if n < 2:
        if n == 1:
            return np.tile(points, (t, 1))
        else:
            raise ValueError("input is empty")

    eq_spaced_points = np.linspace(0, 1, t)
    
    # Compute the chordal length of each segment
    chordlen = np.linalg.norm(np.diff(points, axis=0), axis=1)

    eps = 1e-10
    chordlen = np.maximum(chordlen, eps)
    total_length = np.sum(chordlen)
    if total_length < eps:
       return np.linspace(points[0], points[-1], t)

    chordlen = chordlen / total_length  # Normalize to unit total length
    cumarc = np.zeros(len(chordlen) + 1)
    cumarc[1:] = np.cumsum(chordlen)
    
    # Find the interval each interpolated point falls into
    tbins = np.digitize(eq_spaced_points, bins=cumarc).astype(int)
    
    # Handle edge cases
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

    denominator = chordlen[tbins - 1]
    denominator = np.maximum(denominator, eps)
    
    s = (eq_spaced_points - cumarc[tbins - 1]) / denominator
    s = np.clip(s, 0, 1)

    anchors = points[tbins - 1, :]
    offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
    points_interp = anchors + offsets

    if np.isnan(points_interp).any():
        return np.linspace(points[0], points[-1], t)

    return points_interp

def compute_mid_pivot_arc(single_pt: np.ndarray, arc_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the midpoint for a single point and an arc (for cases where one boundary has only one point).
    
    Args:
        single_pt: A single 2D or 3D point (1, 2) or (1, 3).
        arc_pts: A 2D or 3D array of points defining an arc (N, 2) or (N, 3).
    
    Returns:
        Tuple containing:
            - centerline_pts: A single point as the midpoint (1, 2) or (1, 3).
            - lane_width: A scalar value representing the width.
    """
    # Compute the midpoint (just average with the arc_pts)
    centerline_pts = (arc_pts + single_pt) / 2.0
    
    # Compute width as the distance between the single point and the arc points
    lane_width = np.linalg.norm(arc_pts - single_pt, axis=1).mean()
    
    return centerline_pts, lane_width