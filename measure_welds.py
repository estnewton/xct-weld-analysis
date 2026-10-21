"""
Automated measurement pipeline for X-CT laser weld images.
Extracts: Bonded Width (mm) and Depth D (mm) per image.

Scale: 1.37 µm/pixel  |  Upper steel nominal: 1.5 mm

Coordinate system
-----------------
  Origin (0,0) : LEFT BW endpoint on the interface line
  x-axis       : unit vector from left → right BW endpoint (along interface)
  y-axis       : perpendicular to x-axis, positive y pointing UP toward top surface

BW definition  : distance along x-axis between left and right BW endpoints
                 (endpoints = centroids of first dark object on each side of weld centre)

D  definition  : perpendicular distance from the deepest teal-line dip
                 (between BW endpoints) DOWN to the x-axis (interface line)
                 D makes an exact 90° angle with BW.

L  definition  : concavity depth (top surface sag at BW midpoint vs. flat-top ref)
                 kept in CSV for D4 flag only
"""

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, csv

# ── Constants ──────────────────────────────────────────────────────────────
SCALE_UM         = 1.37          # µm per pixel
SCALE_MM         = SCALE_UM / 1000
UPPER_SHEET_MM   = 1.5           # nominal upper steel thickness (mm)
UPPER_SHEET_PX   = int(UPPER_SHEET_MM / SCALE_MM)   # ≈ 1095 px
SMOOTH_COL       = 20            # pixel smoothing along a column
PORE_THRESH_FRAC = 0.05          # iface_val < bulk * this  → dark object (void/pore/gap)

# ── Helper: load and return float64 array ─────────────────────────────────
def load_image(path):
    return np.array(Image.open(path), dtype=np.float64)

# ── Step 1: find the bottom of the material ───────────────────────────────
def find_bottom_surface(arr, bulk_val):
    """Scan from bottom upward per column; requires SUSTAIN consecutive px above threshold."""
    H, W = arr.shape
    threshold = bulk_val * 0.35
    SUSTAIN = 40
    bottom = np.full(W, H - 1, dtype=int)
    for c in range(W):
        col = uniform_filter1d(arr[:, c].astype(float), size=SMOOTH_COL)
        run = 0
        for r in range(H - 1, H // 2, -1):
            if col[r] > threshold:
                run += 1
                if run >= SUSTAIN:
                    bottom[c] = r + SUSTAIN
                    break
            else:
                run = 0
    return bottom

# ── Step 2: find the top surface ──────────────────────────────────────────
def find_top_surface(arr, bulk_val):
    """Scan from top downward per column; requires SUSTAIN consecutive px above threshold."""
    H, W = arr.shape
    threshold = bulk_val * 0.35
    SUSTAIN = 40
    top = np.full(W, H, dtype=int)   # H = invalid sentinel
    for c in range(W):
        col = uniform_filter1d(arr[:, c].astype(float), size=SMOOTH_COL)
        run = 0
        for r in range(0, H // 2):
            if col[r] > threshold:
                run += 1
                if run >= SUSTAIN:
                    top[c] = r - SUSTAIN
                    break
            else:
                run = 0
    return top

# ── Step 3: flat-top reference row ────────────────────────────────────────
def flat_top_reference(top_surface, valid_mask):
    """5th percentile of top surface in central 50% of width = flat-top reference."""
    W = len(top_surface)
    cx0, cx1 = W // 4, 3 * W // 4
    central = valid_mask.copy()
    central[:cx0] = False
    central[cx1:] = False
    vals = top_surface[central]
    if len(vals) == 0:
        return 0
    return int(np.percentile(vals, 5))

# ── Step 4: find interface dark line per column ───────────────────────────
def find_interface(arr, top_surface, bottom_surface, valid_mask):
    """
    Per-column: search for the darkest point in 70–130% of nominal steel depth
    below the local top surface.  Returns iface_row[W] and iface_val[W].
    """
    H, W = arr.shape
    FRAC_LOW, FRAC_HIGH = 0.70, 1.30

    iface_row = np.zeros(W, dtype=int)
    iface_val = np.full(W, np.nan)

    for c in range(W):
        if not valid_mask[c]:
            continue
        r0 = int(top_surface[c] + FRAC_LOW  * UPPER_SHEET_PX)
        r1 = int(top_surface[c] + FRAC_HIGH * UPPER_SHEET_PX)
        r0 = max(r0, top_surface[c] + 30)
        r1 = min(r1, bottom_surface[c] - 30)
        if r1 <= r0:
            continue
        col = uniform_filter1d(arr[:, c].astype(float), size=SMOOTH_COL)
        zone = col[r0:r1]
        min_idx = int(np.argmin(zone))
        iface_row[c] = min_idx + r0
        iface_val[c] = zone[min_idx]

    return iface_row, iface_val

# ── Step 5: find weld centre column ───────────────────────────────────────
def find_weld_center(top_surface, valid_mask):
    """
    Column of the deepest top-surface concavity in the central 50% of image width.
    This is the laser-weld centreline.
    """
    W = len(top_surface)
    cx0, cx1 = W // 4, 3 * W // 4
    search = valid_mask.copy()
    search[:cx0] = False
    search[cx1:] = False
    if search.sum() == 0:
        return W // 2
    cols = np.where(search)[0]
    # 95th-percentile row = deepest point
    deepest_row = int(np.percentile(top_surface[search], 95))
    center_col  = int(cols[np.argmin(np.abs(top_surface[cols] - deepest_row))])
    return center_col

# ── Step 6: find dark-object endpoints on each side of weld centre ────────
def find_bw_endpoints(iface_val, valid_mask, weld_center, bulk_val):
    """
    A dark object is identified as a run of ≥ MIN_RUN consecutive columns where
    iface_val < PORE_THRESH_FRAC * bulk.  These can be pores, bars, or gaps.

    LEFT  endpoint = OUTERMOST (leftmost) dark object left of weld centre.
    RIGHT endpoint = INNERMOST (closest to centre) dark object right of weld centre.

    Returns centroid column of each detected object.

    Returns
    -------
    left_col   : int or None
    right_col  : int or None
    left_found : bool
    right_found: bool
    """
    W           = len(iface_val)
    pore_thr    = bulk_val * PORE_THRESH_FRAC
    SEARCH_HALF = 1200   # max columns to search from centre
    MIN_RUN     = 5      # consecutive dark-level columns needed

    iface_finite = np.where(np.isnan(iface_val), bulk_val, iface_val)
    is_dark = valid_mask & (iface_finite < pore_thr)

    def collect_runs(c_start, c_end):
        """Return list of (centroid_col, run_len) for all dark runs in [c_start, c_end)."""
        runs = []
        run_start = None
        for c in range(c_start, c_end):
            if is_dark[c]:
                if run_start is None:
                    run_start = c
            else:
                if run_start is not None:
                    length = c - run_start
                    if length >= MIN_RUN:
                        centroid = (run_start + c - 1) // 2
                        runs.append((centroid, length))
                    run_start = None
        # Handle run that extends to boundary
        if run_start is not None:
            length = c_end - run_start
            if length >= MIN_RUN:
                centroid = (run_start + c_end - 1) // 2
                runs.append((centroid, length))
        return runs

    # ── LEFT side: scan LEFT-to-RIGHT, take the OUTERMOST (leftmost) run ──
    left_limit  = max(0, weld_center - SEARCH_HALF)
    left_runs   = collect_runs(left_limit, weld_center)
    left_col, left_found = None, False
    if left_runs:
        left_col, _ = min(left_runs, key=lambda r: r[0])   # leftmost centroid
        left_found   = True
    print(f"  Left dark-obj runs : {[(c, l) for c, l in left_runs]} "
          f"→ chosen col={left_col}")

    # ── RIGHT side: scan LEFT-to-RIGHT, take the INNERMOST (leftmost) run ─
    right_limit  = min(W, weld_center + SEARCH_HALF)
    right_runs   = collect_runs(weld_center + 1, right_limit)
    right_col, right_found = None, False
    if right_runs:
        right_col, _ = min(right_runs, key=lambda r: r[0])  # leftmost = closest to centre
        right_found   = True
    print(f"  Right dark-obj runs: {[(c, l) for c, l in right_runs]} "
          f"→ chosen col={right_col}")

    return left_col, right_col, left_found, right_found


# ── Step 6b: RANSAC-based interface detection (robust to ring artifacts) ──
def find_interface_ransac(arr, top_surface, bottom_surface, valid_mask,
                          bulk_val, weld_center):
    """
    Robust interface detection using RANSAC line fitting + gap detection.

    The steel-Al interface appears as two dark horizontal segments separated
    by a gap (the bonded region):
      LEFT segment  |  GAP (bonded)  |  RIGHT segment

    Algorithm
    ---------
    1. Per-column darkest point in 70-130% steel zone  (find_interface)
    2. Select candidate columns: dark enough (< 50% bulk) and away from edges
    3. RANSAC line fit:  iface_row ≈ slope * col + intercept
    4. Sanity check: |angle| ≤ 5°
    5. Identify largest gap > 200 px in RANSAC inlier columns near weld centre
    6. Left BW endpoint = rightmost inlier before the gap (inner tip of left segment)
       Right BW endpoint = leftmost inlier after the gap (inner tip of right segment)
    7. Endpoint sanity checks: ≥ EDGE_MARGIN px from image edges

    Falls back to the original find_bw_endpoints() when RANSAC cannot find a
    clear gap (e.g., images without a visible unbonded interface).

    Returns
    -------
    iface_row   : ndarray[W, int]  – per-column interface row
    iface_val   : ndarray[W, float] – per-column interface intensity
    left_col    : int or None
    right_col   : int or None
    left_found  : bool
    right_found : bool
    """
    H, W = arr.shape

    # Always compute per-column interface using the column-minimum method
    iface_row, iface_val = find_interface(arr, top_surface, bottom_surface, valid_mask)

    def _fallback(reason=''):
        if reason:
            print(f"  RANSAC: {reason} — original endpoint detection")
        lc, rc, lf, rf = find_bw_endpoints(iface_val, valid_mask, weld_center, bulk_val)
        return iface_row, iface_val, lc, rc, lf, rf

    # ─── Candidate columns ────────────────────────────────────────────────
    # Interface dark line is clearly darker than bonded material but not void-level.
    # Exclude columns within EDGE_MARGIN of the left/right image boundary so that
    # the sample-edge darkness cannot become a BW endpoint.
    EDGE_MARGIN   = 100    # pixels
    IFACE_MAX_VAL = 0.50   # interface dark line < 50 % of bulk

    col_idx   = np.arange(W)
    cand_mask = (valid_mask
                 & ~np.isnan(iface_val)
                 & (iface_val < bulk_val * IFACE_MAX_VAL)
                 & (col_idx >= EDGE_MARGIN)
                 & (col_idx <  W - EDGE_MARGIN))
    cand_cols = np.where(cand_mask)[0]

    if len(cand_cols) < 30:
        return _fallback(f'only {len(cand_cols)} candidates (< 30)')

    # ─── Pure-NumPy RANSAC line fit ───────────────────────────────────────
    # Fit:  iface_row ≈ slope * col + intercept
    cx = cand_cols.astype(np.float64)
    cy = iface_row[cand_cols].astype(np.float64)

    RESID_THR  = 30.0    # ± 30 px ≈ ± 0.04 mm
    N_TRIALS   = 300
    rng        = np.random.default_rng(42)

    best_slope, best_intercept, best_n = 0.0, float(np.median(cy)), 0
    n = len(cx)

    for _ in range(N_TRIALS):
        i, j = rng.choice(n, size=2, replace=False)
        dx = cx[i] - cx[j]
        if abs(dx) < 1.0:
            continue
        s = (cy[i] - cy[j]) / dx
        b = cy[i] - s * cx[i]
        resid = np.abs(cy - (s * cx + b))
        n_in  = int((resid <= RESID_THR).sum())
        if n_in > best_n:
            best_n, best_slope, best_intercept = n_in, s, b

    # Refit using all inliers
    inl = np.abs(cy - (best_slope * cx + best_intercept)) <= RESID_THR
    if inl.sum() >= 2:
        inl_x = cx[inl]; inl_y = cy[inl]
        A = np.column_stack([inl_x, np.ones_like(inl_x)])
        best_slope, best_intercept = np.linalg.lstsq(A, inl_y, rcond=None)[0]
        inl = np.abs(cy - (best_slope * cx + best_intercept)) <= RESID_THR

    slope      = float(best_slope)
    intercept  = float(best_intercept)
    n_inliers  = int(inl.sum())
    angle_deg  = float(np.degrees(np.arctan(slope)))

    print(f"  RANSAC: angle={angle_deg:+.1f}°  row0={intercept:.0f}  "
          f"inliers={n_inliers}/{len(cand_cols)}")

    # ─── Angle sanity check ───────────────────────────────────────────────
    if abs(angle_deg) > 5.0:
        return _fallback(f'|angle|={abs(angle_deg):.1f}° > 5°')

    # ─── Gap detection via raw pixel values at the fitted RANSAC row ─────
    # For each column, sample arr at the single predicted interface row.
    # This avoids SMOOTH_COL=20 diluting thin interface segments (recon_00541).
    # Threshold: strict 5% of bulk (same as find_bw_endpoints) to reject the
    # bonded aluminide layer, which appears bright at the interface row.
    # Gap selection: largest gap > MIN_GAP_PX among those whose centre is
    # within NEAR_CENTRE of weld_center.  "Largest" is more robust than
    # "nearest-to-centre" when a small artefact gap sits closer to weld_center.
    MIN_RUN     = 5      # consecutive dark-at-row pixels to form a run
    MIN_GAP_PX  = 300    # bonded region must be ≥ 300 px (≈ 0.41 mm)
    NEAR_CENTRE = 1200   # gap centre within this many px of weld_center

    pore_thr = bulk_val * PORE_THRESH_FRAC   # same strict threshold as find_bw_endpoints

    col_range_arr = np.arange(W, dtype=int)
    pred_rows_arr = np.clip(
        np.round(slope * col_range_arr.astype(np.float64) + intercept).astype(int),
        0, H - 1)

    # Dark-at-fitted-row mask (no smoothing, strict threshold)
    dark_row = np.zeros(W, dtype=bool)
    for c in range(EDGE_MARGIN, W - EDGE_MARGIN):
        if valid_mask[c]:
            dark_row[c] = arr[pred_rows_arr[c], c] < pore_thr

    # Collect all dark runs (col_start, col_end_inclusive, length)
    dark_runs = []
    run_start = None
    for c in range(EDGE_MARGIN, W - EDGE_MARGIN + 1):
        in_dark = (c < W - EDGE_MARGIN) and dark_row[c]
        if in_dark:
            if run_start is None:
                run_start = c
        else:
            if run_start is not None:
                length = c - run_start
                if length >= MIN_RUN:
                    dark_runs.append((run_start, c - 1, length))
                run_start = None

    print(f"  Dark-at-row runs (≥{MIN_RUN}px): {dark_runs}")

    if len(dark_runs) < 2:
        return _fallback(f'fewer than 2 dark-at-row runs ({len(dark_runs)})')

    # Find all gaps between consecutive dark runs that meet MIN_GAP_PX and NEAR_CENTRE
    best_gap_left  = None
    best_gap_right = None
    best_gap_width = 0

    for i in range(len(dark_runs) - 1):
        gap_left  = dark_runs[i][1]           # last dark col before gap
        gap_right = dark_runs[i + 1][0]       # first dark col after gap
        gap_width = gap_right - gap_left - 1
        if gap_width < MIN_GAP_PX:
            continue
        gap_centre = (gap_left + gap_right) / 2.0
        if abs(gap_centre - weld_center) > NEAR_CENTRE:
            continue
        if gap_width > best_gap_width:
            best_gap_width = gap_width
            best_gap_left  = gap_left
            best_gap_right = gap_right

    if best_gap_left is None:
        max_near = max(
            (dark_runs[i + 1][0] - dark_runs[i][1] - 1
             for i in range(len(dark_runs) - 1)
             if abs(((dark_runs[i][1] + dark_runs[i+1][0]) / 2) - weld_center) <= NEAR_CENTRE),
            default=0)
        return _fallback(
            f'no dark-at-row gap ≥ {MIN_GAP_PX} px near centre '
            f'(largest qualifying gap: {max_near} px)')

    left_col  = best_gap_left
    right_col = best_gap_right
    gap_px    = right_col - left_col - 1

    print(f"  Gap (dark-at-row): col {left_col} → {right_col}  "
          f"({gap_px} px = {gap_px * SCALE_MM:.3f} mm BW estimate)")

    # ─── Set endpoint rows from the fitted line ────────────────────────────
    def _pred_row(c):
        return int(np.clip(round(slope * c + intercept), 0, H - 1))

    iface_row[left_col]  = _pred_row(left_col)
    iface_row[right_col] = _pred_row(right_col)

    # ─── Endpoint edge check ──────────────────────────────────────────────
    left_found  = (left_col  >= EDGE_MARGIN)
    right_found = (right_col <  W - EDGE_MARGIN)

    if not left_found:
        print(f"  ⚠ Left endpoint col={left_col} < {EDGE_MARGIN} (too close to edge)")
    if not right_found:
        print(f"  ⚠ Right endpoint col={right_col} ≥ {W - EDGE_MARGIN} (too close to edge)")

    lc = left_col  if left_found  else None
    rc = right_col if right_found else None
    return iface_row, iface_val, lc, rc, left_found, right_found


# ── Step 7: BW + D in interface coordinate system ────────────────────────
def compute_bw_d(left_col, right_col, iface_row, top_surface, valid_mask,
                 flat_top):
    """
    Build the interface coordinate system and compute BW and D.

    Coordinate system
    -----------------
      x-axis : unit vector left_endpoint → right_endpoint
      y-axis : rotate x-axis by 90°, flip to point toward top surface (upward)
      origin : left BW endpoint

    BW = distance between the two endpoints along x-axis (mm)
    D  = y-coordinate (in local system) of the deepest top-surface dip
         between the two BW endpoints  (mm)
    L  = concavity depth at BW midpoint vs flat_top reference (mm, for D4 flag)

    Returns a dict with all geometry.
    """
    if left_col is None or right_col is None:
        return {
            'bw_mm': 0.0, 'D_mm': 0.0, 'L_mm': 0.0,
            'left_col': left_col, 'left_row': -1,
            'right_col': right_col, 'right_row': -1,
            'origin_col': -1.0, 'origin_row': -1.0,
            'u_col': 1.0, 'u_row': 0.0,
            'n_col': 0.0, 'n_row': -1.0,
            'dip_col': -1.0, 'dip_row': -1.0,
            'foot_col': -1.0, 'foot_row': -1.0,
            'x_dip_mm': 0.0, 'y_dip_mm': 0.0,
            'mid_col': -1,
        }

    # BW endpoint pixel coords
    x1, y1 = float(left_col),  float(iface_row[left_col])   # origin
    x2, y2 = float(right_col), float(iface_row[right_col])  # right end

    dx     = x2 - x1
    dy     = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-6:
        length = 1.0

    # x-axis unit vector (left → right along interface)
    u_col = dx / length   # x component (pixel col direction)
    u_row = dy / length   # y component (pixel row direction, + = down)

    # y-axis: rotate x-axis by 90°  →  n = (-u_y, u_x)
    n_col = -u_row
    n_row =  u_col
    # Flip if n points downward (n_row > 0 means pointing toward larger row = downward)
    if n_row > 0:
        n_col = -n_col
        n_row = -n_row

    # Verify perpendicularity
    dot = u_col * n_col + u_row * n_row
    print(f"  u · n  = {dot:.2e}  (must be 0)")

    bw_mm = length * SCALE_MM
    mid_col = (left_col + right_col) // 2

    # ── Find deepest dip of top surface between BW endpoints ──────────────
    W_img = len(valid_mask)
    min_y_local = np.inf
    best_c = -1
    for c in range(W_img):
        if not valid_mask[c]:
            continue
        dc = float(c) - x1
        dr = float(top_surface[c]) - y1
        x_local = dc * u_col + dr * u_row   # along x-axis
        y_local = dc * n_col + dr * n_row   # perpendicular (D direction)
        if x_local < 0.0 or x_local > length:
            continue
        if y_local < min_y_local:
            min_y_local = y_local
            best_c = c

    if best_c < 0:
        best_c = mid_col

    dip_col = float(best_c)
    dip_row = float(top_surface[best_c])

    # Project dip point onto local coordinate system
    dc = dip_col - x1
    dr = dip_row - y1
    x_dip = dc * u_col + dr * u_row   # position along x-axis
    y_dip = dc * n_col + dr * n_row   # D in pixels (positive = above interface)

    D_mm = y_dip * SCALE_MM  # should be positive (top surface above interface)

    # Foot of D on x-axis (drop perpendicularly from dip to x-axis)
    foot_col = x1 + x_dip * u_col
    foot_row = y1 + x_dip * u_row

    # L: concavity depth at midpoint vs flat_top
    use_mid = mid_col if valid_mask[mid_col] else best_c
    L_mm = (top_surface[use_mid] - flat_top) * SCALE_MM

    # ── Print verification ─────────────────────────────────────────────────
    print(f"  Left BW endpoint : pixel col={left_col}, row={int(y1)}"
          f"  →  x/y = (0.000, 0.000) mm")
    print(f"  Right BW endpoint: pixel col={right_col}, row={int(y2)}"
          f"  →  x/y = ({bw_mm:.3f}, 0.000) mm")
    print(f"  BW               : {bw_mm:.4f} mm")
    print(f"  Teal dip         : pixel col={best_c}, row={int(dip_row)}"
          f"  →  x={x_dip*SCALE_MM:.3f} mm, y={y_dip*SCALE_MM:.3f} mm")
    print(f"  Foot of D        : pixel col={foot_col:.1f}, row={foot_row:.1f}"
          f"  →  x={x_dip*SCALE_MM:.3f} mm, y=0.000 mm")
    print(f"  D                : {D_mm:.4f} mm")
    print(f"  Angle D vs x-axis: 90.0°  (by construction)")
    print(f"  L                : {L_mm:.4f} mm")
    print(f"  u (x-axis)       : ({u_col:+.6f}, {u_row:+.6f})")
    print(f"  n (y-axis)       : ({n_col:+.6f}, {n_row:+.6f})")

    return {
        'bw_mm': bw_mm,
        'D_mm':  D_mm,
        'L_mm':  L_mm,
        'left_col': left_col,  'left_row':  int(round(y1)),
        'right_col': right_col, 'right_row': int(round(y2)),
        'origin_col': x1,      'origin_row': y1,
        'u_col': u_col,        'u_row': u_row,
        'n_col': n_col,        'n_row': n_row,
        'dip_col': dip_col,    'dip_row': dip_row,
        'foot_col': foot_col,  'foot_row': foot_row,
        'x_dip_mm': x_dip * SCALE_MM,
        'y_dip_mm': y_dip * SCALE_MM,
        'mid_col': mid_col,
    }


# ── Helper: draw one axis with arrow, label, and mm tick marks ─────────────
def _draw_axis_with_ticks(ax, ox, oy, u_col, u_row, perp_col, perp_row,
                          axis_len_px, color, label_txt,
                          tick_spacing_mm=0.5, tick_half_len=35,
                          ext_neg_px=250, ext_pos_px=350):
    """
    Draw a dashed labelled axis with mm tick marks and an arrowhead at the tip.

    Parameters
    ----------
    ox, oy          : origin in pixel coords (col, row)
    u_col, u_row    : unit vector along this axis (pixel col, row)
    perp_col/row    : unit vector perpendicular to this axis (for tick marks)
    axis_len_px     : length of the positive half of the axis in pixels
    color           : line / tick / label color
    label_txt       : 'x' or 'y'
    tick_spacing_mm : distance between ticks (mm)
    tick_half_len   : half-length of each tick mark in pixels
    ext_neg_px      : how far to extend the axis behind the origin (pixels)
    ext_pos_px      : how far to extend beyond axis_len_px (pixels)
    """
    ppm = 1.0 / SCALE_MM  # pixels per mm

    # Start and tip of axis
    x0 = ox - ext_neg_px * u_col
    y0 = oy - ext_neg_px * u_row
    x1 = ox + (axis_len_px + ext_pos_px) * u_col
    y1 = oy + (axis_len_px + ext_pos_px) * u_row

    # ── Dashed axis line (stops a bit short of tip so arrowhead is clean) ──
    arrow_back = 20   # pixels before tip where the dashed line ends
    ax.plot([x0, x1 - arrow_back * u_col],
            [y0, y1 - arrow_back * u_row],
            color=color, lw=1.5, linestyle='--', alpha=0.85, zorder=4,
            solid_capstyle='butt')

    # ── Arrowhead at positive tip ─────────────────────────────────────────
    ax.annotate('', xy=(x1, y1),
                xytext=(x1 - arrow_back * u_col, y1 - arrow_back * u_row),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                zorder=4)

    # ── Tick marks at integer multiples of tick_spacing_mm ─────────────────
    t = 0.0
    tick_step_px = tick_spacing_mm * ppm
    while t <= axis_len_px + tick_half_len:
        tx = ox + t * u_col
        ty = oy + t * u_row
        ax.plot([tx - tick_half_len * perp_col, tx + tick_half_len * perp_col],
                [ty - tick_half_len * perp_row, ty + tick_half_len * perp_row],
                color=color, lw=1.0, alpha=0.8, zorder=4)
        if t > 0:
            lbl = f'{t * SCALE_MM:.1f}'
            lbl_x = tx + (tick_half_len + 18) * perp_col
            lbl_y = ty + (tick_half_len + 18) * perp_row
            ax.text(lbl_x, lbl_y, lbl, color=color, fontsize=6,
                    ha='center', va='center', alpha=0.85, zorder=4)
        t += tick_step_px

    # ── Axis label ('x' or 'y') at the arrowhead ──────────────────────────
    lbl_off = 60   # pixels beyond arrowhead
    ax.text(x1 + lbl_off * u_col, y1 + lbl_off * u_row,
            label_txt, color=color, fontsize=13, fontweight='bold',
            ha='center', va='center', zorder=5)


# ── Step 8: annotated visualisation ──────────────────────────────────────
def annotate_image(arr, top_surface, iface_row, iface_val, valid_mask,
                   flat_top, geo,
                   flag_false_friend, flag_insuff_pen, flag_concav_L, flag_concav_D,
                   flag_burn_through,
                   weld_center, filename, out_path):
    """
    Draw the annotated result image with:
      - x/y coordinate axes with mm tick marks (origin = left BW endpoint)
      - BW as thick colored line between endpoints, circles at endpoints
      - D as thick arrow from teal-dip point down to x-axis
      - right-angle marker where D meets x-axis
      - defect labels and measurement box
    """
    H, W = arr.shape
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    disp = np.clip((arr - lo) / (hi - lo), 0, 1)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(disp, cmap='gray')
    ax.set_aspect('equal')   # pixels are square → 90° looks correct

    # ── Top surface profile (teal) ─────────────────────────────────────────
    cols = np.where(valid_mask)[0]
    ax.scatter(cols, top_surface[cols], s=0.4, c='cyan', alpha=0.6,
               label='Top surface')

    # ── Interface profile (yellow) ─────────────────────────────────────────
    valid_iface = valid_mask & ~np.isnan(iface_val)
    ax.scatter(np.where(valid_iface)[0], iface_row[valid_iface],
               s=0.4, c='yellow', alpha=0.6, label='Interface (steel/Al)')

    # ── Flat-top reference line (dashed lime) ─────────────────────────────
    ax.axhline(flat_top, color='lime', linewidth=1.0, linestyle='--', alpha=0.6,
               label=f'Flat top ref (row {flat_top})')

    # ── Weld centre marker ────────────────────────────────────────────────
    ax.axvline(weld_center, color='white', linewidth=0.8, linestyle=':',
               alpha=0.4, label=f'Weld centre (col {weld_center})')

    bw_mm = geo['bw_mm']
    D_mm  = geo['D_mm']
    L_mm  = geo['L_mm']

    endpoints_found = geo['left_col'] is not None and geo['right_col'] is not None

    if endpoints_found:
        ox   = geo['origin_col']   # left endpoint col (float)
        oy   = geo['origin_row']   # left endpoint row (float)
        u_col = geo['u_col']       # x-axis unit col
        u_row = geo['u_row']       # x-axis unit row
        n_col = geo['n_col']       # y-axis unit col
        n_row = geo['n_row']       # y-axis unit row

        right_col = geo['right_col']
        right_row = geo['right_row']

        dip_col  = geo['dip_col']
        dip_row  = geo['dip_row']
        foot_col = geo['foot_col']
        foot_row = geo['foot_row']

        bw_px = np.sqrt((right_col - ox)**2 + (right_row - oy)**2)
        D_px  = geo['y_dip_mm'] / SCALE_MM   # pixels

        MAGENTA = '#FF00FF'

        # ── x-axis: dashed magenta, spanning most of sample width ─────────
        x_ext_neg = max(250, int(ox) - 150)
        x_ext_pos = max(350, W - int(ox + bw_px) - 150)
        _draw_axis_with_ticks(ax, ox, oy, u_col, u_row, n_col, n_row,
                              axis_len_px=bw_px, color=MAGENTA,
                              label_txt='x', tick_spacing_mm=0.5,
                              tick_half_len=30,
                              ext_neg_px=x_ext_neg, ext_pos_px=x_ext_pos)

        # ── y-axis: dashed magenta, past top surface and to image bottom ──
        y_axis_len = max(D_px * 1.2, float(oy - flat_top) + 250)
        y_ext_neg = int(H - oy) + 50   # extend down to bottom of image
        _draw_axis_with_ticks(ax, ox, oy, n_col, n_row, u_col, u_row,
                              axis_len_px=y_axis_len, color=MAGENTA,
                              label_txt='y', tick_spacing_mm=0.5,
                              tick_half_len=30, ext_neg_px=y_ext_neg, ext_pos_px=150)

        # ── Right-angle marker at origin (where x and y axes meet) ────────
        sq_o = 40
        Ax_o = ox + sq_o * u_col;    Ay_o = oy + sq_o * u_row
        Bx_o = ox + sq_o * n_col;    By_o = oy + sq_o * n_row
        Cx_o = Ax_o + sq_o * n_col;  Cy_o = Ay_o + sq_o * n_row
        ax.plot([Ax_o, Cx_o, Bx_o], [Ay_o, Cy_o, By_o],
                color=MAGENTA, linewidth=1.5, alpha=0.9, zorder=6)

        # ── Origin marker and "(0, 0)" label ──────────────────────────────
        ax.plot(ox, oy, 'o', color=MAGENTA, markersize=8,
                markerfacecolor=MAGENTA, zorder=7)
        # Label offset: to the left along x-axis and below (−y direction)
        lbl_off_col = -90 * u_col - 55 * n_col
        lbl_off_row = -90 * u_row - 55 * n_row
        ax.text(ox + lbl_off_col, oy + lbl_off_row,
                '(0, 0)', color=MAGENTA, fontsize=9, fontweight='bold',
                ha='center', va='center', zorder=7,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                          edgecolor=MAGENTA, linewidth=0.8, alpha=0.7))

        # ── BW: thick orange line along x-axis between endpoints ──────────
        ax.plot([ox, right_col], [oy, right_row],
                color='orange', linewidth=4, alpha=0.85, zorder=8,
                label=f'BW = {bw_mm:.3f} mm')

        # Endpoint circles
        ax.plot(ox, oy, 'o', color='orange',
                markersize=14, markerfacecolor='none', markeredgewidth=2.5,
                zorder=9, label='BW endpoints')
        ax.plot(right_col, right_row, 'o', color='orange',
                markersize=14, markerfacecolor='none', markeredgewidth=2.5,
                zorder=9)

        # BW label above midpoint of the BW line
        mid_bw_col = (ox + right_col) / 2
        mid_bw_row = (oy + right_row) / 2
        offset = 80   # pixels above the line
        ax.text(mid_bw_col + offset * n_col, mid_bw_row + offset * n_row,
                f'BW = {bw_mm:.3f} mm', color='orange', fontsize=11,
                fontweight='bold', ha='center', va='center', zorder=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                          edgecolor='orange', linewidth=1, alpha=0.7))

        # ── D: thick red arrow from teal dip DOWN to foot on x-axis ──────
        ax.annotate('', xy=(foot_col, foot_row), xytext=(dip_col, dip_row),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3.0),
                    zorder=8)

        # Dot at teal dip
        ax.plot(dip_col, dip_row, 'o', color='red', markersize=8,
                markerfacecolor='red', zorder=9, label=f'D = {D_mm:.3f} mm')

        # Dot at foot
        ax.plot(foot_col, foot_row, 'o', color='red', markersize=7,
                markerfacecolor='white', markeredgecolor='red',
                markeredgewidth=2, zorder=9)

        # D label to the right of the arrow midpoint
        mid_d_col = (dip_col + foot_col) / 2
        mid_d_row = (dip_row + foot_row) / 2
        ax.text(mid_d_col + 80, mid_d_row,
                f'D = {D_mm:.3f} mm', color='red', fontsize=11,
                fontweight='bold', ha='left', va='center', zorder=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                          edgecolor='red', linewidth=1, alpha=0.7))

        # ── Right-angle marker where D meets x-axis (foot), in red ────────
        sq = 28
        Ax = foot_col + sq * u_col;   Ay = foot_row + sq * u_row
        Bx = foot_col + sq * n_col;   By = foot_row + sq * n_row
        Cx = Ax + sq * n_col;         Cy = Ay + sq * n_row
        ax.plot([Ax, Cx, Bx], [Ay, Cy, By],
                color='red', linewidth=1.5, alpha=0.9, zorder=8)

    # ── Defect label box ──────────────────────────────────────────────────
    defect_lines = []
    if flag_false_friend:
        defect_lines.append(('D1  FALSE FRIEND', '#FF4444'))
    if flag_insuff_pen and not flag_false_friend:
        defect_lines.append(('D2  INSUFFICIENT PENETRATION', '#FF8800'))
    if flag_concav_L or flag_concav_D:
        defect_lines.append(('D4  CONCAVITY', '#FF44FF'))
    if flag_burn_through:
        defect_lines.append(('D3  BURN-THROUGH', '#FF0000'))

    if defect_lines:
        box_x, box_y = 30, flat_top + 50
        for i, (label_txt, color) in enumerate(defect_lines):
            ax.text(box_x, box_y + i * 70, f'⚠ {label_txt}',
                    color=color, fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                              edgecolor=color, linewidth=2, alpha=0.85))
    else:
        ax.text(30, flat_top + 50, '✓  NO DEFECTS DETECTED',
                color='#44FF44', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                          edgecolor='#44FF44', linewidth=2, alpha=0.85))

    # ── BW-endpoint-not-found warnings ───────────────────────────────────
    warn_y = flat_top + 200
    if not endpoints_found or geo['left_col'] is None:
        ax.text(30, warn_y, '⚠ No dark object found LEFT of centre',
                color='yellow', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        warn_y += 50
    if not endpoints_found or geo['right_col'] is None:
        ax.text(30, warn_y, '⚠ No dark object found RIGHT of centre',
                color='yellow', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # ── Measurement box (bottom-right) ────────────────────────────────────
    meas_txt = (f"BW = {bw_mm:.3f} mm\n"
                f"D  = {D_mm:.3f} mm\n"
                f"L  = {L_mm:.3f} mm")
    ax.text(W - 30, H - 30, meas_txt,
            color='white', fontsize=11, fontweight='bold',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#111111',
                      edgecolor='white', linewidth=1.5, alpha=0.9))

    title = (f"{filename}  |  BW={bw_mm:.3f} mm  |  D={D_mm:.3f} mm  |  L={L_mm:.3f} mm")
    ax.set_title(title, fontsize=11)

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color='#FF00FF', lw=1.5, linestyle='--',
               label='x-axis (baseline)'),
        Line2D([0], [0], color='#FF00FF', lw=1.5, linestyle='--',
               label='y-axis (perpendicular)'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=8, markerscale=4,
              facecolor='black', labelcolor='white')
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


# ── Main pipeline ─────────────────────────────────────────────────────────
def process_image(path, out_dir):
    fname = os.path.basename(path)
    print(f"\n{'='*60}")
    print(f"Processing: {fname}")

    arr = load_image(path)
    H, W = arr.shape

    # Estimate bulk material value from a safe interior region
    bulk_val = float(np.median(arr[H // 3:2 * H // 3, W // 4:3 * W // 4]))
    print(f"  Bulk intensity : {bulk_val:.4f}")

    # ── Surface detection ─────────────────────────────────────────────────
    top_surface    = find_top_surface(arr, bulk_val)
    bottom_surface = find_bottom_surface(arr, bulk_val)

    # Valid columns: top surface found and geometry looks reasonable
    raw_valid  = (top_surface < H * 0.45) & (top_surface > 20)
    mat_height = bottom_surface - top_surface
    height_ok  = (mat_height > 1500) & (mat_height < 3100)
    valid      = raw_valid & height_ok

    # ── Flat-top reference ────────────────────────────────────────────────
    flat_top = flat_top_reference(top_surface, valid)
    print(f"  Flat top ref   : row {flat_top}  ({flat_top * SCALE_MM:.3f} mm)")

    # ── Weld centre ───────────────────────────────────────────────────────
    weld_center = find_weld_center(top_surface, valid)
    print(f"  Weld centre    : col {weld_center}")

    # ── Interface + BW endpoints (RANSAC robust detection) ────────────────
    iface_row, iface_val, left_col, right_col, left_found, right_found = \
        find_interface_ransac(arr, top_surface, bottom_surface,
                              valid, bulk_val, weld_center)

    pore_thr = bulk_val * PORE_THRESH_FRAC
    print(f"  Dark obj thr   : iface_val < {pore_thr:.4f}  (= {PORE_THRESH_FRAC*100:.0f}% of bulk)")

    if left_found:
        lc = left_col
        print(f"  Left endpoint  : col={lc}  row={iface_row[lc]}  "
              f"iface_val={iface_val[lc]:.4f}  "
              f"({lc * SCALE_MM:.3f} mm, {iface_row[lc] * SCALE_MM:.3f} mm)")
    else:
        print(f"  Left endpoint  : NOT FOUND — flagging D1")

    if right_found:
        rc = right_col
        print(f"  Right endpoint : col={rc}  row={iface_row[rc]}  "
              f"iface_val={iface_val[rc]:.4f}  "
              f"({rc * SCALE_MM:.3f} mm, {iface_row[rc] * SCALE_MM:.3f} mm)")
    else:
        print(f"  Right endpoint : NOT FOUND — flagging D1")

    # ── Geometry: BW + D in interface coordinate system ───────────────────
    geo = compute_bw_d(left_col, right_col, iface_row, top_surface, valid, flat_top)

    bw_mm = geo['bw_mm']
    D_mm  = geo['D_mm']
    L_mm  = geo['L_mm']

    print(f"  BW             : {bw_mm:.3f} mm")
    print(f"  D              : {D_mm:.3f} mm  (perpendicular to interface x-axis)")
    print(f"  L              : {L_mm:.3f} mm")

    # ── Defect flags ──────────────────────────────────────────────────────
    flag_false_friend  = not left_found or not right_found
    flag_insuff_pen    = (bw_mm <= 1.5) and not flag_false_friend
    flag_concav_L      = L_mm >= 0.6
    flag_concav_D      = D_mm < 0.9

    # D3: burn-through restricted to the bonded zone
    iface_chk = np.where(np.isnan(iface_val), 0, iface_val)
    if left_col is not None and right_col is not None:
        burn_zone = valid.copy()
        burn_zone[:left_col]  = False
        burn_zone[right_col:] = False
        flag_burn_through = bool((burn_zone & (iface_chk < -1.0)).sum() > 50)
    else:
        flag_burn_through = bool((valid & (iface_chk < -1.0)).sum() > 50)

    print(f"  Defects        : "
          f"{'D1 ' if flag_false_friend else ''}"
          f"{'D2 ' if flag_insuff_pen else ''}"
          f"{'D3 ' if flag_burn_through else ''}"
          f"{'D4(L) ' if flag_concav_L else ''}"
          f"{'D4(D) ' if flag_concav_D else ''}"
          f"or NONE")

    # ── Annotated image ───────────────────────────────────────────────────
    out_img = os.path.join(out_dir, fname.replace('.tiff', '_annotated.png'))
    annotate_image(arr, top_surface, iface_row, iface_val, valid,
                   flat_top, geo,
                   flag_false_friend, flag_insuff_pen, flag_concav_L, flag_concav_D,
                   flag_burn_through,
                   weld_center, fname, out_img)
    print(f"  Saved          : {out_img}")

    return {
        'filename':                fname,
        'bulk_intensity':          round(bulk_val, 4),
        'flat_top_row':            flat_top,
        'weld_center_col':         weld_center,
        'bw_left_col':             left_col if left_found else -1,
        'bw_left_row':             int(iface_row[left_col]) if left_found else -1,
        'bw_right_col':            right_col if right_found else -1,
        'bw_right_row':            int(iface_row[right_col]) if right_found else -1,
        'bw_midpoint_col':         geo['mid_col'],
        'bonded_width_mm':         round(bw_mm, 3),
        'D_mm':                    round(D_mm, 3),
        'dip_col':                 int(round(geo['dip_col'])) if geo['dip_col'] >= 0 else -1,
        'dip_row':                 int(round(geo['dip_row'])) if geo['dip_row'] >= 0 else -1,
        'foot_col':                int(round(geo['foot_col'])) if geo['foot_col'] >= 0 else -1,
        'foot_row':                int(round(geo['foot_row'])) if geo['foot_row'] >= 0 else -1,
        'L_mm':                    round(L_mm, 3),
        'flag_false_friend':       flag_false_friend,
        'flag_insuff_penetration': flag_insuff_pen,
        'flag_burn_through':       flag_burn_through,
        'flag_concavity_L':        flag_concav_L,
        'flag_concavity_D':        flag_concav_D,
    }


if __name__ == '__main__':
    import glob, sys
    IMG_DIR = sys.argv[1] if len(sys.argv) > 1 else 'dataset'
    OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'results'
    os.makedirs(OUT_DIR, exist_ok=True)

    images = sorted(glob.glob(os.path.join(IMG_DIR, '*.tiff')))
    print(f"Found {len(images)} images in {IMG_DIR}")

    results = []
    for path in images:
        r = process_image(path, OUT_DIR)
        results.append(r)

    if not results:
        print("No images processed.")
        sys.exit(0)

    # Write CSV
    csv_path = os.path.join(OUT_DIR, 'measurements.csv')
    fieldnames = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nCSV saved to {csv_path}")

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  {r['filename']}: BW={r['bonded_width_mm']} mm  "
              f"D={r['D_mm']} mm  L={r['L_mm']} mm"
              f"  {'*** D1 ***' if r['flag_false_friend'] else ''}")
