"""
Automated measurement pipeline for X-CT laser weld images.
Extracts: Bonded Width (mm) and Concavity Distance D (mm) per image.

Scale: 1.37 µm/pixel  |  Upper steel nominal: 1.5 mm  |  Lower Al nominal: 0.85 mm

BW definition  : horizontal distance between the two pores flanking the weld centre
                 (pore = column where interface intensity < 5% of bulk, i.e. air-void)
D  definition  : perpendicular distance from the interface LINE (straight line between
                 the two BW pore endpoints) up to the top surface, measured along the
                 perpendicular from the midpoint origin.
L  definition  : concavity depth (top surface sag) at the BW midpoint vs. flat-top ref
"""

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d, label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, csv

# ── Constants ──────────────────────────────────────────────────────────────
SCALE_UM       = 1.37          # µm per pixel
SCALE_MM       = SCALE_UM / 1000
UPPER_SHEET_MM = 1.5           # nominal upper steel thickness (mm)
UPPER_SHEET_PX = int(UPPER_SHEET_MM / SCALE_MM)   # ≈ 1095 px
SMOOTH_COL     = 20            # pixel smoothing along a column
PORE_THRESH_FRAC = 0.05        # iface_val < bulk * this  → air void / pore

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

# ── Step 6: find pore (void) endpoints on each side of weld centre ────────
def find_pore_endpoints(iface_val, valid_mask, weld_center, bulk_val):
    """
    A pore is identified as a run of ≥ MIN_RUN consecutive columns where
    iface_val < PORE_THRESH_FRAC * bulk (i.e. near-air / void intensity).

    LEFT  endpoint = OUTERMOST (leftmost / farthest-from-centre) pore in the
                     left search zone.  Scanning left-to-right and taking the
                     first valid run ensures we pick the boundary pore at the
                     edge of the bonded region, not any small interior voids
                     that might be closer to the weld centre.

    RIGHT endpoint = INNERMOST (leftmost / closest-to-centre) pore in the
                     right search zone.  This avoids picking up large burn-
                     through voids that can lie further to the right.

    Returns
    -------
    left_col   : int or None  — pixel column of the left-side pore centre
    right_col  : int or None  — pixel column of the right-side pore centre
    left_found : bool
    right_found: bool
    """
    W           = len(iface_val)
    pore_thr    = bulk_val * PORE_THRESH_FRAC
    SEARCH_HALF = 1200   # max columns to search from centre
    MIN_RUN     = 5      # consecutive pore-level columns needed

    iface_finite = np.where(np.isnan(iface_val), bulk_val, iface_val)
    is_pore = valid_mask & (iface_finite < pore_thr)

    def collect_runs(c_start, c_end):
        """Return list of (mid_col, run_len) for all pore runs in [c_start, c_end)."""
        runs = []
        run_start = None
        for c in range(c_start, c_end):
            if is_pore[c]:
                if run_start is None:
                    run_start = c
            else:
                if run_start is not None:
                    length = c - run_start
                    if length >= MIN_RUN:
                        runs.append(((run_start + c - 1) // 2, length))
                    run_start = None
        # Handle run that extends to the boundary of the search zone
        if run_start is not None:
            length = c_end - run_start
            if length >= MIN_RUN:
                runs.append(((run_start + c_end - 1) // 2, length))
        return runs

    # ── LEFT side: scan LEFT-to-RIGHT, take the OUTERMOST (leftmost) run ──
    left_limit  = max(0, weld_center - SEARCH_HALF)
    left_runs   = collect_runs(left_limit, weld_center)
    left_col, left_found = None, False
    if left_runs:
        left_col, _ = min(left_runs, key=lambda r: r[0])   # leftmost mid_col
        left_found   = True
    print(f"  Left pore runs : {[(c, l) for c, l in left_runs]} "
          f"→ chosen col={left_col}")

    # ── RIGHT side: scan LEFT-to-RIGHT, take the INNERMOST (leftmost) run ─
    right_limit  = min(W, weld_center + SEARCH_HALF)
    right_runs   = collect_runs(weld_center + 1, right_limit)
    right_col, right_found = None, False
    if right_runs:
        right_col, _ = min(right_runs, key=lambda r: r[0])  # leftmost = closest to centre
        right_found   = True
    print(f"  Right pore runs: {[(c, l) for c, l in right_runs]} "
          f"→ chosen col={right_col}")

    return left_col, right_col, left_found, right_found

# ── Step 7a: BW and L from pore endpoints ────────────────────────────────
def compute_bw_and_L(left_col, right_col,
                     top_surface, valid_mask, flat_top):
    """
    BW = horizontal pixel distance between left_col and right_col (mm)
    L  = concavity depth at the BW midpoint column vs. flat_top reference

    Returns (bw_mm, L_mm, mid_col)
    """
    if left_col is None or right_col is None:
        return 0.0, 0.0, -1

    bw_mm   = (right_col - left_col) * SCALE_MM
    mid_col = (left_col + right_col) // 2

    if valid_mask[mid_col]:
        use_col = mid_col
    else:
        valid_cols = np.where(valid_mask)[0]
        if len(valid_cols) == 0:
            return bw_mm, 0.0, mid_col
        use_col = int(valid_cols[np.argmin(np.abs(valid_cols - mid_col))])

    L_mm = (top_surface[use_col] - flat_top) * SCALE_MM
    return bw_mm, L_mm, mid_col


# ── Step 7b: perpendicular D measurement ─────────────────────────────────
def compute_D_perpendicular(left_col, right_col, iface_row, top_surface):
    """
    Build an interface coordinate system from the two BW pore endpoints:
      X-axis : line through P1=(left_col, iface_row[left_col]) and
                             P2=(right_col, iface_row[right_col])
      Origin : midpoint O on that line
      Y-axis : perpendicular to X, pointing toward the top surface

    D = Euclidean distance from O to the point where Y-axis hits top_surface.

    Returns
    -------
    D_mm        : float  — D in millimetres
    O_col       : int    — origin column (pixel)
    O_row       : int    — origin row    (pixel)
    top_col     : int    — top-surface intersection column (pixel)
    top_row     : int    — top-surface intersection row    (pixel)
    perp_col    : float  — perpendicular unit-vector column component
    perp_row    : float  — perpendicular unit-vector row    component
    dx_norm     : float  — interface x-axis unit-vector col component
    dy_norm     : float  — interface x-axis unit-vector row component
    """
    if left_col is None or right_col is None:
        return 0.0, -1, -1, -1, -1, 0.0, -1.0, 1.0, 0.0

    W = len(top_surface)

    x1, y1 = float(left_col),  float(iface_row[left_col])
    x2, y2 = float(right_col), float(iface_row[right_col])

    dx = x2 - x1
    dy = y2 - y1
    mag = np.sqrt(dx**2 + dy**2)
    if mag < 1e-6:
        mag = 1.0

    # Unit vectors
    dx_norm = dx / mag   # x-axis direction (col component)
    dy_norm = dy / mag   # x-axis direction (row component)

    # Perpendicular pointing upward (row decreases = -dx component)
    perp_col =  dy_norm   # column component of perpendicular
    perp_row = -dx_norm   # row component (-dx_norm < 0 → upward in image)

    # Origin = midpoint on interface line
    O_col = (x1 + x2) / 2.0
    O_row = (y1 + y2) / 2.0

    # Trace ray from O along perp until it reaches the top surface
    dt = 0.5  # sub-pixel step
    top_col, top_row, t_hit = -1, -1, 0.0

    for step in range(int(mag * 4) + 8000):
        t = step * dt
        p_col = O_col + t * perp_col
        p_row = O_row + t * perp_row

        c = int(round(p_col))
        if c < 0 or c >= W:
            break
        if p_row <= top_surface[c]:
            # Refine with bisection
            t_lo, t_hi = max(0.0, t - dt), t
            for _ in range(16):
                t_mid  = (t_lo + t_hi) / 2.0
                c_mid  = int(round(O_col + t_mid * perp_col))
                c_mid  = max(0, min(W - 1, c_mid))
                p_row_m = O_row + t_mid * perp_row
                if p_row_m <= top_surface[c_mid]:
                    t_hi = t_mid
                else:
                    t_lo = t_mid
            t_hit   = (t_lo + t_hi) / 2.0
            top_col = int(round(O_col + t_hit * perp_col))
            top_row = int(round(O_row + t_hit * perp_row))
            top_col = max(0, min(W - 1, top_col))
            break

    if top_col == -1:
        # Fallback: use column nearest O
        top_col = int(round(O_col))
        top_col = max(0, min(W - 1, top_col))
        top_row = int(top_surface[top_col])
        t_hit   = np.sqrt((top_col - O_col)**2 + (top_row - O_row)**2)

    D_mm = t_hit * SCALE_MM

    print(f"  D perpendicular:")
    print(f"    Left pore   : ({int(round(x1))}, {int(round(y1))})")
    print(f"    Right pore  : ({int(round(x2))}, {int(round(y2))})")
    print(f"    Origin O    : ({int(round(O_col))}, {int(round(O_row))})")
    print(f"    Top surface : ({top_col}, {top_row})")
    print(f"    t_hit (px)  : {t_hit:.2f}")
    print(f"    D           : {D_mm:.4f} mm")

    return D_mm, int(round(O_col)), int(round(O_row)), top_col, top_row, perp_col, perp_row, dx_norm, dy_norm


# Kept for backward compat — now only used for L
def compute_measurements(left_col, right_col,
                         iface_row, top_surface, valid_mask, flat_top):
    bw_mm, L_mm, mid_col = compute_bw_and_L(
        left_col, right_col, top_surface, valid_mask, flat_top)
    D_mm, *_ = compute_D_perpendicular(
        left_col, right_col, iface_row, top_surface)
    return bw_mm, D_mm, L_mm, mid_col

# ── Step 8: annotated visualisation ──────────────────────────────────────
def annotate_image(arr, top_surface, iface_row, iface_val, valid_mask,
                   flat_top,
                   bw_mm, D_mm, L_mm,
                   left_col, right_col, mid_col,
                   weld_center,
                   flag_false_friend, flag_insuff_pen, flag_concav_L, flag_concav_D,
                   flag_burn_through,
                   pore_coords,
                   filename, out_path,
                   # perpendicular-D geometry (new)
                   D_O_col=-1, D_O_row=-1,
                   D_top_col=-1, D_top_row=-1,
                   D_perp_col=0.0, D_perp_row=-1.0,
                   D_dx_norm=1.0,  D_dy_norm=0.0):
    H, W = arr.shape
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    disp = np.clip((arr - lo) / (hi - lo), 0, 1)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(disp, cmap='gray', aspect='auto')

    # Top surface profile
    cols = np.where(valid_mask)[0]
    ax.scatter(cols, top_surface[cols], s=0.4, c='cyan', alpha=0.6, label='Top surface')

    # Interface profile
    valid_iface = valid_mask & ~np.isnan(iface_val)
    ax.scatter(np.where(valid_iface)[0], iface_row[valid_iface],
               s=0.4, c='yellow', alpha=0.6, label='Interface (steel/Al)')

    # Flat-top reference line
    ax.axhline(flat_top, color='lime', linewidth=1.5, linestyle='--',
               label=f'Flat top ref (row {flat_top})')

    # Weld centre marker (thin vertical dotted line)
    ax.axvline(weld_center, color='white', linewidth=0.8, linestyle=':',
               alpha=0.5, label=f'Weld centre (col {weld_center})')

    # ── BW and D arrows ──────────────────────────────────────────────────
    if left_col is not None and right_col is not None:
        left_iface_r  = int(iface_row[left_col])
        right_iface_r = int(iface_row[right_col])

        # Pore endpoint circles
        ax.plot(left_col,  left_iface_r,  'o', color='deepskyblue',
                markersize=14, markerfacecolor='none', markeredgewidth=2,
                label='BW endpoints (pores)')
        ax.plot(right_col, right_iface_r, 'o', color='deepskyblue',
                markersize=14, markerfacecolor='none', markeredgewidth=2)

        # BW double-arrow at interface level (slightly below)
        bw_y = max(left_iface_r, right_iface_r) + 50
        ax.annotate('', xy=(right_col, bw_y), xytext=(left_col, bw_y),
                    arrowprops=dict(arrowstyle='<->', color='orange', lw=2.5))
        ax.text((left_col + right_col) // 2, bw_y + 60,
                f'BW={bw_mm:.3f}mm', color='orange',
                fontsize=10, fontweight='bold', ha='center')

        # ── Interface coordinate system & perpendicular D ─────────────────
        if D_O_col > 0 and D_top_col > 0:
            EXT = 200   # pixels to extend the x-axis beyond each pore

            # X-axis: dashed line through both pores, extended slightly
            x_ax_x0 = left_col  - EXT * D_dx_norm
            x_ax_y0 = left_iface_r  - EXT * D_dy_norm
            x_ax_x1 = right_col + EXT * D_dx_norm
            x_ax_y1 = right_iface_r + EXT * D_dy_norm
            ax.plot([x_ax_x0, x_ax_x1], [x_ax_y0, x_ax_y1],
                    color='deepskyblue', linewidth=1.2, linestyle='--',
                    alpha=0.8, label='Interface x-axis')

            # Y-axis: dashed line from origin to top surface intersection
            ax.plot([D_O_col, D_top_col], [D_O_row, D_top_row],
                    color='lime', linewidth=1.2, linestyle='--',
                    alpha=0.8, label='D y-axis (perp.)')

            # Right-angle mark at origin O
            s = 22   # size of the right-angle square in pixels
            # Corner point along x-axis from O
            A_col = D_O_col + s * D_dx_norm
            A_row = D_O_row + s * D_dy_norm
            # Corner point along y-axis from O
            C_col = D_O_col + s * D_perp_col
            C_row = D_O_row + s * D_perp_row
            # Far corner
            B_col = A_col + s * D_perp_col
            B_row = A_row + s * D_perp_row
            sq_cols = [A_col, B_col, C_col]
            sq_rows = [A_row, B_row, C_row]
            ax.plot(sq_cols, sq_rows, color='white', linewidth=1.0, alpha=0.7)

            # Origin dot
            ax.plot(D_O_col, D_O_row, 's', color='white',
                    markersize=5, markerfacecolor='white')

            # Top-surface intersection dot
            ax.plot(D_top_col, D_top_row, 's', color='red',
                    markersize=5, markerfacecolor='red')

            # D double-arrow from origin to top-surface intersection
            ax.annotate('', xy=(D_top_col, D_top_row),
                        xytext=(D_O_col, D_O_row),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))

            # D label — offset perpendicular to the arrow for readability
            lbl_col = (D_O_col + D_top_col) / 2 + 50
            lbl_row = (D_O_row + D_top_row) / 2
            ax.text(lbl_col, lbl_row, f'D={D_mm:.3f}mm',
                    color='red', fontsize=10, fontweight='bold')

    # ── Defect label box ─────────────────────────────────────────────────
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

    # ── Pore-not-found warnings ───────────────────────────────────────────
    warn_y = flat_top + 200
    if not pore_coords['left_found']:
        ax.text(30, warn_y, '⚠ No pore found LEFT of centre',
                color='yellow', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        warn_y += 50
    if not pore_coords['right_found']:
        ax.text(30, warn_y, '⚠ No pore found RIGHT of centre',
                color='yellow', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # ── Measurement legend box (bottom-right) ────────────────────────────
    meas_txt = (f"BW = {bw_mm:.3f} mm\n"
                f"D  = {D_mm:.3f} mm\n"
                f"L  = {L_mm:.3f} mm")
    ax.text(W - 30, H - 30, meas_txt,
            color='white', fontsize=11, fontweight='bold',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#111111',
                      edgecolor='white', linewidth=1.5, alpha=0.9))

    title = (f"{filename}  |  BW={bw_mm:.3f}mm  |  D={D_mm:.3f}mm  |  L={L_mm:.3f}mm")
    ax.set_title(title, fontsize=11)
    ax.legend(loc='lower left', fontsize=8, markerscale=4,
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

    # ── Interface ─────────────────────────────────────────────────────────
    iface_row, iface_val = find_interface(arr, top_surface, bottom_surface, valid)

    # ── Weld centre ───────────────────────────────────────────────────────
    weld_center = find_weld_center(top_surface, valid)
    print(f"  Weld centre    : col {weld_center}")

    # ── Pore endpoints ────────────────────────────────────────────────────
    left_col, right_col, left_found, right_found = find_pore_endpoints(
        iface_val, valid, weld_center, bulk_val)

    pore_thr = bulk_val * PORE_THRESH_FRAC
    print(f"  Pore threshold : iface_val < {pore_thr:.4f}  (= {PORE_THRESH_FRAC*100:.0f}% of bulk)")

    if left_found:
        lc = left_col
        print(f"  Left pore      : col={lc}  row={iface_row[lc]}  "
              f"iface_val={iface_val[lc]:.4f}  "
              f"({lc * SCALE_MM:.3f} mm, {iface_row[lc] * SCALE_MM:.3f} mm)")
    else:
        print(f"  Left pore      : NOT FOUND — BW will be 0")

    if right_found:
        rc = right_col
        print(f"  Right pore     : col={rc}  row={iface_row[rc]}  "
              f"iface_val={iface_val[rc]:.4f}  "
              f"({rc * SCALE_MM:.3f} mm, {iface_row[rc] * SCALE_MM:.3f} mm)")
    else:
        print(f"  Right pore     : NOT FOUND — BW will be 0")

    # ── Measurements ──────────────────────────────────────────────────────
    bw_mm, L_mm, mid_col = compute_bw_and_L(
        left_col, right_col, top_surface, valid, flat_top)

    # Perpendicular D measurement (new)
    (D_mm, D_O_col, D_O_row,
     D_top_col, D_top_row,
     D_perp_col, D_perp_row,
     D_dx_norm, D_dy_norm) = compute_D_perpendicular(
        left_col, right_col, iface_row, top_surface)

    if mid_col >= 0:
        print(f"  BW midpoint    : col={mid_col}  row_top={top_surface[mid_col]}")

    print(f"  BW             : {bw_mm:.3f} mm")
    print(f"  D              : {D_mm:.3f} mm  (perpendicular to interface line)")
    print(f"  L              : {L_mm:.3f} mm")

    # ── Defect flags ──────────────────────────────────────────────────────
    flag_false_friend  = not left_found or not right_found   # couldn't find one pore
    flag_insuff_pen    = bw_mm <= 1.5
    flag_concav_L      = L_mm >= 0.6
    flag_concav_D      = D_mm < 0.9

    # D3: burn-through restricted to the bonded zone between the two pore endpoints
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
          f"{'D2 ' if flag_insuff_pen and not flag_false_friend else ''}"
          f"{'D3 ' if flag_burn_through else ''}"
          f"{'D4(L) ' if flag_concav_L else ''}"
          f"{'D4(D) ' if flag_concav_D else ''}"
          f"or NONE")

    pore_coords = {
        'left_found': left_found,  'left_col': left_col,
        'right_found': right_found, 'right_col': right_col,
    }

    # ── Annotated image ───────────────────────────────────────────────────
    out_img = os.path.join(out_dir, fname.replace('.tiff', '_annotated.png'))
    annotate_image(arr, top_surface, iface_row, iface_val, valid,
                   flat_top,
                   bw_mm, D_mm, L_mm,
                   left_col, right_col, mid_col,
                   weld_center,
                   flag_false_friend, flag_insuff_pen, flag_concav_L, flag_concav_D,
                   flag_burn_through,
                   pore_coords,
                   fname, out_img,
                   D_O_col=D_O_col, D_O_row=D_O_row,
                   D_top_col=D_top_col, D_top_row=D_top_row,
                   D_perp_col=D_perp_col, D_perp_row=D_perp_row,
                   D_dx_norm=D_dx_norm, D_dy_norm=D_dy_norm)
    print(f"  Saved          : {out_img}")

    return {
        'filename':              fname,
        'bulk_intensity':        round(bulk_val, 4),
        'flat_top_row':          flat_top,
        'weld_center_col':       weld_center,
        'pore_left_col':         left_col if left_found else -1,
        'pore_left_row':         int(iface_row[left_col]) if left_found else -1,
        'pore_right_col':        right_col if right_found else -1,
        'pore_right_row':        int(iface_row[right_col]) if right_found else -1,
        'bw_midpoint_col':       mid_col,
        'bonded_width_mm':       round(bw_mm, 3),
        'D_mm':                  round(D_mm, 3),
        'D_origin_col':          D_O_col,
        'D_origin_row':          D_O_row,
        'D_top_col':             D_top_col,
        'D_top_row':             D_top_row,
        'L_mm':                  round(L_mm, 3),
        'flag_false_friend':     flag_false_friend,
        'flag_insuff_penetration': flag_insuff_pen,
        'flag_burn_through':     flag_burn_through,
        'flag_concavity_L':      flag_concav_L,
        'flag_concavity_D':      flag_concav_D,
    }


if __name__ == '__main__':
    import glob
    IMG_DIR = 'dataset'
    OUT_DIR = 'results'
    os.makedirs(OUT_DIR, exist_ok=True)

    images = sorted(glob.glob(os.path.join(IMG_DIR, '*.tiff')))
    print(f"Found {len(images)} images")

    results = []
    for path in images:
        r = process_image(path, OUT_DIR)
        results.append(r)

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
        pore_info = (f"  pore_L=col{r['pore_left_col']}/row{r['pore_left_row']}"
                     f"  pore_R=col{r['pore_right_col']}/row{r['pore_right_row']}")
        print(f"  {r['filename']}: BW={r['bonded_width_mm']}mm  "
              f"D={r['D_mm']}mm  L={r['L_mm']}mm")
        print(f"  {pore_info}")
