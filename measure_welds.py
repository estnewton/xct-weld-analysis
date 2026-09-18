"""
Automated measurement pipeline for X-CT laser weld images.
Extracts: Bonded Width (mm) and Concavity Distance D (mm) per image.

Scale: 1.37 µm/pixel  |  Upper steel nominal: 1.5 mm  |  Lower Al nominal: 0.85 mm
"""

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d, label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, csv

# ── Constants ──────────────────────────────────────────────────────────────
SCALE_UM = 1.37          # µm per pixel
SCALE_MM = SCALE_UM / 1000
UPPER_SHEET_MM = 1.5     # nominal upper steel thickness (mm)
UPPER_SHEET_PX = int(UPPER_SHEET_MM / SCALE_MM)   # ≈ 1095 px
SMOOTH_COL   = 20        # pixel smoothing along a column
SMOOTH_ROW   = 30        # pixel smoothing along a row

# ── Helper: load and return float64 array ─────────────────────────────────
def load_image(path):
    return np.array(Image.open(path), dtype=np.float64)

# ── Step 1: find the bottom of the material ───────────────────────────────
def find_bottom_surface(arr, bulk_val):
    """
    Scan from bottom upward per column; return array of bottom-surface rows.
    Requires SUSTAIN consecutive pixels above threshold to avoid CT artifacts.
    """
    H, W = arr.shape
    threshold = bulk_val * 0.35
    SUSTAIN = 40   # consecutive pixels required
    bottom = np.full(W, H-1, dtype=int)
    for c in range(W):
        col = uniform_filter1d(arr[:, c].astype(float), size=SMOOTH_COL)
        run = 0
        for r in range(H-1, H//2, -1):
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
    """
    Scan from top downward per column; return array of top-surface rows.
    Requires SUSTAIN consecutive pixels above threshold to avoid CT artifacts.
    """
    H, W = arr.shape
    threshold = bulk_val * 0.35
    SUSTAIN = 40   # consecutive pixels required
    top = np.full(W, H, dtype=int)   # H = invalid sentinel
    for c in range(W):
        col = uniform_filter1d(arr[:, c].astype(float), size=SMOOTH_COL)
        run = 0
        for r in range(0, H//2):
            if col[r] > threshold:
                run += 1
                if run >= SUSTAIN:
                    top[c] = r - SUSTAIN
                    break
            else:
                run = 0
    return top

# ── Step 3: estimate the flat-top reference row ───────────────────────────
def flat_top_reference(top_surface, valid_mask):
    """5th percentile of top surface in central 50% of width = flat top."""
    W = len(top_surface)
    cx0, cx1 = W//4, 3*W//4
    central = valid_mask.copy()
    central[:cx0] = False
    central[cx1:] = False
    vals = top_surface[central]
    if len(vals) == 0:
        return 0
    return int(np.percentile(vals, 5))

# ── Step 4: find the interface dark line per column ───────────────────────
def find_interface(arr, top_surface, bottom_surface, valid_mask):
    """
    For each column: search for the interface dark line in the range
    [top_surface + 0.80*UPPER_SHEET_PX, top_surface + 1.20*UPPER_SHEET_PX].
    This is relative per-column, avoiding the flat-top calibration issue.
    """
    H, W = arr.shape
    FRAC_LOW, FRAC_HIGH = 0.70, 1.30   # search 70%–130% of nominal depth

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

# ── Step 5: compute bonded width ──────────────────────────────────────────
def compute_bonded_width(iface_val, valid_mask, bulk_val):
    """
    Bonded / weld zone: columns where the interface dip is moderate.
    - Threshold LOW:  iface_val < 0.80 * bulk  → some dip exists (material contact)
    - Threshold HIGH: iface_val > 0.05 * bulk  → exclude air gaps / burn-through
      (air gaps / missing material gives very negative iface_val in float X-CT)
    Returns: bonded_width_mm, left_col, right_col of main segment.
    """
    low_thr  = bulk_val * 0.80   # must be below 80% of bulk
    high_thr = bulk_val * 0.05   # must be above 5% of bulk (exclude air/void)
    iface_finite = np.where(np.isnan(iface_val), bulk_val, iface_val)
    bonded = valid_mask & (iface_finite < low_thr) & (iface_finite > high_thr)
    lbl, n = label(bonded)
    if n == 0:
        return 0.0, -1, -1

    # Pick the widest segment
    best = max(range(1, n+1), key=lambda i: len(np.where(lbl==i)[0]))
    cols = np.where(lbl == best)[0]
    if len(cols) < 10:
        return 0.0, -1, -1
    left, right = int(cols[0]), int(cols[-1])
    width_mm = (right - left) * SCALE_MM
    return width_mm, left, right

# ── Step 6: compute concavity L and D ────────────────────────────────────
def compute_concavity(top_surface, iface_row, valid_mask, flat_top):
    """
    L = deepest concavity depth below flat top (mm)
    D = remaining steel thickness at deepest concavity point (mm)
    """
    W = len(top_surface)
    cx0, cx1 = W//4, 3*W//4
    central = valid_mask.copy()
    central[:cx0] = False
    central[cx1:] = False

    if central.sum() == 0:
        return 0.0, 0.0

    tops = top_surface[central]
    deepest_row = int(np.percentile(tops, 95))   # 95th pct = deepest concavity

    # D = distance from deepest concavity to the interface AT THAT POINT
    deepest_col = np.where(central)[0][np.argmin(np.abs(tops - deepest_row))]
    iface_at_deepest = int(iface_row[deepest_col])

    L_mm = (deepest_row - flat_top) * SCALE_MM
    D_mm = (iface_at_deepest - deepest_row) * SCALE_MM
    return L_mm, D_mm

# ── Step 7: annotated visualization ──────────────────────────────────────
def annotate_image(arr, top_surface, iface_row, iface_val, valid_mask,
                   flat_top, L_mm, D_mm, bonded_width_mm, bond_left, bond_right,
                   flag_false_friend, flag_insuff_pen, flag_concav_L, flag_concav_D,
                   filename, out_path):
    H, W = arr.shape
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    disp = np.clip((arr - lo) / (hi - lo), 0, 1)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(disp, cmap='gray', aspect='auto')

    cols = np.where(valid_mask)[0]
    # Top surface profile
    ax.scatter(cols, top_surface[cols], s=0.4, c='cyan', alpha=0.6, label='Top surface')
    # Interface profile
    valid_iface = valid_mask & ~np.isnan(iface_val)
    ax.scatter(np.where(valid_iface)[0], iface_row[valid_iface],
               s=0.4, c='yellow', alpha=0.6, label='Interface (steel/Al)')

    # Flat top reference line
    ax.axhline(flat_top, color='lime', linewidth=1.5, linestyle='--',
               label=f'Flat top ref (row {flat_top})')

    # Deepest concavity
    cx0, cx1 = W//4, 3*W//4
    central = valid_mask.copy(); central[:cx0]=False; central[cx1:]=False
    if central.sum() > 0:
        deepest_row = int(np.percentile(top_surface[central], 95))
        deepest_col_idx = np.where(central)[0][np.argmin(np.abs(top_surface[central] - deepest_row))]
        iface_at_deepest = int(iface_row[deepest_col_idx])
        ax.annotate('', xy=(deepest_col_idx, iface_at_deepest),
                    xytext=(deepest_col_idx, deepest_row),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(deepest_col_idx + 30, (deepest_row + iface_at_deepest)//2,
                f'D={D_mm:.2f}mm', color='red', fontsize=10, fontweight='bold')

    # Bonded width span
    if bond_left >= 0:
        median_iface = int(np.median(iface_row[valid_iface]))
        ax.annotate('', xy=(bond_right, median_iface + 40),
                    xytext=(bond_left, median_iface + 40),
                    arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
        ax.text((bond_left+bond_right)//2, median_iface + 80,
                f'BW={bonded_width_mm:.2f}mm', color='orange', fontsize=10,
                fontweight='bold', ha='center')

    # ── Defect label box ─────────────────────────────────────────────────
    # Build list of active defects
    defect_lines = []
    if flag_false_friend:
        defect_lines.append(('D1  FALSE FRIEND', '#FF4444'))
    if flag_insuff_pen and not flag_false_friend:
        defect_lines.append(('D2  INSUFFICIENT PENETRATION', '#FF8800'))
    if flag_concav_L or flag_concav_D:
        defect_lines.append(('D4  CONCAVITY', '#FF44FF'))
    # D3 Burn-Through: detected if a large region of very negative iface_val exists
    burn_mask = valid_mask & (np.where(np.isnan(iface_val), 0, iface_val) < -1.0)
    if burn_mask.sum() > 50:
        defect_lines.append(('D3  BURN-THROUGH', '#FF0000'))

    if defect_lines:
        box_x, box_y = 30, flat_top + 50
        for i, (label_txt, color) in enumerate(defect_lines):
            ax.text(box_x, box_y + i * 70,
                    f'⚠ {label_txt}',
                    color=color, fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                              edgecolor=color, linewidth=2, alpha=0.85))
    else:
        ax.text(30, flat_top + 50, '✓  NO DEFECTS DETECTED',
                color='#44FF44', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                          edgecolor='#44FF44', linewidth=2, alpha=0.85))

    # Measurement legend box (bottom-right)
    meas_txt = (f"BW = {bonded_width_mm:.3f} mm\n"
                f"D  = {D_mm:.3f} mm\n"
                f"L  = {L_mm:.3f} mm")
    ax.text(W - 30, H - 30, meas_txt,
            color='white', fontsize=11, fontweight='bold',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#111111',
                      edgecolor='white', linewidth=1.5, alpha=0.9))

    title = f"{filename}  |  BW={bonded_width_mm:.2f}mm  |  D={D_mm:.2f}mm  |  L={L_mm:.2f}mm"
    ax.set_title(title, fontsize=11)
    ax.legend(loc='lower left', fontsize=8, markerscale=8,
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
    bulk_val = float(np.median(arr[H//3:2*H//3, W//4:3*W//4]))
    print(f"  Bulk intensity: {bulk_val:.4f}")

    # Step 1-2: surface detection
    top_surface  = find_top_surface(arr, bulk_val)
    bottom_surface = find_bottom_surface(arr, bulk_val)

    # Valid columns: top surface found (sentinel=H means not found) and in upper half
    # Material height check: actual stack is ~2100–2700 px (steel + Al + backing layers)
    raw_valid = (top_surface < H * 0.45) & (top_surface > 20)
    mat_height = bottom_surface - top_surface
    height_ok  = (mat_height > 1500) & (mat_height < 3100)
    valid = raw_valid & height_ok

    # Step 3: flat top reference (now only over geometrically sane columns)
    flat_top = flat_top_reference(top_surface, valid)
    print(f"  Flat top ref: row {flat_top} = {flat_top*SCALE_MM:.3f} mm")

    # Step 4: interface (per-column relative search; no flat_top arg needed)
    iface_row, iface_val = find_interface(arr, top_surface, bottom_surface, valid)

    # Step 5: bonded width
    bonded_width_mm, bond_left, bond_right = compute_bonded_width(iface_val, valid, bulk_val)

    # Step 6: concavity
    L_mm, D_mm = compute_concavity(top_surface, iface_row, valid, flat_top)

    # Defect flags
    flag_concav_L  = L_mm >= 0.6
    flag_concav_D  = D_mm < 0.9
    flag_insuff_pen = bonded_width_mm <= 1.5
    flag_false_friend = bonded_width_mm == 0.0

    print(f"  Bonded Width:   {bonded_width_mm:.3f} mm  (flag: {flag_insuff_pen})")
    print(f"  D (remaining):  {D_mm:.3f} mm  (flag: {flag_concav_D})")
    print(f"  L (depth):      {L_mm:.3f} mm  (flag: {flag_concav_L})")

    # Annotated image
    out_img = os.path.join(out_dir, fname.replace('.tiff', '_annotated.png'))
    annotate_image(arr, top_surface, iface_row, iface_val, valid,
                   flat_top, L_mm, D_mm, bonded_width_mm, bond_left, bond_right,
                   flag_false_friend, flag_insuff_pen, flag_concav_L, flag_concav_D,
                   fname, out_img)
    print(f"  Saved: {out_img}")

    return {
        'filename': fname,
        'bulk_intensity': round(bulk_val, 4),
        'flat_top_row': flat_top,
        'bonded_width_mm': round(bonded_width_mm, 3),
        'D_mm': round(D_mm, 3),
        'L_mm': round(L_mm, 3),
        'flag_false_friend': flag_false_friend,
        'flag_insuff_penetration': flag_insuff_pen,
        'flag_concavity_L': flag_concav_L,
        'flag_concavity_D': flag_concav_D,
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
        print(f"  {r['filename']}: BW={r['bonded_width_mm']}mm  D={r['D_mm']}mm  L={r['L_mm']}mm")
