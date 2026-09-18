# X-CT Laser Weld Measurement Pipeline

**IE 496 Independent Study — Penn State | Group D**

Automated pipeline to extract two key measurements from X-ray CT cross-section images of laser-welded steel–aluminum joints.

---

## Measurements extracted

| Measurement | Symbol | Defect threshold |
|---|---|---|
| Bonded Width (weld zone span at interface) | BW | ≤ 1.5 mm → D2 Insufficient Penetration |
| Remaining steel thickness at deepest concavity | D | < 0.9 mm → D4 Concavity |
| Concavity depth from flat-top reference | L | ≥ 0.6 mm → D4 Concavity |

### Defect classification
- **D1 False Friend** — BW = 0 mm (no bonding detected)
- **D2 Insufficient Penetration** — BW > 0 but ≤ 1.5 mm
- **D3 Burn-Through** — Large void detected through aluminum (very negative intensity region)
- **D4 Concavity** — L ≥ 0.6 mm OR D < 0.9 mm

---

## Setup

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

---

## Usage

1. Place your `.tiff` X-CT images in a folder called `dataset/`
2. Run the pipeline:

```bash
python measure_welds.py
```

3. Results are saved to `results/`:
   - `measurements.csv` — one row per image with BW, D, L and defect flags
   - `*_annotated.png` — each image overlaid with detected surfaces, measurements, and defect labels

---

## Image format

- Format: float32 TIFF (single channel)
- Typical size: 3232 × 3232 pixels
- Scale: **1.37 µm/pixel**
- Upper steel sheet nominal thickness: 1.5 mm (~1095 px)
- Lower aluminum sheet nominal thickness: 0.85 mm (~620 px)

---

## How it works

```
load image (float32 TIFF)
    │
    ▼
find top steel surface per column
  (sustained threshold crossing, avoids CT ring artifacts)
    │
    ▼
find flat-top reference row
  (5th percentile of top surface in central 50% of image width)
    │
    ▼
find steel–aluminum interface per column
  (search 70%–130% of nominal steel depth below each column's top surface)
    │
    ▼
compute Bonded Width
  (columns where interface shows moderate intensity dip vs. bulk)
    │
    ▼
compute Concavity (L, D)
  (95th percentile of top surface = deepest concavity; D = interface - deepest point)
    │
    ▼
classify defects + save annotated image + write CSV
```

---

## Constants (edit in `measure_welds.py`)

```python
SCALE_UM      = 1.37      # µm per pixel
UPPER_SHEET_MM = 1.5      # nominal steel thickness (mm)
SMOOTH_COL    = 20        # smoothing kernel size along columns (px)
```

---

## Output CSV columns

| Column | Description |
|---|---|
| filename | Image filename |
| bulk_intensity | Median intensity of steel interior |
| flat_top_row | Row index of flat-top reference |
| bonded_width_mm | BW in millimeters |
| D_mm | Remaining steel thickness at deepest concavity (mm) |
| L_mm | Concavity depth from flat-top reference (mm) |
| flag_false_friend | True if BW = 0 |
| flag_insuff_penetration | True if BW ≤ 1.5 mm |
| flag_concavity_L | True if L ≥ 0.6 mm |
| flag_concavity_D | True if D < 0.9 mm |

---

## File structure

```
.
├── measure_welds.py     # Main pipeline script
├── requirements.txt     # Python dependencies
├── README.md
├── dataset/             # Put your .tiff images here
│   └── recon_00511.tiff
│   └── ...
└── results/             # Auto-created on first run
    ├── measurements.csv
    ├── recon_00511_annotated.png
    └── ...
```
