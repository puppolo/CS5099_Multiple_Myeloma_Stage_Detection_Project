import sys
from helper import _Log
import math
import random
import numpy as np
import pandas as pd
import openslide
import matplotlib
matplotlib.use("Agg")
from PIL import Image, PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import cv2
from tqdm import tqdm

# Config #####################################################################

output_directory_name = "01_preprocssing_out" # Name of a directory to store the outputs.

READ_SIZE     = 512 # how large a region is read in a patch.
PATCH_SIZE    = 512 # pixels of saved image patch.
TISSUE_THRESH = 0.4 # minimum amount of tissue in a patch. less than TISSUE_THRESH is discarded.

TCIA_CSV = "../data/TCIA Biobank Pathology Portal.csv" # path to csv from TCIA
SVS_DIR = Path("../data/CMB-MML") # path to directory storing histopathological images in svs.
TIMEPOINT_TO_LABEL = {
    "Archival"    : 0,
    "Baseline"    : 0,
    "On Treatment": 1,
    "Progression" : 2,
}

# Only for displays
LABEL_NAMES   = {
    0: "Early",
    1: "On Treatment",
    2: "Progression"
}
MANIFEST_PATH = Path("../data/patch_table.csv") # Output path of the patch csv

VAL_SIZE    = 0.15 # Validation set size (% of (1-TEST_SIZE))
TEST_SIZE   = 0.15 # Test set size
RANDOM_SEED = 5099

DEMO_SVS = "MSB-00089-01-02.svs" # a slide for demo visualisation

##########################################################################################

OUT_DIR = Path("../"+output_directory_name)
OUT_DIR.mkdir(exist_ok=True)

_log_file   = open(OUT_DIR / "log.txt", "w")
sys.stdout  = _Log(sys.__stdout__, _log_file)

# region Function

def inspect_slide(_svs_path, save_thumbnail: Path = None):
    _id = Path(_svs_path).stem
    _slide = openslide.OpenSlide(str(_svs_path))
    width, height = _slide.dimensions
    mag = _slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, "?")
    last_lvl_w, last_lvl_h = _slide.level_dimensions[_slide.level_count - 1]

    print(f"Properties of {_id}:")
    print(f"Dimensions : {width} x {height} px")
    print(f"Magnification : {mag}x")
    print(f"Pyramid levels : {_slide.level_count}")
    for lvl in range(_slide.level_count):
        lw, lh = _slide.level_dimensions[lvl]
        ds     = _slide.level_downsamples[lvl]
        print(f"    Level {lvl}: {lw}×{lh}  (downsample {ds:.1f}x)")
    n_px = width // READ_SIZE
    n_py = height // READ_SIZE
    print(f"Expected patch grid: {n_px} × {n_py} = {n_px * n_py:,} patches")

    if save_thumbnail is not None:
        thumbnail_visualisation(_slide, _id, (last_lvl_h, last_lvl_w), save_thumbnail)

    return _slide, _id, (last_lvl_w, last_lvl_h)


def thumbnail_visualisation(_slide, _id, thumb_size, out_path: Path):
    thumbnail = _slide.get_thumbnail(thumb_size)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(thumbnail)
    ax.set_title(f"{_id} - thumbnail ({thumbnail.width}×{thumbnail.height})", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Thumbnail saved --> {out_path}")


def build_tissue_mask(_slide, thumb_size):
    thumbnail = _slide.get_thumbnail(thumb_size)
    tn_np = np.array(thumbnail.convert("RGB"))

    gray = cv2.cvtColor(tn_np, cv2.COLOR_RGB2GRAY)
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(f"Otsu threshold: {threshold:.2f}")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    _mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return _mask.astype(bool), tn_np


def mask_display(_id, _mask, tn_np, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(tn_np)
    axes[0].set_title("Thumbnail")
    axes[0].axis("off")

    axes[1].imshow(_mask, cmap="gray")
    axes[1].set_title("Tissue Mask")
    axes[1].axis("off")

    axes[2].imshow(tn_np)
    axes[2].imshow(_mask, cmap="Reds", alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle(f"{_id} — tissue detection", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    tissue_pct = _mask.mean() * 100
    print(f"Tissue coverage: {tissue_pct:.1f}% - mask saved --> {out_path}")


def patch_is_tissue(_mask, slide_w, slide_h, read_size, tx, ty, threshold):
    mh, mw = _mask.shape
    scale_x = mw / slide_w
    scale_y = mh / slide_h

    x0 = max(0, int(tx * scale_x))
    y0 = max(0, int(ty * scale_y))
    x1 = min(mw, int((tx + read_size) * scale_x))
    y1 = min(mh, int((ty + read_size) * scale_y))

    if x1 <= x0 or y1 <= y0:
        return False, 0.0

    frac = _mask[y0:y1, x0:x1].mean()
    return frac >= threshold, float(frac)


def preview_patch_grid(_mask, _slide, _id, out_path: Path):
    width, height = _slide.dimensions
    mh, mw = _mask.shape
    scale_x = mw / width
    scale_y = mh / height
    tn_np = np.array(_slide.get_thumbnail((mw, mh)).convert("RGB"))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(tn_np)

    kept, skipped = 0, 0
    for row in range(height // READ_SIZE):
        for col in range(width  // READ_SIZE):
            tx = col * READ_SIZE
            ty = row * READ_SIZE
            is_tissue, _ = patch_is_tissue(_mask, width, height, READ_SIZE, tx, ty, TISSUE_THRESH)
            rx = tx * scale_x;  ry = ty * scale_y
            rw = READ_SIZE * scale_x;  rh = READ_SIZE * scale_y
            color = "lime" if is_tissue else "red"
            rect = mpatches.Rectangle((rx, ry), rw, rh, linewidth=0.4, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            if is_tissue: kept += 1
            else:
                skipped += 1

    ax.set_title(f"{_id} - patch grid | green=kept ({kept}) red=background ({skipped})", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Patches kept={kept} skipped={skipped} - grid saved --> {out_path}")


def extract_patches(_slide, _id, _mask, _split):
    width, height = _slide.dimensions
    out_records = []
    _out_dir = Path("../data/patch") / _id
    _out_dir.mkdir(parents=True, exist_ok=True)

    for row in tqdm(range(height // READ_SIZE), desc="  Extracting"):
        for col in range(width // READ_SIZE):
            tx = col * READ_SIZE
            ty = row * READ_SIZE

            is_tissue, frac = patch_is_tissue(
                _mask, width, height, READ_SIZE, tx, ty, TISSUE_THRESH
            )
            if not is_tissue:
                continue

            patch = _slide.read_region((tx, ty), 0, (READ_SIZE, READ_SIZE)).convert("RGB")
            if READ_SIZE != PATCH_SIZE:
                patch = patch.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS)

            _patch_name = f"patch_r{row:04d}_c{col:04d}.png"
            _patch_path = _out_dir / _patch_name
            patch.save(_patch_path)

            out_records.append({
                "slide_id" : _id,
                "split" : _split,
                "patch_path" : str(_patch_path),
                "row" : row,
                "col" : col,
                "x_level0" : tx,
                "y_level0" : ty,
                "tissue_frac" : round(frac, 4),
            })

    return out_records


def split_class(_slide_list):
    n = len(_slide_list)
    s = _slide_list.copy()
    random.shuffle(s)

    if n == 1:
        return s, [], []
    elif n == 2:
        return [s[0]], [], [s[1]]
    else:
        n_test = max(1, math.ceil(n * TEST_SIZE))
        n_val = max(1, math.ceil((n - n_test) * VAL_SIZE))
        n_train = n - n_test - n_val
        if n_train < 1:
            n_val = max(0, n_val - 1)
            n_train = n - n_test - n_val
        return s[:n_train], s[n_train:n_train + n_val], s[n_train + n_val:]

# endregion

print("SECTION 1 - Demo Slide Sanity Check")

demo_slide, demo_id, demo_tn_size = inspect_slide(
    "../data/CMB-MML/" + DEMO_SVS,
    save_thumbnail=OUT_DIR / "demo_thumbnail.png",
)

demo_mask, demo_tn_np = build_tissue_mask(demo_slide, demo_tn_size)
mask_display(demo_id, demo_mask, demo_tn_np, OUT_DIR / "demo_mask.png")

preview_patch_grid(demo_mask, demo_slide, demo_id, OUT_DIR / "demo_patch_grid.png")

demo_slide.close()

print()
print("SECTION 2 - Dataset Overview")

tcia_df = pd.read_csv(TCIA_CSV)
slide_label_map = (
    tcia_df[tcia_df["Timepoint"].notna()]
    .set_index("pub_subspec_id")["Timepoint"]
    .map(TIMEPOINT_TO_LABEL)
    .to_dict()
)

all_svs      = list(SVS_DIR.glob("*.svs"))
labelled_svs = [f for f in all_svs if f.stem in slide_label_map]
dropped      = [f.stem for f in all_svs if f.stem not in slide_label_map]
print(f"SVS files found : {len(all_svs)}")
print(f"Labelled : {len(labelled_svs)}")
print(f"Dropped (no label) : {len(dropped)} → {dropped}")

labels       = [slide_label_map[f.stem] for f in labelled_svs]
label_counts = pd.Series(labels).value_counts().rename(LABEL_NAMES)
print("\nClass balance:")
print(label_counts.to_string())

median_days = tcia_df.groupby("Timepoint")["Days_From_Enrollment"].median()
print("\nMedian Days From Enrollment:")
print(median_days.to_string())

print()
print("SECTION 3 - Train / Val / Test Split")
print(f"TEST_SIZE={TEST_SIZE:.3f}  VAL_SIZE={(1-TEST_SIZE)*VAL_SIZE:.3f}  TRAIN_SIZE={(1-TEST_SIZE)*(1-VAL_SIZE):.3f} SEED={RANDOM_SEED}")

random.seed(RANDOM_SEED)

slide_entries   = [(f.stem, slide_label_map[f.stem]) for f in labelled_svs]
class_slide_map = {}
for slide_id, label in slide_entries:
    class_slide_map.setdefault(label, []).append(slide_id)

slide_split_map = {}
for label, slide_list in class_slide_map.items():
    train_s, val_s, test_s = split_class(slide_list)
    for s in train_s: slide_split_map[s] = "train"
    for s in val_s:   slide_split_map[s] = "val"
    for s in test_s:  slide_split_map[s] = "test"

split_df = pd.DataFrame([
    {"slide_id": sid, "label": slide_label_map[sid], "split": split}
    for sid, split in slide_split_map.items()
]).sort_values(["split", "label"]).reset_index(drop=True)

print()
print("Slide split assignment:")
print(split_df.to_string(index=False))

print()
print("Distribution per split:")
dist = split_df.groupby(["split", "label"]).size().unstack(fill_value=0)
dist.columns    = [LABEL_NAMES[c] for c in dist.columns]
dist.index.name = "Split"
print(dist.to_string())

n_train = (split_df["split"] == "train").sum()
n_val   = (split_df["split"] == "val").sum()
n_test  = (split_df["split"] == "test").sum()
print()
print(f"Totals — Train: {n_train}  Val: {n_val}  Test: {n_test}")

split_csv = OUT_DIR / "slide_splits.csv"
split_df.to_csv(split_csv, index=False)
print()
print(f"Split table saved --> {split_csv}")

print()
print("SECTION 4 - Patch Extraction")

all_records = []

for svs_path in labelled_svs:
    slide_id = svs_path.stem
    label = slide_label_map[slide_id]
    split = slide_split_map[slide_id]
    print()
    print(f"[{split.upper()}] {slide_id} (label={LABEL_NAMES[label]})")

    slide = openslide.OpenSlide(str(svs_path))
    last_lw, last_lh = slide.level_dimensions[slide.level_count - 1]
    mask, _ = build_tissue_mask(slide, (last_lw, last_lh))
    records = extract_patches(slide, slide_id, mask, split)

    for r in records:
        r["label"] = label

    all_records.extend(records)
    slide.close()
    print(f"--> {len(records):,} patches saved to ../data/patch/{slide_id}/")

# Save manifest
manifest = pd.DataFrame(all_records)
manifest.to_csv(MANIFEST_PATH, index=False)
manifest.to_csv(OUT_DIR / "patch_table.csv", index=False) # Save a copy to output directory

print()
print(f"Total patches : {len(manifest):,}")
print(f"Manifest --> {MANIFEST_PATH}")
print(f"Manifest copy --> {OUT_DIR / 'patch_table.csv'}")

print()
print("Patch counts per split x class:")
pivot = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
pivot.columns = [LABEL_NAMES[c] for c in pivot.columns]
pivot.index.name = "Split"
print(pivot.to_string())

print()
print("Completed. All outputs written to:", OUT_DIR.resolve())

sys.stdout = sys.__stdout__
_log_file.close()
