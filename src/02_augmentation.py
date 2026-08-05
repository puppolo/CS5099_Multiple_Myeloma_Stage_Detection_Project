import sys
import time
import random
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)
from pathlib import Path
from tqdm import tqdm
from helper import _Log

# Config ###########################################################

output_directory_name = "02_augmentation_out" # Name of a directory to store the outputs.

MANIFEST_IN = Path("../data/patch_table.csv")
MANIFEST_OUT = Path("../data/patch_table_augmented.csv")
MINORITY_LABELS = [1, 2]
LABEL_NAMES = {0: "Early", 1: "On Treatment", 2: "Progression"}
RANDOM_SEED = 5099

BRIGHTNESS = 0.1
CONTRAST   = 0.1
SATURATION = 0.1

##############################################################################

OUT_DIR = Path("../" + output_directory_name)
OUT_DIR.mkdir(exist_ok=True)

_log_file   = open(OUT_DIR / "log.txt", "w")
sys.stdout  = _Log(sys.__stdout__, _log_file)
_start_time = time.time()


def print_imbalance(df, title, out_path: Path):
    train_df = df[df["split"] == "train"]

    counts = train_df.groupby("label").size()
    names = [LABEL_NAMES[i] for i in sorted(counts.index)]
    values = [counts[i] for i in sorted(counts.index)]
    majority = max(values)
    minority = min(values)
    ratio = majority / minority if minority > 0 else float("inf")

    print(f"\n{title}")
    print("-" * 50)
    for name, val in zip(names, values):
        bar = "█" * int(40 * val / majority)
        print(f"  {name:<25} {val:>7,}  {bar}")
    print()
    print(f"Imbalance ratio (majority / minority): {ratio:.2f}x")
    print(f"Most common  : {names[values.index(majority)]} ({majority:,})")
    print(f"Least common : {names[values.index(minority)]} ({minority:,})")

    # Bar chart
    colours = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=colours[:len(names)], edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Tile count")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, max(values) * 1.15)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out_path}")

def color_jitter(patch):
    patch = ImageEnhance.Brightness(patch).enhance(1 + random.uniform(-BRIGHTNESS, BRIGHTNESS))
    patch = ImageEnhance.Contrast(patch).enhance(1 + random.uniform(-CONTRAST, CONTRAST))
    patch = ImageEnhance.Color(patch).enhance(1 + random.uniform(-SATURATION, SATURATION))
    return patch

AUGMENTATIONS = [
    ("aug_hflip", lambda p: color_jitter(p.transpose(Image.FLIP_LEFT_RIGHT))),
    ("aug_vflip", lambda p: color_jitter(p.transpose(Image.FLIP_TOP_BOTTOM))),
    ("aug_rot180", lambda p: color_jitter(p.transpose(Image.ROTATE_180))),
    ("aug_color", lambda p: color_jitter(p)),
]

print()
print("SECTION 1 - Load Manifest")
print()

manifest = pd.read_csv(MANIFEST_IN)
print(f"Loaded {len(manifest):,} rows from {MANIFEST_IN}")

print("\nTile counts per split × class:")
pivot = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
pivot.columns = [LABEL_NAMES[c] for c in pivot.columns]
pivot.index.name = "Split"
print(pivot.to_string())

print_imbalance(
    manifest,
    title="Class Distribution - BEFORE Augmentation (train split)",
    out_path=OUT_DIR / "imbalance_before.png",
)

print()
print("SECTION 2 - Augmentation Preview")

random.seed(RANDOM_SEED)

sample_row = manifest[manifest["label"].isin(MINORITY_LABELS)].iloc[0]
sample_img = Image.open(sample_row["patch_path"]).convert("RGB")

n_cols = 1 + len(AUGMENTATIONS)
fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

axes[0].imshow(sample_img)
axes[0].set_title("Original", fontsize=11)
axes[0].axis("off")

for i, (suffix, fn) in enumerate(AUGMENTATIONS):
    axes[i + 1].imshow(fn(sample_img))
    axes[i + 1].set_title(suffix, fontsize=11)
    axes[i + 1].axis("off")

label_name = LABEL_NAMES[sample_row["label"]]
plt.suptitle(
    f"Augmentation preview - {sample_row['slide_id']}  ({label_name})", fontsize=12
)
plt.tight_layout()

preview_path = OUT_DIR / "augmentation_preview.png"
plt.savefig(preview_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Preview saved --> {preview_path}")

print()
print("SECTION 3 - Augmentation (train split only)")

random.seed(RANDOM_SEED)

minority_rows = manifest[
    manifest["label"].isin(MINORITY_LABELS) & (manifest["split"] == "train")
]
print(f"Minority TRAIN patches to augment: {len(minority_rows):,}")
for lbl in MINORITY_LABELS:
    n = len(minority_rows[minority_rows["label"] == lbl])
    print(f"{LABEL_NAMES[lbl]}: {n:,} patches --> {n * len(AUGMENTATIONS):,} augmented")

new_records = []

for _, row in tqdm(minority_rows.iterrows(), total=len(minority_rows), desc="Augmenting"):
    img = Image.open(row["patch_path"]).convert("RGB")

    for suffix, fn in AUGMENTATIONS:
        aug_img = fn(img)
        orig_path = Path(row["patch_path"])
        aug_path = orig_path.parent / f"{orig_path.stem}_{suffix}.png"
        aug_img.save(aug_path)

        new_records.append({
            "slide_id" : row["slide_id"],
            "split" : row["split"],
            "label" : row["label"],
            "patch_path" : str(aug_path),
            "row" : row["row"],
            "col" : row["col"],
            "x_level0" : row["x_level0"],
            "y_level0" : row["y_level0"],
            "tissue_frac" : row["tissue_frac"],
            "is_augmented" : True,
        })

print()
print(f"Augmented tiles saved: {len(new_records):,}")

print()
print("SECTION 4 — Save Manifest")

manifest["is_augmented"] = False

augmented_manifest = pd.concat(
    [manifest, pd.DataFrame(new_records)],
    ignore_index=True,
)

augmented_manifest.to_csv(MANIFEST_OUT, index=False)
augmented_manifest.to_csv(OUT_DIR / "patch_table_augmented.csv", index=False)

print("Patches counts after augmentation:")
pivot = augmented_manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
pivot.columns    = [LABEL_NAMES[c] for c in pivot.columns]
pivot.index.name = "Split"
print(pivot.to_string())

print_imbalance(
    augmented_manifest,
    title="Class Distribution - AFTER Augmentation (train split)",
    out_path=OUT_DIR / "imbalance_after.png",
)

print(f"\nOriginal tiles : {len(manifest):,}")
print(f"Augmented tiles : {len(new_records):,}")
print(f"Total : {len(augmented_manifest):,}")
print(f"\nManifest --> {MANIFEST_OUT}")
print(f"Manifest copy --> {OUT_DIR / 'patch_table_augmented.csv'}")


_elapsed = time.time() - _start_time
_mins, _secs = divmod(int(_elapsed), 60)
print()
print("Completed. All outputs written to:", OUT_DIR.resolve())
print(f"Total runtime: {_mins}m {_secs}s  ({_elapsed:.1f}s)")

sys.stdout = sys.__stdout__
_log_file.close()
