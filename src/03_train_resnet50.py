import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              classification_report, confusion_matrix)
from helper import _Log

# Config #####################################################################

output_directory_name = "03_train_resnet50_out"

MANIFEST_PATH = Path("../data/patch_table_augmented.csv")
EMBED_CACHE = Path("../embeddings_resnet50")
EPOCHS = 100
LR = 1e-4
NUM_WORKERS = 0 # Set to 12 for HPC
BATCH_SIZE = 64

LABEL_NAMES = {0: "Early", 1: "On Treatment", 2: "Progression"}
N_CLASSES = 3
BAG_WEIGHT = 0.7

##############################################################################

EMBED_CACHE.mkdir(exist_ok=True)
OUT_DIR = Path("../" + output_directory_name)
OUT_DIR.mkdir(exist_ok=True)

_log_file  = open(OUT_DIR / "log.txt", "w")
sys.stdout = _Log(sys.__stdout__, _log_file)
t_start    = time.perf_counter()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
print(f"Output dir: {OUT_DIR}")

# load data
manifest = pd.read_csv(MANIFEST_PATH)

slides = (
    manifest[manifest['is_augmented'] == False]
    .drop_duplicates('slide_id')[['slide_id', 'label', 'split']]
    .reset_index(drop=True)
)
print()
print("Slide split assignment:")
print(slides.to_string(index=False))

print()
print("Slides per split × class:")
pivot = slides.groupby(['split', 'label']).size().unstack(fill_value=0)
pivot.columns = [LABEL_NAMES[c] for c in pivot.columns]
pivot.index.name = 'Split'
print(pivot.to_string())

train_manifest = manifest[manifest['split'] == 'train'].reset_index(drop=True)
val_manifest = manifest[
    (manifest['split'] == 'val') & (manifest['is_augmented'] == False)
].reset_index(drop=True)

def tile_counts(df):
    return '  '.join(f"{LABEL_NAMES[lbl]}={len(df[df['label']==lbl]):,}" for lbl in [0,1,2])

print()
print(f"Train tiles: {len(train_manifest):,}  {tile_counts(train_manifest)}")
print(f"Val   tiles: {len(val_manifest):,}  {tile_counts(val_manifest)}")

# ResNet50 feature extractor
TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class TileDataset(Dataset):
    def __init__(self, tile_paths):
        self.tile_paths = tile_paths
    def __len__(self):
        return len(self.tile_paths)
    def __getitem__(self, idx):
        img = Image.open(self.tile_paths[idx]).convert('RGB')
        return TRANSFORM(img)

backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
backbone.fc = nn.Identity()
backbone = backbone.to(DEVICE).eval()
print()
print("ResNet50 loaded — embedding dim: 2048")

def _valid_tile_paths(tile_paths):
    valid, skipped = [], []
    for p in tile_paths:
        try:
            with Image.open(p) as img:
                img.verify()
            valid.append(p)
        except Exception:
            skipped.append(p)
    return valid, skipped

total_skipped = 0

def extract_slide_embeddings(slide_id, tile_paths):
    global total_skipped
    cache_path = EMBED_CACHE / f"{slide_id}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location='cpu')
    tile_paths, skipped = _valid_tile_paths(tile_paths)
    if skipped:
        total_skipped += len(skipped)
        print(f"  Warning: {slide_id} — skipped {len(skipped)} corrupt tile(s):")
        for p in skipped:
            print(f"    {p}")
    if not tile_paths:
        raise RuntimeError(f"{slide_id}: no valid tiles found")
    dataset = TileDataset(tile_paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                         num_workers=NUM_WORKERS, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            embeddings.append(backbone(batch.to(DEVICE)).cpu())
    embeddings = torch.cat(embeddings, dim=0)
    torch.save(embeddings, cache_path)
    return embeddings

def build_slide_embeddings(split_manifest, desc):
    result = {}
    for slide_id, group in tqdm(split_manifest.groupby('slide_id'), desc=desc):
        label = group['label'].iloc[0]
        emb = extract_slide_embeddings(slide_id, group['patch_path'].tolist())
        result[slide_id] = (emb, label)
        print(f"  {slide_id}: {emb.shape}  label={label}")
    return result

print()
print("Extracting train embeddings")
train_embeds = build_slide_embeddings(train_manifest, "Train slides")

print()
print("Extracting val embeddings")
val_embeds = build_slide_embeddings(val_manifest, "Val slides")

print()
print("Extracting test embeddings")
test_manifest = manifest[
    (manifest['split'] == 'test') & (~manifest['is_augmented'])
].reset_index(drop=True)
build_slide_embeddings(test_manifest, "Test slides")

print()
print(f"Total corrupt tiles skipped: {total_skipped}")

# ── Model definitions ─────────────────────────────────────────────────────────
class MeanPoolMIL(nn.Module):
    def __init__(self, in_dim=2048, n_classes=3, dropout=0.25):
        super().__init__()
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, n_classes))
    def forward(self, h):
        return self.classifier(h.mean(dim=0, keepdim=True)), None

class ABMIL(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=256, n_classes=3, dropout=0.25):
        super().__init__()
        self.attention  = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, n_classes))
    def forward(self, h):
        a = torch.softmax(self.attention(h), dim=0)
        return self.classifier((a * h).sum(dim=0, keepdim=True)), a

class CLAM_SB(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=256, n_classes=3, dropout=0.25, k_sample=8):
        super().__init__()
        self.k_sample  = k_sample
        self.n_classes = n_classes
        self.attention  = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, n_classes))
        self.instance_classifiers = nn.ModuleList([nn.Linear(in_dim, 2) for _ in range(n_classes)])
    def _instance_loss(self, h, a, label):
        k = min(self.k_sample, h.shape[0] // 2)
        if k == 0:
            return torch.tensor(0.0, device=h.device)
        a_flat = a.squeeze(1)
        top_idx = a_flat.topk(k).indices
        bot_idx = a_flat.topk(k, largest=False).indices
        ic = self.instance_classifiers[label.item()]
        logits  = torch.cat([ic(h[top_idx]), ic(h[bot_idx])], dim=0)
        targets = torch.cat([torch.ones(k, dtype=torch.long, device=h.device),
                              torch.zeros(k, dtype=torch.long, device=h.device)])
        return nn.CrossEntropyLoss()(logits, targets)
    def forward(self, h, label=None):
        a = torch.softmax(self.attention(h), dim=0)
        logits = self.classifier((a * h).sum(dim=0, keepdim=True))
        inst_loss = self._instance_loss(h, a, label) if label is not None else None
        return logits, a, inst_loss

mean_pool_model = MeanPoolMIL(in_dim=2048, n_classes=N_CLASSES).to(DEVICE)
abmil_model = ABMIL(in_dim=2048, hidden_dim=256, n_classes=N_CLASSES).to(DEVICE)
clam_model = CLAM_SB(in_dim=2048, hidden_dim=256, n_classes=N_CLASSES).to(DEVICE)

print()
print("model architecture and parameters")
for mname, m in [('MeanPool', mean_pool_model), ('ABMIL', abmil_model), ('CLAM-SB', clam_model)]:
    n_total     = sum(p.numel() for p in m.parameters())
    n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print()
    print(f"{mname}  -  total: {n_total:,}  trainable: {n_trainable:,}")
    for pname, p in m.named_parameters():
        print(f"{pname:<52}  {str(list(p.shape))}")

class_counts = np.bincount([train_embeds[s][1] for s in train_embeds], minlength=N_CLASSES)
class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32).to(DEVICE)
class_weights = class_weights / class_weights.sum() * N_CLASSES
criterion = nn.CrossEntropyLoss(weight=class_weights)

train_slides_data = [
    (train_embeds[s][0].float().to(DEVICE),
     torch.tensor([train_embeds[s][1]]).long().to(DEVICE))
    for s in train_embeds
]

val_slides_data = [
    (val_embeds[s][0].float().to(DEVICE),
     torch.tensor([val_embeds[s][1]]).long().to(DEVICE))
    for s in val_embeds
]

def train_model(model, model_name, is_clam=False):
    opt          = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    train_hist   = []
    val_hist     = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for h, label in train_slides_data:
            opt.zero_grad()
            if is_clam:
                logits, _, inst_loss = model(h, label=label)
                loss = BAG_WEIGHT * criterion(logits, label) + (1 - BAG_WEIGHT) * inst_loss
            else:
                logits, _ = model(h)
                loss      = criterion(logits, label)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        train_avg = epoch_loss / len(train_slides_data)
        train_hist.append(train_avg)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for h, label in val_slides_data:
                if is_clam:
                    logits, _, inst_loss = model(h, label=label)
                    loss = BAG_WEIGHT * criterion(logits, label) + (1 - BAG_WEIGHT) * inst_loss
                else:
                    logits, _ = model(h)
                    loss      = criterion(logits, label)
                val_loss += loss.item()
        val_avg = val_loss / len(val_slides_data)
        val_hist.append(val_avg)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  train_loss={train_avg:.4f}  val_loss={val_avg:.4f}")
    return train_hist, val_hist

print()
print("Training Mean Pooling...")
mp_train_hist,   mp_val_hist   = train_model(mean_pool_model, 'MeanPool')

print()
print("Training ABMIL...")
abmil_train_hist, abmil_val_hist = train_model(abmil_model, 'ABMIL')

print()
print("Training CLAM-SB...")
clam_train_hist,  clam_val_hist  = train_model(clam_model, 'CLAM-SB', is_clam=True)

# loss curves
epochs_range = range(1, EPOCHS + 1)
model_hists = [
    ('Mean Pooling', mp_train_hist,   mp_val_hist),
    ('ABMIL',        abmil_train_hist, abmil_val_hist),
    ('CLAM-SB',      clam_train_hist,  clam_val_hist),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, (name, t_hist, v_hist) in zip(axes, model_hists):
    ax.plot(epochs_range, t_hist, label='Train')
    ax.plot(epochs_range, v_hist, label='Val', linestyle='--')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(name); ax.legend()
plt.suptitle('Training vs Validation Loss - All Models', fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'training_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR / 'training_loss.png'}")

# Weight statistics
print()
print("Weight Statistics")
for mname, model in [('MeanPool', mean_pool_model), ('ABMIL', abmil_model), ('CLAM-SB', clam_model)]:
    print()
    print(f"  {mname}:")
    for pname, p in model.named_parameters():
        d = p.detach().cpu()
        print(f"    {pname:<52}  norm={d.norm():.4f}  mean={d.mean():.6f}  std={d.std():.6f}")

print()
print("Completed.")

def evaluate(model, embeds, model_name, is_clam=False):
    model.eval()
    y_pred, y_true, slide_attn = [], [], {}
    with torch.no_grad():
        for slide_id, (emb, label) in embeds.items():
            h = emb.float().to(DEVICE)
            if is_clam:
                logits, a, _ = model(h, label=None)
            else:
                logits, a    = model(h)
            pred = logits.argmax(dim=1).item()
            y_pred.append(pred)
            y_true.append(label)
            mark = "✓" if pred == label else "✗"
            if a is not None:
                a_np = a.squeeze().cpu().numpy()
                slide_attn[slide_id] = a_np
                print(f"{mark} {slide_id}  true={LABEL_NAMES[label]}  pred={LABEL_NAMES[pred]}"
                      f"attn[mean={a_np.mean():.4f}  max={a_np.max():.4f}  std={a_np.std():.4f}]")
            else:
                print(f"{mark} {slide_id}  true={LABEL_NAMES[label]}  pred={LABEL_NAMES[pred]}")
    return np.array(y_true), np.array(y_pred), slide_attn

print()
print(f"Evaluating on val set ({len(val_embeds)} slides)")
print()

all_models = [
    ('Mean Pooling', mean_pool_model, False),
    ('ABMIL', abmil_model,     False),
    ('CLAM-SB', clam_model,      True),
]

results      = {}
attn_results = {}
for name, mdl, is_clam in all_models:
    print()
    print(f"── {name} ──")
    yt, yp, attn     = evaluate(mdl, val_embeds, name, is_clam)
    results[name]      = (yt, yp)
    attn_results[name] = attn

# Summary table
print()
print("=" * 55)
print("summary - Val Set (ResNet50 features)")
print("=" * 55)
summary_rows = []
for name, (yt, yp) in results.items():
    summary_rows.append({
        'Model' : name,
        'Accuracy' : round(accuracy_score(yt, yp), 3),
        'Bal Acc' : round(balanced_accuracy_score(yt, yp), 3),
        'F1 (macro)' : round(f1_score(yt, yp, average='macro', zero_division=0), 3),
    })
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# Classification reports
for name, (yt, yp) in results.items():
    print()
    print(f"{name}:")
    print(classification_report(yt, yp, target_names=list(LABEL_NAMES.values()), zero_division=0))

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, (yt, yp)) in zip(axes, results.items()):
    present    = sorted(set(yt) | set(yp))
    tick_names = [LABEL_NAMES[l] for l in present]
    sns.heatmap(confusion_matrix(yt, yp, labels=present), annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_names, yticklabels=tick_names, ax=ax)
    ax.set_title(name); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.suptitle('Confusion Matrices - Val Set (ResNet50)', y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / 'confusion_matrices_mil.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR / 'confusion_matrices_mil.png'}")

results_csv = OUT_DIR / 'results.csv'
pd.DataFrame(summary_rows).to_csv(results_csv, index=False)
print(f"Saved: {results_csv}")

# run time
elapsed = time.perf_counter() - t_start
mins, secs = divmod(elapsed, 60)
print()
print(f"Total runtime: {int(mins)}m {secs:.1f}s")
