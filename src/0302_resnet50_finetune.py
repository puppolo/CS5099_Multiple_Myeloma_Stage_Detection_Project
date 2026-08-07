import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from helper import _Log

# Config ##############################################################

output_directory_name = "0302_resnet50_finetune_out"

MANIFEST_PATH = Path("../data/patch_table_augmented.csv")
EMBED_CACHE   = Path("../embeddings_resnet50")
EMBED_DIM     = 2048
N_CLASSES     = 3
LABEL_NAMES   = {0: "Early", 1: "On Treatment", 2: "Progression"}

CLASS_WEIGHT_POW = 1

CLAM_HIDDEN   = 64
CLAM_DROP     = 0.75
CLAM_K        = 500
CLAM_LR       = 1e-5
CLAM_WD       = 1e-4
CLAM_EPOCHS   = 2000
CLAM_BAG_W    = 0.9
CLAM_PATIENCE = 15
FOCAL_GAMMA   = 2.5

##############################################################################

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("../" + output_directory_name)
OUT_DIR.mkdir(exist_ok=True)

_fig_counter = [0]
def _save_show(*args, **kwargs):
    _fig_counter[0] += 1
    fname = OUT_DIR / f"fig_{_fig_counter[0]:02d}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"[figure saved: {fname.name}]")
    plt.close()
plt.show = _save_show

_log_file = open(OUT_DIR / "log.txt", "w")
sys.stdout = _Log(sys.__stdout__, _log_file)
_start_time = time.time()

DEVICE = torch.device(
    os.environ.get("TORCH_DEVICE") or
    ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
)
print(f"Device : {DEVICE}")
print(f"Output : {OUT_DIR}")

manifest = pd.read_csv(MANIFEST_PATH)

train_manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)
if "is_augmented" in manifest.columns:
    val_manifest = manifest[
        (manifest["split"] == "val") & (~manifest["is_augmented"])
    ].reset_index(drop=True)
else:
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)

print(f"Train: {train_manifest['slide_id'].nunique()} slides, {len(train_manifest):,} tiles")
print(f"Val: {val_manifest['slide_id'].nunique()} slides, {len(val_manifest):,} tiles")

def load_slide_embeddings(split_manifest, split_name):
    result = {}
    for slide_id, group in split_manifest.groupby("slide_id"):
        cache_path = EMBED_CACHE / f"{slide_id}.pt"
        if not cache_path.exists():
            print(f"[{split_name}] SKIPPED {slide_id} - not found in {EMBED_CACHE}")
            continue
        emb = torch.load(cache_path, map_location="cpu")
        label = group["label"].iloc[0]
        result[slide_id] = (emb, label)
        print(f"[{split_name}] {slide_id}: {emb.shape}  label={LABEL_NAMES[label]}")
    return result


train_embeds = load_slide_embeddings(train_manifest, "train")
val_embeds = load_slide_embeddings(val_manifest,"val")

class_counts = np.bincount([train_embeds[s][1] for s in train_embeds], minlength=N_CLASSES)
class_weights = torch.tensor(1.0 / (class_counts + 1e-6) ** CLASS_WEIGHT_POW, dtype=torch.float32)
class_weights = (class_weights / class_weights.sum() * N_CLASSES).to(DEVICE)
print()
print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class CLAM_SB(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_sample  = CLAM_K
        self.n_classes = N_CLASSES
        self.attention = nn.Sequential(
            nn.Linear(EMBED_DIM, CLAM_HIDDEN),
            nn.Tanh(),
            nn.Dropout(CLAM_DROP),
            nn.Linear(CLAM_HIDDEN, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(CLAM_DROP),
            nn.Linear(EMBED_DIM, N_CLASSES),
        )
        self.inst_classifiers = nn.ModuleList([
            nn.Linear(EMBED_DIM, 2) for _ in range(N_CLASSES)
        ])

    def _inst_loss(self, h, a, label):
        k = min(self.k_sample, h.shape[0] // 2)
        if k == 0:
            return torch.tensor(0.0, device=h.device)
        a_flat = a.squeeze(1)
        top_idx = a_flat.topk(k).indices
        bot_idx = a_flat.topk(k, largest=False).indices
        ic = self.inst_classifiers[label.item()]
        logits = torch.cat([ic(h[top_idx]), ic(h[bot_idx])], dim=0)
        targets = torch.cat([
            torch.ones(k, dtype=torch.long, device=h.device),
            torch.zeros(k, dtype=torch.long, device=h.device)
        ])
        return F.cross_entropy(logits, targets)

    def forward(self, h, label=None):
        a = torch.softmax(self.attention(h), dim=0)
        z = (a * h).sum(dim=0, keepdim=True)
        logits = self.classifier(z)
        i_loss = self._inst_loss(h, a, label) if label is not None and self.training else torch.tensor(0.0, device=h.device)
        return logits, a, i_loss


def train_clam(train_data, val_data):
    model = CLAM_SB().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CLAM_LR, weight_decay=CLAM_WD)
    criterion = FocalLoss(weight=class_weights, gamma=FOCAL_GAMMA)

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, CLAM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for slide_id, (emb, label) in train_data.items():
            h = emb.float().to(DEVICE)
            label_t = torch.tensor([label], dtype=torch.long, device=DEVICE)
            optimizer.zero_grad()
            logits, _, i_loss = model(h, label=label_t)
            loss = CLAM_BAG_W * criterion(logits, label_t) + (1 - CLAM_BAG_W) * i_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_preds, train_true = [], []
                for _, (emb, label) in train_data.items():
                    logits, _, _ = model(emb.float().to(DEVICE))
                    train_preds.append(logits.argmax(dim=1).item())
                    train_true.append(label)
                train_acc = sum(p == l for p, l in zip(train_preds, train_true)) / len(train_true)

                val_preds, val_true, val_loss_total = [], [], 0.0
                for _, (emb, label) in val_data.items():
                    h_v = emb.float().to(DEVICE)
                    label_t = torch.tensor([label], dtype=torch.long, device=DEVICE)
                    logits, _, _ = model(h_v)
                    val_loss_total += criterion(logits, label_t).item()
                    val_preds.append(logits.argmax(dim=1).item())
                    val_true.append(label)
                val_acc = sum(p == l for p, l in zip(val_preds, val_true)) / len(val_true)
                val_loss = val_loss_total / len(val_data)

            print(f"  Epoch {epoch:3d}/{CLAM_EPOCHS}  loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= CLAM_PATIENCE:
                    print(f"Early stopping at epoch {epoch} — no improvement for {CLAM_PATIENCE} checks")
                    break

    if best_state:
        model.load_state_dict(best_state)
    print()
    print(f"Best val_acc: {best_val_acc:.3f}  - best checkpoint restored")
    return model


print()
print("Training CLAM-SB")
clam_model = train_clam(train_embeds, val_embeds)

torch.save({
    'state_dict': clam_model.state_dict(),
    'embed_dim': EMBED_DIM,
    'n_classes': N_CLASSES,
    'hidden': CLAM_HIDDEN,
    'drop': CLAM_DROP,
    'k_sample': CLAM_K,
    'label_names': LABEL_NAMES,
}, OUT_DIR / 'clam_weights.pt')
print(f"Weights saved --> {OUT_DIR / 'clam_weights.pt'}")

clam_model.eval()
clam_preds, clam_true = [], []
with torch.no_grad():
    for slide_id, (emb, label) in val_embeds.items():
        logits, _, _ = clam_model(emb.float().to(DEVICE))
        clam_preds.append(logits.argmax(dim=1).item())
        clam_true.append(label)

print()
print("CLAM-SB - Val Set")
print(classification_report(clam_true, clam_preds,
                             labels=list(LABEL_NAMES.keys()),
                             target_names=list(LABEL_NAMES.values()),
                             zero_division=0))

present = sorted(set(clam_true) | set(clam_preds))
cm = confusion_matrix(clam_true, clam_preds, labels=present)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[LABEL_NAMES[l] for l in present],
            yticklabels=[LABEL_NAMES[l] for l in present], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("CLAM-SB - Val Set")
plt.tight_layout()
plt.show()

print()
print("=" * 40)
print("Hyperparameters used")
print("=" * 40)
print(f"CLAM_HIDDEN : {CLAM_HIDDEN}")
print(f"CLAM_DROP : {CLAM_DROP}")
print(f"CLAM_K : {CLAM_K}")
print(f"CLAM_LR : {CLAM_LR}")
print(f"CLAM_WD : {CLAM_WD}")
print(f"CLAM_EPOCHS : {CLAM_EPOCHS}")
print(f"CLAM_BAG_W : {CLAM_BAG_W}")
print(f"CLAM_PATIENCE : {CLAM_PATIENCE}")
print(f"FOCAL_GAMMA : {FOCAL_GAMMA}")

# Runtime
elapsed = time.time() - _start_time
print()
print(f"Total runtime: {int(elapsed // 3600):02d}h {int(elapsed % 3600 // 60):02d}m {elapsed % 60:.1f}s")
