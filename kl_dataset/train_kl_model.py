#!/usr/bin/env python3
"""
train_kl_model.py
=================
Two-stream EfficientNet baseline for kinematic lensing shear regression.

Architecture
------------
Physical motivation:
  - Photometric image: strongly distorted by shear (shape change, ellipticity).
    EfficientNet stem is useful here because pretrained ImageNet features encode
    shape and texture, which map onto the kind of structural distortions shear
    induces in galaxy morphology.
  - Velocity map: weakly distorted by shear (only the minor-axis velocity
    v'_minor is sensitive to g×; see Xu+2022 Eq. 7).  A lightweight CNN or
    second EfficientNet stem extracts these asymmetric kinematic signatures.

The two streams are merged after their respective stems so that the shared
EfficientNet body can learn the *coupling* between photometric and kinematic
distortions — which is exactly the KL signal.  Two separate regression heads
then predict g+ and g× independently.

Data layout expected on disk (produced by generate_kl_tng50.py):
  <images_root>/
    snap<N>/
      galaxy_<ID>/
        galaxy_<ID>_image_sheared.fits    ← photometric input  (npix × npix)
        galaxy_<ID>_velmap_original.fits  ← used to compute observed velmap
                                             via shear transform; not passed
                                             directly as a model input
        (other FITS files ignored by this script)

Labels come from the plan CSV columns: subhalo_id, snap, g1, g2.
  g1 → g+  (tangential shear component)
  g2 → g×  (cross shear component)

Usage
-----
  python train_kl_model.py \
      --images_root  /path/to/kl_dataset/images \
      --csv          /path/to/dataset_plan_with_ids.csv \
      --outdir       ./kl_model_output \
      --epochs       50 \
      --batch_size   32

Dependencies
------------
  pip install torch torchvision timm astropy numpy pandas \
              scikit-learn matplotlib tqdm
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from astropy.io import fits
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def load_fits_image(path: Path, npix: int) -> np.ndarray:
    """
    Load a single-extension FITS file → (npix, npix) float32, normalised [0,1].
    NaNs (empty velmap pixels) are filled with 0.
    """
    with fits.open(str(path)) as hdul:
        data = hdul[0].data.astype(np.float32)

    data = np.nan_to_num(data, nan=0.0)

    if data.shape != (npix, npix):
        from PIL import Image
        img  = Image.fromarray(data).resize((npix, npix), Image.BILINEAR)
        data = np.array(img, dtype=np.float32)

    lo, hi = np.percentile(data, 1), np.percentile(data, 99)
    if hi > lo:
        data = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    else:
        data = np.zeros_like(data)

    return data


def apply_shear_numpy(img: np.ndarray, g1: float, g2: float) -> np.ndarray:
    """
    Apply cosmic shear (g1, g2) to a source-plane image at the pixel level.
    Uses the lensing matrix A (Xu+2022 Eq. 1):
        theta_S = A · theta_O,   A = [[1-g1, -g2], [-g2, 1+g1]]
    i.e. we pull each output pixel from the source-plane coords.

    This is identical to shear_image_remap() in generate_kl_tng50.py and
    lets us apply arbitrary shear values at training time without re-generating
    FITS files, enabling the multi-draw expansion strategy.
    """
    from scipy.ndimage import map_coordinates

    npix   = img.shape[0]
    centre = (npix - 1) / 2.0
    col_o, row_o = np.meshgrid(np.arange(npix), np.arange(npix))
    xo = (col_o - centre) / centre
    yo = (row_o - centre) / centre

    xs = (1 - g1) * xo - g2       * yo
    ys = -g2      * xo + (1 + g1) * yo

    col_s = xs * centre + centre
    row_s = ys * centre + centre

    return map_coordinates(img, [row_s.ravel(), col_s.ravel()],
                           order=3, mode="constant", cval=0.0
                           ).reshape(npix, npix).astype(np.float32)


class KLShearDataset(Dataset):
    """
    Loads inputs grounded in Xu+2022 kinematic lensing theory.
    ALL inputs are strictly what an observer would measure — no leakage
    from the unsheared/source-plane data.

    Photometric input (3-channel):
        ch0-2: sheared stellar image (replicated to 3 channels for EfficientNet)

    Kinematic input (3-channel, all derived from the observed velocity map):
        ch0: v_obs     — v'_LoS(θ_O) = v_LoS(A·θ_O): the observed LoS
                         velocity field after lensing (Xu+2022 Eq. 6).
                         This is what Roman's grism actually measures.

        ch1: v_asym    — v_obs(x,y) + v_obs(-x,-y): residual from 180°
                         rotation antisymmetry.  An unsheared rotating disc
                         is perfectly antisymmetric (v → -v under π rotation),
                         so this channel is zero for g=0 and its amplitude is
                         directly proportional to g× via Eq. 7:
                             v'_minor ∝ g×
                         This is the primary g× carrier.  Derived from v_obs.

        ch2: v_sym     — v_obs(x,y) - v_obs(-x,-y): the antisymmetric
                         (rotation-preserving) component.  For a disc this
                         is the dominant term encoding sin(i) and the disc
                         rotation amplitude, which via the TFR connects to
                         the intrinsic ellipticity e_int needed to infer g+
                         (Xu+2022 Eq. 5, 8, 9).  Also derived from v_obs.

    ch1 and ch2 are the even/odd decomposition of v_obs under 180° rotation —
    a natural basis given the disc symmetry argument of §2.2.  Both are
    derived purely from the observed field; no source-plane information is used.

    Modes (use_original_image):
        False: load pre-sheared stellar image FITS directly
        True : load original stellar image FITS, apply shear on-the-fly
               (required for multi-draw CSVs from expand_shear_draws.py)
    """

    def __init__(self, df: pd.DataFrame, images_root: Path,
                 npix: int = 256, use_original_image: bool = False):
        self.df                 = df.reset_index(drop=True)
        self.images_root        = images_root
        self.npix               = npix
        self.use_original_image = use_original_image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        sid  = int(row["subhalo_id"])
        snap = int(row["snap"])
        g1   = float(row["g1"])
        g2   = float(row["g2"])

        gal_dir = self.images_root / f"snap{snap}" / f"galaxy_{sid}"

        # ── Photometric input: sheared stellar image ──────────────────────
        if self.use_original_image:
            photo_raw = load_fits_image(
                gal_dir / f"galaxy_{sid}_image_original.fits", self.npix)
            photo = apply_shear_numpy(photo_raw, g1, g2)
        else:
            photo = load_fits_image(
                gal_dir / f"galaxy_{sid}_image_sheared.fits", self.npix)

        # ── Kinematic input: all derived from observed (sheared) velmap ───
        # Apply shear to the original velocity map to get what Roman observes
        # (Eq. 6: v'_LoS(θ_O) = v_LoS(A·θ_O))
        vel_orig_raw = load_fits_image(
            gal_dir / f"galaxy_{sid}_velmap_original.fits", self.npix)
        v_obs = apply_shear_numpy(vel_orig_raw, g1, g2)

        # Decompose v_obs into even and odd parts under 180° rotation
        v_rot180 = np.rot90(v_obs, 2)          # 180° rotation
        v_asym   = v_obs + v_rot180            # even part: zero for unsheared disc → encodes g×
        v_sym    = v_obs - v_rot180            # odd part:  disc rotation signal → encodes sin(i), g+

        # Normalise v_asym and v_sym independently to [0,1]
        def norm(arr):
            lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
            if hi > lo:
                return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            return np.zeros_like(arr)

        # v_obs is already normalised by load_fits_image
        v_asym = norm(v_asym)
        v_sym  = norm(v_sym)

        # ── Tensors ───────────────────────────────────────────────────────
        photo_t = torch.from_numpy(photo).unsqueeze(0).expand(3, -1, -1)
        vel_t   = torch.stack([
            torch.from_numpy(v_obs),    # ch0: observed velocity (Eq. 6)
            torch.from_numpy(v_asym),   # ch1: g× carrier (Eq. 7)
            torch.from_numpy(v_sym),    # ch2: disc rotation / g+ carrier
        ], dim=0)                        # (3, H, W)

        labels = torch.tensor([g1, g2], dtype=torch.float32)
        return photo_t, vel_t, labels


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class TwoStreamShearNet(nn.Module):
    """
    Two-stream architecture (compatible with timm >= 0.9):

      Photo stream  : EfficientNet-B0 stem (conv_stem + bn1 + blocks[0:2])
                      → 24-channel feature map at H/4, W/4
      Velocity stream: lightweight CNN → 24-channel feature map at H/4, W/4
      Fusion        : channel concat (48 ch) → 1×1 proj back to 24 ch
      Shared body   : EfficientNet-B0 blocks[2:] + conv_head + bn2
      Regression    : two separate linear heads for g+ and g×

    timm >= 0.9 fuses the activation into bn1/bn2 as BatchNormAct2d,
    so act1/act2 no longer exist as standalone named children.
    Named children are: conv_stem, bn1, blocks, conv_head, bn2, global_pool, classifier.

    EfficientNet-B0 channel map (timm 1.x):
      conv_stem + bn1       → (B,  32, H/2,  W/2)
      blocks[0] MBConv1 ×1 → (B,  16, H/2,  W/2)
      blocks[1] MBConv6 ×2 → (B,  24, H/4,  W/4)  ← stem cutpoint
      blocks[2] MBConv6 ×2 → (B,  40, H/8,  W/8)
      blocks[3] MBConv6 ×3 → (B,  80, H/16, W/16)
      blocks[4] MBConv6 ×3 → (B, 112, H/16, W/16)
      blocks[5] MBConv6 ×4 → (B, 192, H/32, W/32)
      blocks[6] MBConv6 ×1 → (B, 320, H/32, W/32)
      conv_head + bn2       → (B,1280, H/32, W/32)
    """

    def __init__(self, pretrained: bool = True, freeze_stem: bool = False):
        super().__init__()

        backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, features_only=False
        )

        # ── Photo stem ───────────────────────────────────────────────────────
        # bn1 is BatchNormAct2d in timm >= 0.9 (activation fused in)
        self.photo_stem = nn.Sequential(
            backbone.conv_stem,
            backbone.bn1,
            backbone.blocks[0],
            backbone.blocks[1],
        )  # output: (B, 24, H/4, W/4)

        if freeze_stem:
            for p in self.photo_stem.parameters():
                p.requires_grad = False

        # ── Fusion projection ────────────────────────────────────────────────
        # blocks[2] expects 24-ch input; after concat we have 48, so project back
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
        )

        # ── Shared body ──────────────────────────────────────────────────────
        # bn2 is also BatchNormAct2d in timm >= 0.9
        self.shared_body = nn.Sequential(
            backbone.blocks[2],
            backbone.blocks[3],
            backbone.blocks[4],
            backbone.blocks[5],
            backbone.blocks[6],
            backbone.conv_head,
            backbone.bn2,
        )  # output: (B, 1280, H/32, W/32)

        self.pool = nn.AdaptiveAvgPool2d(1)

        # ── Velocity stream ──────────────────────────────────────────────────
        # 3-channel input — all derived from the observed velocity map only
        # (no source-plane / unsheared data; observer-only inputs):
        #   ch0: v_obs   — observed LoS velocity after lensing (Eq. 6)
        #   ch1: v_asym  — even part under 180° rotation ∝ g× (Eq. 7)
        #   ch2: v_sym   — odd part under 180° rotation ∝ sin(i), disc rotation
        self.vel_stream = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.SiLU(),
            nn.Conv2d(48, 24, 3, padding=1), nn.BatchNorm2d(24), nn.SiLU(),
            nn.MaxPool2d(2),
        )  # output: (B, 24, H/4, W/4)

        # ── Regression heads ─────────────────────────────────────────────────
        feat_dim = 1280
        self.head_g1 = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),       nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.head_g2 = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),       nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, photo, vel):
        """
        photo : (B, 3, H, W)   sheared stellar image
        vel   : (B, 1, H, W)   original LoS velocity map
        """
        f_photo = self.photo_stem(photo)          # (B, 24, H/4, W/4)
        f_vel   = self.vel_stream(vel)            # (B, 24, H/4, W/4)

        fused   = torch.cat([f_photo, f_vel], dim=1)   # (B, 48, H/4, W/4)
        fused   = self.fusion_proj(fused)               # (B, 24, H/4, W/4)

        features = self.shared_body(fused)        # (B, 1280, H/32, W/32)
        pooled   = self.pool(features).flatten(1) # (B, 1280)

        g1_pred  = self.head_g1(pooled).squeeze(1)  # (B,)
        g2_pred  = self.head_g2(pooled).squeeze(1)  # (B,)

        return g1_pred, g2_pred


# ═══════════════════════════════════════════════════════════════════════════════
# Train / eval helpers
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for photo, vel, labels in loader:
            photo  = photo.to(device)
            vel    = vel.to(device)
            labels = labels.to(device)          # (B, 2)

            g1_pred, g2_pred = model(photo, vel)
            preds = torch.stack([g1_pred, g2_pred], dim=1)  # (B, 2)

            loss = criterion(preds, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    n          = len(all_labels)

    mse_g1 = float(np.mean((all_preds[:, 0] - all_labels[:, 0]) ** 2))
    mse_g2 = float(np.mean((all_preds[:, 1] - all_labels[:, 1]) ** 2))

    return total_loss / n, mse_g1, mse_g2, all_preds, all_labels


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(history: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train", color="#2E86AB")
    axes[0].plot(history["val_loss"],   label="Val",   color="#E84855")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Smooth L1 Loss")
    axes[0].set_title("Total Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_mse_g1"], label="Train g+", color="#2E86AB")
    axes[1].plot(history["val_mse_g1"],   label="Val g+",   color="#E84855")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE")
    axes[1].set_title("g+ (g1) MSE"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(history["train_mse_g2"], label="Train g×", color="#2E86AB")
    axes[2].plot(history["val_mse_g2"],   label="Val g×",   color="#E84855")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("MSE")
    axes[2].set_title("g× (g2) MSE"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out_dir / "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: loss_curves.png")


def plot_shear_scatter(preds: np.ndarray, labels: np.ndarray,
                       split_name: str, out_dir: Path):
    """
    Two scatter plots side by side:
      Left:  g+ axis vs g× axis, coloured by fractional error in g+
      Right: g+ axis vs g× axis, coloured by fractional error in g×

    Fractional error: (pred - true) / (|true| + eps) to avoid div-by-zero.
    """
    eps   = 1e-4
    err1  = (preds[:, 0] - labels[:, 0]) / (np.abs(labels[:, 0]) + eps)
    err2  = (preds[:, 1] - labels[:, 1]) / (np.abs(labels[:, 1]) + eps)

    # Clip colour range at ±2 for readability
    vmax  = 2.0
    cmap  = "RdBu_r"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, err, comp in zip(axes, [err1, err2], ["g+", "g×"]):
        sc = ax.scatter(labels[:, 0], labels[:, 1],
                        c=np.clip(err, -vmax, vmax),
                        cmap=cmap, vmin=-vmax, vmax=vmax,
                        s=8, alpha=0.6, rasterized=True)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(f"({comp} pred − true) / |true|", fontsize=9)
        ax.set_xlabel("g+ true"); ax.set_ylabel("g× true")
        ax.set_title(f"Fractional error in {comp}  [{split_name}]")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    fname = out_dir / f"shear_scatter_{split_name}.png"
    fig.savefig(str(fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_pred_vs_true(preds: np.ndarray, labels: np.ndarray,
                      split_name: str, out_dir: Path):
    """Predicted vs. true scatter for g+ and g×."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, name, color in zip(
        axes, [0, 1], ["g+", "g×"], ["#2E86AB", "#E84855"]
    ):
        lim = max(np.abs(labels[:, col]).max(), np.abs(preds[:, col]).max()) * 1.1
        ax.scatter(labels[:, col], preds[:, col],
                   s=6, alpha=0.5, color=color, rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
        mse = float(np.mean((preds[:, col] - labels[:, col]) ** 2))
        ax.set_xlabel(f"{name} true"); ax.set_ylabel(f"{name} predicted")
        ax.set_title(f"{name} pred vs true  [{split_name}]  MSE={mse:.2e}")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

    plt.tight_layout()
    fname = out_dir / f"pred_vs_true_{split_name}.png"
    fig.savefig(str(fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_residual_hist(preds: np.ndarray, labels: np.ndarray,
                       split_name: str, out_dir: Path,
                       eps: float = 0.035):
    """
    Histogram of fractional errors (pred - true) / |true| for g+ and g×.

    eps=0.035 is the KL shape noise floor from Xu+2022 §3.3.  Galaxies with
    |g_true| below this are excluded since fractional error is not meaningful
    when the signal is smaller than the measurement noise floor.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, name, color in zip(
        axes, [0, 1], ["g+", "g×"], ["#2E86AB", "#E84855"]
    ):
        true = labels[:, col]
        pred = preds[:, col]

        # Exclude near-zero true values where fractional error is degenerate
        valid     = np.abs(true) >= eps
        n_excl    = (~valid).sum()
        frac_err  = (pred[valid] - true[valid]) / np.abs(true[valid])

        # Clip display range to ±3 so a few wild outliers don't compress the histogram
        clip      = 3.0
        frac_plot = np.clip(frac_err, -clip, clip)
        n_clipped = int(np.sum(np.abs(frac_err) > clip))

        median = float(np.median(frac_err))
        mean   = float(np.mean(frac_err))
        std    = float(np.std(frac_err))

        ax.hist(frac_plot, bins=50, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(0,      color="k",      lw=1.0)
        ax.axvline(median, color="orange", lw=1.5, ls="--",
                   label=f"median={median:+.3f}")
        ax.axvline(mean,   color="red",    lw=1.0, ls=":",
                   label=f"mean={mean:+.3f}")

        title = (f"{name} fractional error  [{split_name}]\n"
                 f"σ={std:.3f}   "
                 f"n={valid.sum()}"
                 + (f"  excl={n_excl}" if n_excl else "")
                 + (f"  clipped={n_clipped}" if n_clipped else ""))
        ax.set_xlabel(f"(pred − true) / |true|  [{name}]")
        ax.set_ylabel("Count")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    fname = out_dir / f"residuals_{split_name}.png"
    fig.savefig(str(fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Data splitting  (stratified by galaxy ID to prevent leakage)
# ═══════════════════════════════════════════════════════════════════════════════

def build_splits(df: pd.DataFrame, images_root: Path, npix: int,
                 use_original_image: bool = False,
                 train_frac=0.70, val_frac=0.15, seed=42):
    """
    Split df into train / val / test with no subhalo_id overlap between splits.
    GroupShuffleSplit groups by subhalo_id so each physical galaxy lands in
    exactly one split, preventing the model from memorising galaxy morphology
    independent of the shear signal.
    """
    # First pass: split off test set
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=1.0 - train_frac - val_frac, random_state=seed
    )
    trainval_idx, test_idx = next(
        gss_test.split(df, groups=df["subhalo_id"])
    )

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test     = df.iloc[test_idx].reset_index(drop=True)

    # Second pass: split trainval into train / val
    val_relative = val_frac / (train_frac + val_frac)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_relative, random_state=seed
    )
    train_idx, val_idx = next(
        gss_val.split(df_trainval, groups=df_trainval["subhalo_id"])
    )

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val   = df_trainval.iloc[val_idx].reset_index(drop=True)

    print(f"Split sizes — train: {len(df_train)}  val: {len(df_val)}  "
          f"test: {len(df_test)}")
    print(f"Unique galaxies — train: {df_train['subhalo_id'].nunique()}  "
          f"val: {df_val['subhalo_id'].nunique()}  "
          f"test: {df_test['subhalo_id'].nunique()}")

    train_ds = KLShearDataset(df_train, images_root, npix, use_original_image)
    val_ds   = KLShearDataset(df_val,   images_root, npix, use_original_image)
    test_ds  = KLShearDataset(df_test,  images_root, npix, use_original_image)

    return train_ds, val_ds, test_ds, df_train, df_val, df_test


# ═══════════════════════════════════════════════════════════════════════════════
# File existence filter
# ═══════════════════════════════════════════════════════════════════════════════

def filter_to_existing(df: pd.DataFrame, images_root: Path,
                       use_original_image: bool = False) -> pd.DataFrame:
    """Drop rows where the required FITS files do not exist on disk."""
    keep = []
    for _, row in df.iterrows():
        sid     = int(row["subhalo_id"])
        snap    = int(row["snap"])
        gal_dir = images_root / f"snap{snap}" / f"galaxy_{sid}"
        photo_name = (f"galaxy_{sid}_image_original.fits" if use_original_image
                      else f"galaxy_{sid}_image_sheared.fits")
        if (
            (gal_dir / photo_name).exists() and
            (gal_dir / f"galaxy_{sid}_velmap_original.fits").exists()
        ):
            keep.append(True)
        else:
            keep.append(False)

    n_dropped = len(df) - sum(keep)
    if n_dropped:
        print(f"  [WARN] Dropping {n_dropped} rows with missing FITS files.")
    return df[keep].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-stream KL shear regressor training pipeline."
    )
    p.add_argument("--images_root", required=True,
                   help="Root directory of generated FITS images (contains snap*/)")
    p.add_argument("--csv",         required=True,
                   help="dataset_plan_with_ids.csv (must have subhalo_id, snap, g1, g2)")
    p.add_argument("--outdir",      default="./kl_model_output")
    p.add_argument("--npix",        type=int,   default=128,
                   help="Resize images to this square size (default 128; "
                        "use 224 for full EfficientNet resolution)")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--no_pretrain", action="store_true",
                   help="Disable ImageNet pretraining for the EfficientNet stem")
    p.add_argument("--freeze_stem", action="store_true",
                   help="Freeze the EfficientNet photo stem during training")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--patience",    type=int,   default=10,
                   help="Early-stopping patience on val loss")
    p.add_argument("--use_original_image", action="store_true",
                   help="Load image_original.fits and apply shear on-the-fly. "
                        "Required when using an expanded multi-draw CSV from "
                        "expand_shear_draws.py. Slower per epoch but enables "
                        "unlimited shear augmentation without new API calls.")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load and validate CSV ────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    required = {"subhalo_id", "snap", "g1", "g2"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df[df["subhalo_id"] != -1].reset_index(drop=True)

    images_root = Path(args.images_root)
    print(f"\nFiltering to rows with existing FITS files …")
    df = filter_to_existing(df, images_root, args.use_original_image)
    print(f"Usable rows: {len(df)}")

    if len(df) < 10:
        raise RuntimeError("Too few valid rows after filtering. "
                           "Check --images_root and --csv paths.")

    # ── Splits ───────────────────────────────────────────────────────────────
    print("\nBuilding train / val / test splits (stratified by subhalo_id) …")
    train_ds, val_ds, test_ds, df_train, df_val, df_test = build_splits(
        df, images_root, args.npix, args.use_original_image
    )
    df_train.to_csv(out_dir / "split_train.csv", index=False)
    df_val.to_csv(  out_dir / "split_val.csv",   index=False)
    df_test.to_csv( out_dir / "split_test.csv",  index=False)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    print("\nBuilding model …")
    model = TwoStreamShearNet(
        pretrained=not args.no_pretrain,
        freeze_stem=args.freeze_stem,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Loss and optimiser ───────────────────────────────────────────────────
    # Smooth L1 (Huber) is more robust than MSE to occasional outliers that
    # arise from badly-resolved galaxies (very few star particles, edge cases).
    criterion = nn.SmoothL1Loss()

    # Separate LR groups: lower LR for pretrained stem, higher for new weights.
    stem_params  = list(model.photo_stem.parameters())
    other_params = [p for p in model.parameters()
                    if not any(p is s for s in stem_params)]
    optimizer = optim.AdamW([
        {"params": stem_params,  "lr": args.lr * 0.1},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training loop ────────────────────────────────────────────────────────
    history = {k: [] for k in
               ["train_loss", "val_loss",
                "train_mse_g1", "val_mse_g1",
                "train_mse_g2", "val_mse_g2"]}

    best_val_loss  = float("inf")
    patience_count = 0
    best_ckpt_path = out_dir / "best_model.pt"

    print(f"\nTraining for up to {args.epochs} epochs "
          f"(patience={args.patience}) …\n")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mse1, tr_mse2, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        vl_loss, vl_mse1, vl_mse2, _, _ = run_epoch(
            model, val_loader,   criterion, optimizer, device, training=False
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_mse_g1"].append(tr_mse1)
        history["val_mse_g1"].append(vl_mse1)
        history["train_mse_g2"].append(tr_mse2)
        history["val_mse_g2"].append(vl_mse2)

        lr_now = scheduler.get_last_lr()[-1]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss  train={tr_loss:.4f}  val={vl_loss:.4f} | "
              f"MSE g+ train={tr_mse1:.4e}  val={vl_mse1:.4e} | "
              f"MSE g× train={tr_mse2:.4e}  val={vl_mse2:.4e} | "
              f"lr={lr_now:.2e}")

        # Early stopping & checkpointing
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_count = 0
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":   vl_loss,
                "args":       vars(args),
            }, best_ckpt_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs).")
                break

    # ── Load best checkpoint for evaluation ─────────────────────────────────
    print(f"\nLoading best checkpoint (val loss = {best_val_loss:.4f}) …")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Test evaluation ──────────────────────────────────────────────────────
    print("\n── Test set evaluation ──────────────────────────────────────")
    te_loss, te_mse1, te_mse2, te_preds, te_labels = run_epoch(
        model, test_loader, criterion, optimizer, device, training=False
    )
    print(f"  Test Smooth-L1 loss : {te_loss:.4f}")
    print(f"  Test MSE  g+        : {te_mse1:.4e}")
    print(f"  Test MSE  g×        : {te_mse2:.4e}")
    print(f"  Test RMSE g+        : {np.sqrt(te_mse1):.4e}")
    print(f"  Test RMSE g×        : {np.sqrt(te_mse2):.4e}")

    # ── Fractional error percentiles ─────────────────────────────────────────
    # We use two complementary error metrics:
    #
    # 1. Fractional: |pred - true| / |true|  — meaningful only when |true|
    #    is large enough to be physically significant.  eps is set to the
    #    KL shape noise floor from Xu+2022 (σ_ε^KL = 0.035), below which
    #    fractional error is dominated by noise rather than model quality.
    #
    # 2. Absolute: |pred - true|  — no exclusion needed; directly comparable
    #    to the shape noise floor and to the σ values in Xu+2022 Fig. 4.
    eps  = 0.035   # KL shape noise floor (Xu+2022 §3.3)
    pcts = [50, 75, 90, 95, 99]

    print("\n  Absolute error percentiles  |pred − true|")
    print(f"  {'Percentile':>12}  {'g+':>10}  {'g×':>10}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}")
    abs_errs = {}
    for col, tag in [(0, "g+"), (1, "g×")]:
        abs_errs[tag] = np.abs(te_preds[:, col] - te_labels[:, col])
    for p in pcts:
        gp = np.percentile(abs_errs["g+"], p)
        gx = np.percentile(abs_errs["g×"], p)
        print(f"  {p:>11}th  {gp:>10.4f}  {gx:>10.4f}")

    print(f"\n  Fractional error percentiles  |pred − true| / |true|")
    print(f"  (only galaxies with |g_true| ≥ {eps} — the KL shape noise floor)")
    print(f"  {'Percentile':>12}  {'g+':>10}  {'g×':>10}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}")
    frac_vals = {}
    for col, tag in [(0, "g+"), (1, "g×")]:
        true     = te_labels[:, col]
        pred     = te_preds[:, col]
        valid    = np.abs(true) >= eps
        n_excl   = int((~valid).sum())
        abs_frac = np.abs((pred[valid] - true[valid]) / np.abs(true[valid]))
        frac_vals[tag] = (abs_frac, n_excl)
    n_excl_reported = frac_vals["g+"][1]
    if n_excl_reported:
        print(f"  ({n_excl_reported} galaxies excluded per component "
              f"as |g_true| < {eps})")
    for p in pcts:
        gp = np.percentile(frac_vals["g+"][0], p)
        gx = np.percentile(frac_vals["g×"][0], p)
        print(f"  {p:>11}th  {gp*100:>9.1f}%  {gx*100:>9.1f}%")

    gp50 = np.percentile(frac_vals["g+"][0], 50)
    gx50 = np.percentile(frac_vals["g×"][0], 50)
    print(f"\n  Interpretation: for galaxies with |g| ≥ {eps}, 50% of "
          f"predictions are within {gp50*100:.1f}% (g+) / "
          f"{gx50*100:.1f}% (g×) of the true shear value.")

    # ── Also run val set for diagnostic plots ────────────────────────────────
    _, _, _, val_preds, val_labels = run_epoch(
        model, val_loader, criterion, optimizer, device, training=False
    )

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating diagnostic plots …")
    plot_loss_curves(history, out_dir)

    for preds, labels, name in [
        (val_preds,  val_labels,  "val"),
        (te_preds,   te_labels,   "test"),
    ]:
        plot_shear_scatter(preds,  labels, name, out_dir)
        plot_pred_vs_true( preds,  labels, name, out_dir)
        plot_residual_hist(preds,  labels, name, out_dir)

    # ── Save summary ─────────────────────────────────────────────────────────
    summary = {
        "best_epoch":        int(ckpt["epoch"]),
        "best_val_loss":     float(best_val_loss),
        "test_smooth_l1":    float(te_loss),
        "test_mse_g1":       float(te_mse1),
        "test_mse_g2":       float(te_mse2),
        "test_rmse_g1":      float(np.sqrt(te_mse1)),
        "test_rmse_g2":      float(np.sqrt(te_mse2)),
        "n_train":           len(train_ds),
        "n_val":             len(val_ds),
        "n_test":            len(test_ds),
        "n_unique_gal_train": int(df_train["subhalo_id"].nunique()),
        "n_unique_gal_val":   int(df_val["subhalo_id"].nunique()),
        "n_unique_gal_test":  int(df_test["subhalo_id"].nunique()),
    }
    # Add fractional error percentiles to summary
    for col, tag in [(0, "g1"), (1, "g2")]:
        true     = te_labels[:, col]
        pred     = te_preds[:, col]
        valid    = np.abs(true) >= eps   # eps=0.035 set above
        abs_frac = np.abs((pred[valid] - true[valid]) / np.abs(true[valid]))
        for p in pcts:
            summary[f"frac_err_pct{p:02d}_{tag}"] = float(
                np.percentile(abs_frac, p))
        for p in pcts:
            summary[f"abs_err_pct{p:02d}_{tag}"] = float(
                np.percentile(np.abs(pred - true), p))
    pd.DataFrame([summary]).to_csv(out_dir / "test_summary.csv", index=False)

    print(f"\nAll outputs saved to {out_dir.resolve()}")
    print("\nOutput files:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
