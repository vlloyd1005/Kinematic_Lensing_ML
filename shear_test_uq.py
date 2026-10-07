#!/usr/bin/env python3
"""
shear_test_uq.py
================
The injected-shear test of shear_test.py, run with the uncertainty-aware
ensemble (train_kl_model_uq.py, predicted by predict_manga_uq.py).

It uses the SAME prepared sets as shear_test.py (orig, ctl, g1p, g1m, g2p,
g2m, rnd in the same --workdir), so if `shear_test.py prepare` has run,
nothing is sheared again.  Predictions go to pred_uq_<set>/ and results to
compare_uq/, next to the single-model ones.

compare reports
  1. everything shear_test.py reports (response matrix R, additive bias,
     random-set slopes), for the ensemble mean;
  2. what the error bars add -- with the injected shear known:
       calibration    z = (pred − g_true) / σ per set; rms(z) ≈ 1 and 68 / 95 %
                      coverage if σ is right on real (sheared) galaxies.  Also
                      after correcting for the measured response,
                      z_R = (pred − ⟨R⟩ g_true − ⟨c⟩) / σ, which separates "σ too
                      small" from "response below 1".
       σ stability    σ(sheared) / σ(control) per galaxy; ≈ 1, because the
                      uncertainty should not depend on the (small) shear.
       recovery       inverse-variance weighted mean of the predictions in each
                      fixed-shear set, with its error, against the injected g:
                      what a survey-style estimate from these 40 galaxies gives.
       uncertainty budget  aleatoric vs epistemic share per set.

Usage
-----
  # sets already made by shear_test.py prepare:
  python shear_test_uq.py predict --workdir shear_test --ensemble_dir kl_dataset/model_output_uq
  python shear_test_uq.py compare --workdir shear_test

  # or from scratch (same options as shear_test.py):
  python shear_test_uq.py all --workdir shear_test --n 40 --random \
      --pkl_dir "../drive/MyDrive/Manga data" --shear_code /path/to/his/folder \
      --ensemble_dir kl_dataset/model_output_uq \
      --grid_dir /content/Kinematic_Lensing_ML/sdss-manga/fits
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import shear_test as st

st.PRED_PREFIX = "pred_uq"
st.COMPARE_DIR = "compare_uq"
COMPS = [(0, "g1", "g+"), (1, "g2", "g×")]


def find_uq_script(given=None):
    if given:
        if not Path(given).exists():
            raise SystemExit(f"--predict_script {given} does not exist")
        return str(given)
    for base in dict.fromkeys([st.HERE, Path.cwd()]):
        for d in ("", "*/", "*/*/", "*/*/*/"):
            hits = sorted(base.glob(f"{d}predict_manga_uq.py"))
            if hits:
                return str(hits[0])
    raise SystemExit("predict_manga_uq.py not found; pass its path with --predict_script")


def first_member(path):
    """A checkpoint for prepare (npix): the given file, or member_0 of the ensemble."""
    p = Path(path)
    if p.is_file():
        return str(p)
    hits = sorted(p.glob("member_*/best_model.pt")) or sorted(p.parent.glob("member_*/best_model.pt"))
    if not hits:
        raise SystemExit(f"no member_*/best_model.pt under {p}")
    return str(hits[0])


def stage_predict_uq(a, extra):
    work = Path(a.workdir)
    meta = json.load(open(work / "sets.json"))
    script = find_uq_script(a.predict_script)
    print(f"Using {script}  (ensemble: {a.ensemble_dir})")
    for name, s in meta["sets"].items():
        out = work / f"{st.PRED_PREFIX}_{name}"
        if (out / "predictions_manga_uq.csv").exists() and not a.overwrite:
            print(f"{name}: predictions exist, skipping")
            continue
        cmd = [sys.executable, script, "--ensemble_dir", a.ensemble_dir,
               "--data_dir", s["grid"], "--outdir", str(out), "--vel_masks", "strict",
               "--n_train_compare", "0", "--n_saliency", str(a.n_saliency)] + extra
        print(f"\n── predict (ensemble): {name} ──\n  {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


# ═══════════════════════════════════════════════════════════════════════════════
# What the error bars add
# ═══════════════════════════════════════════════════════════════════════════════

def load_uq(work, meta):
    rows = []
    for name in meta["sets"]:
        f = work / f"{st.PRED_PREFIX}_{name}" / "predictions_manga_uq.csv"
        if not f.exists():
            print(f"[WARN] no ensemble predictions for {name} ({f})")
            continue
        p = pd.read_csv(f)
        rows.append(p[p["vel_mask"] == "strict"].assign(set=name))
    return pd.concat(rows, ignore_index=True)


def wmean(x, s):
    w = 1 / s**2
    return float((w * x).sum() / w.sum()), float(1 / np.sqrt(w.sum()))


def uq_analysis(work, meta):
    out_dir = work / st.COMPARE_DIR
    P = load_uq(work, meta)
    truth = pd.read_csv(work / "truth.csv")
    # truth in the grid frame per (set, galaxy); orig has no shear
    T = truth[["set", "mangaid", "g1_grid", "g2_grid"]]
    orig = P.loc[P["set"] == "orig", ["mangaid"]].assign(set="orig", g1_grid=0.0, g2_grid=0.0)
    T = pd.concat([T, orig], ignore_index=True)
    D = P.merge(T, on=["set", "mangaid"], how="left")

    # mean response and additive bias (from shear_test's own summary)
    S = pd.read_csv(out_dir / "summary.csv")
    S = S[S["estimator"] == "pred"].set_index("quantity")["mean"]
    R = np.array([[S["R11"], S["R12"]], [S["R21"], S["R22"]]])
    c = np.array([S["c1 (mean pred, ctl)"], S["c2 (mean pred, ctl)"]])

    g  = D[["g1_pred", "g2_pred"]].to_numpy()
    gt = D[["g1_grid", "g2_grid"]].to_numpy()
    sg = D[["g1_sigma", "g2_sigma"]].to_numpy()
    D[["z1", "z2"]]     = (g - gt) / sg
    D[["zR1", "zR2"]]   = (g - gt @ R.T - c) / sg

    ctl = D[D["set"] == "ctl"].set_index("mangaid")
    rows = []
    print(f"\n── Error bars on real galaxies with known injected shear "
          f"(ensemble, calibrated σ) ──")
    print("  ideal: rms(z) = 1, |z|<1 in 68.3%, |z|<2 in 95.4%, σ ratio = 1, "
          "weighted mean = injected g")
    hdr = (f"  {'set':<5} {'comp':<3} {'med σ':>7} {'rms z':>6} {'<1σ':>5} {'<2σ':>5} "
           f"{'rms z_R':>7} {'σ/σ_ctl':>8} {'epi%':>5} {'inj g':>7} {'wmean ± err':>17}")
    print(hdr)
    for name in meta["sets"]:
        sub = D[D["set"] == name]
        if sub.empty:
            continue
        for ci, tag, sym in COMPS:
            z, zR = sub[f"z{ci+1}"].to_numpy(), sub[f"zR{ci+1}"].to_numpy()
            s = sub[f"{tag}_sigma"].to_numpy()
            ratio = (sub.set_index("mangaid")[f"{tag}_sigma"]
                     / ctl[f"{tag}_sigma"]).dropna().to_numpy()
            epi = np.mean(sub[f"{tag}_sigma_epi"]**2 / sub[f"{tag}_sigma_raw"]**2)
            wm, we = wmean(sub[f"{tag}_pred"].to_numpy(), s)
            inj = float(sub[f"{tag}_grid"].mean())
            r = {"set": name, "comp": tag, "n": len(sub), "median_sigma": float(np.median(s)),
                 "rms_z": float(np.sqrt(np.mean(z**2))), "cov1": float(np.mean(np.abs(z) < 1)),
                 "cov2": float(np.mean(np.abs(z) < 2)),
                 "rms_z_response_corrected": float(np.sqrt(np.mean(zR**2))),
                 "sigma_ratio_to_ctl": float(np.median(ratio)) if len(ratio) else np.nan,
                 "epistemic_share": float(epi), "injected_mean": inj,
                 "wmean_pred": wm, "wmean_err": we}
            rows.append(r)
            inj_s = f"{inj:+.3f}" if name != "rnd" else "  rand"
            print(f"  {name:<5} {sym:<3} {r['median_sigma']:7.4f} {r['rms_z']:6.2f} "
                  f"{r['cov1']:5.0%} {r['cov2']:5.0%} {r['rms_z_response_corrected']:7.2f} "
                  f"{r['sigma_ratio_to_ctl']:8.3f} {epi:5.0%} {inj_s:>7} "
                  f"{wm:+.4f} ± {we:.4f}")
    U = pd.DataFrame(rows)
    U.to_csv(out_dir / "uq_summary.csv", index=False)
    D.to_csv(out_dir / "uq_per_galaxy.csv", index=False)

    allz = D[D["set"] != "orig"][["z1", "z2"]].to_numpy()
    print(f"\n  All processed sets pooled: rms(z) g+ {np.sqrt(np.mean(allz[:, 0]**2)):.2f}, "
          f"g× {np.sqrt(np.mean(allz[:, 1]**2)):.2f}")
    print("  rms(z) > 1: σ too small on real data; rms z_R ≈ 1 while rms z > 1 means the "
          "excess comes from the response, not from σ.")
    plot_uq(D, U, meta, out_dir)
    print(f"\nSaved: {out_dir}/uq_summary.csv, uq_per_galaxy.csv, uq_shear_test.png")


def plot_uq(D, U, meta, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    # 1. weighted-mean recovery in the fixed sets
    lim = 1.8 * meta["g"]
    for ci, tag, sym in COMPS:
        col = (st.C1, st.C2)[ci]
        sub = U[(U["comp"] == tag) & U["set"].isin(["ctl", "g1p", "g1m", "g2p", "g2m"])]
        own = sub[sub["set"].str.startswith(tag) | (sub["set"] == "ctl")]
        ax[0].errorbar(own["injected_mean"] + (ci - 0.5) * 0.001, own["wmean_pred"],
                       yerr=own["wmean_err"], fmt="o", ms=7, capsize=3, color=col,
                       label=f"{sym}: weighted mean ± error")
    ax[0].plot([-lim, lim], [-lim, lim], "k--", lw=1, label="ideal")
    ax[0].set_xlabel("injected g (grid frame), own component")
    ax[0].set_ylabel("inverse-variance weighted mean prediction")
    ax[0].set_title("recovery from the ensemble with its error bars", fontsize=10)
    ax[0].legend(fontsize=8)

    # 2. z distribution, all processed sets pooled
    proc = D[D["set"] != "orig"]
    xs = np.linspace(-5, 5, 200); bins = np.linspace(-5, 5, 41)
    for ci, (sym, col) in enumerate([("g+", st.C1), ("g×", st.C2)]):
        z = proc[f"z{ci+1}"].to_numpy()
        ax[1].hist(np.clip(z, -5, 5), bins=bins, density=True, histtype="step", lw=2,
                   color=col, label=f"{sym}: rms {np.sqrt(np.mean(z**2)):.2f}")
    ax[1].plot(xs, norm.pdf(xs), "k--", lw=1, label="N(0, 1)")
    ax[1].set_xlabel("z = (pred − injected) / σ"); ax[1].set_ylabel("density")
    ax[1].set_title("are the error bars right on real galaxies?", fontsize=10)
    ax[1].legend(fontsize=8)

    # 3. σ stability
    ctl = D[D["set"] == "ctl"].set_index("mangaid")
    sh = D[~D["set"].isin(["ctl", "orig"])]
    for ci, tag, sym in COMPS:
        r = (sh.set_index("mangaid")[f"{tag}_sigma"] / ctl[f"{tag}_sigma"]).dropna()
        ax[2].hist(r, bins=30, histtype="step", lw=2, color=(st.C1, st.C2)[ci],
                   label=f"{sym}: median {np.median(r):.3f}")
    ax[2].axvline(1, color="k", lw=1)
    ax[2].set_xlabel("σ(sheared) / σ(control), per galaxy and set")
    ax[2].set_ylabel("count"); ax[2].set_title("σ should not respond to the shear", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "uq_shear_test.png", dpi=140); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI: shear_test.py's options, with the ensemble instead of one checkpoint
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    argv = sys.argv[1:]
    ens = None
    for flag in ("--ensemble_dir", "--checkpoint"):
        if flag in argv:
            i = argv.index(flag); ens = argv[i + 1]; del argv[i:i + 2]
    sys.argv = [sys.argv[0]] + argv
    a, extra = st.parse_args()
    a.ensemble_dir = ens
    if a.stage in ("prepare", "all"):
        for k in ("pkl_dir", "shear_code"):
            if not getattr(a, k):
                raise SystemExit(f"--{k} is required for prepare")
        a.checkpoint = first_member(ens) if ens else a.checkpoint
        st.stage_prepare(a)
    if a.stage in ("predict", "all"):
        if not ens:
            raise SystemExit("--ensemble_dir (model_output_uq) is required for predict")
        stage_predict_uq(a, extra)
    if a.stage in ("compare", "all"):
        st.EST = {"pred": ("g1_pred", "g2_pred"), "8fold": ("g1_dihedral_mean", "g2_dihedral_mean")}
        st.stage_compare(a)                  # response matrix etc. for the ensemble mean
        uq_analysis(Path(a.workdir), json.load(open(Path(a.workdir) / "sets.json")))


if __name__ == "__main__":
    main()
