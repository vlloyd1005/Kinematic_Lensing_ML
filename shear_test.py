#!/usr/bin/env python3
"""
shear_test.py
=============
Injected-shear test of the TwoStreamShearNet on real MaNGA galaxies, using
the metacal-style shear injection in manga_shear.py (colleague's code) and
the existing pkl -> grid -> predict pipeline.

Stages (run one at a time, or `all`):

  prepare   pick N galaxies from the data_info-*.pkl files, run manga_shear.py
            on them for every set below, and convert each set to model-grid
            FITS (convert_manga_pkl.save_grid_files, strict mask only).
  predict   run predict_manga.py on every set (strict mask, no saliency).
  compare   per-galaxy response matrix, multiplicative / additive bias,
            random-shear recovery, summary tables and plots.

Sets (all from the same galaxies; same per-galaxy noise seed in every
processed set, so differences between sets are almost noise-free):

  orig    the pkl files as they are (no processing)
  ctl     manga_shear noshear: same PSF dilation and added noise as the
          sheared sets, g = 0.  THE reference for every response.
  g1p/g1m fixed shear g1 = +g / -g on the sky
  g2p/g2m fixed shear g2 = +g / -g on the sky
  rnd     (--random) a different random (g1, g2) per galaxy, |g_i| <= gmax

Frames.  manga_shear's g is on the sky, x = East, y = North (g2 > 0 along
NE-SW).  The model predicts in the grid frame, x = West, y = North.  So
    g1_grid = g1_sky,   g2_grid = -g2_sky
which `prepare` checks against manga_shear.shear_to_pixel, and which was
verified end to end on synthetic galaxies (a round source sheared by
manga_shear and gridded by convert_manga_pkl elongates with the sign of
g1_sky in e1 and of -g2_sky in e2).

Why responses and not "prediction vs truth": a galaxy's own shape and
kinematics give predictions with a scatter of several 0.01 even at g = 0,
which swamps injected shears of 0.02.  Subtracting the prediction for the
same galaxy in the control set removes that, so with 40 galaxies the
response R = d(pred)/d(g) is measured to a few per cent.  A perfect
estimator has R = identity (R11 = R22 = 1, R12 = R21 = 0).

Usage (Colab)
-------------
  !pip install -q galsim reproject
  !python shear_test.py all \
      --pkl_dir "../drive/MyDrive/Manga data" \
      --shear_code /content/manga_shear_repo \
      --workdir /content/shear_test --n 40 --random \
      --checkpoint kl_dataset/model_output/best_model.pt \
      --grid_manifest /content/Kinematic_Lensing_ML/sdss-manga/fits/grid_manifest.csv

--shear_code is the folder with manga_shear.py, imcom_shear.py and _vendor/.
predict_manga.py, convert_manga_pkl.py and train_kl_model.py must be
importable (same folder as this script, or the working directory).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EST = {"pred": ("g1_pred", "g2_pred"),
       "8fold": ("g1_dihedral_mean", "g2_dihedral_mean")}
C1, C2 = "#2a78d6", "#eb6834"        # categorical slots 1-2 (blue, orange)
PRED_PREFIX = "pred"                  # pred_<set>/ folders (shear_test_uq.py: "pred_uq")
COMPARE_DIR = "compare"               # results folder      (shear_test_uq.py: "compare_uq")


# ═══════════════════════════════════════════════════════════════════════════════
# Frames
# ═══════════════════════════════════════════════════════════════════════════════

def sky_to_grid(g1, g2):
    """manga_shear sky frame (x=E, y=N) -> model grid frame (x=W, y=N)."""
    return np.asarray(g1, float), -np.asarray(g2, float)


def check_frame(ms):
    """Cross-check sky_to_grid against manga_shear's own frame conversion."""
    J_grid = np.diag([-1.0, 1.0])            # grid pixel -> (dE, dN): x = West
    for g in [(0.02, 0.0), (0.0, 0.02), (-0.013, 0.027)]:
        s = ms.shear_to_pixel(g[0], g[1], J_grid)
        a, b = sky_to_grid(*g)
        assert abs(s.g1 - a) < 1e-9 and abs(s.g2 - b) < 1e-9, (g, s, a, b)


# ═══════════════════════════════════════════════════════════════════════════════
# prepare
# ═══════════════════════════════════════════════════════════════════════════════

def grid_stats(path):
    """
    Fill fraction and image coverage per galaxy, read from the FITS headers
    of the grid files (FILL_STR, IMGCOV, SRCFILE).  `path` is the grid folder
    or its grid_manifest.csv; the headers are used either way, so a manifest
    from an older converter without these columns still works.
    """
    folder = Path(path)
    folder = folder.parent if folder.suffix == ".csv" else folder
    rows = []
    from astropy.io import fits
    for f in sorted(folder.glob("*_grid_velmap.fits")):
        h = fits.getheader(f, 0)
        rows.append({"src": str(h.get("SRCFILE", "")).strip(),
                     "mangaid": str(h.get("MANGAID", "")).strip(),
                     "fill_strict": h.get("FILL_STR", np.nan),
                     "img_coverage": h.get("IMGCOV", np.nan)})
    if not rows:
        raise SystemExit(f"no *_grid_velmap.fits files in {folder}")
    df = pd.DataFrame(rows)
    # SRCFILE is truncated at 68 characters in the header; fall back on the mangaid
    df["src"] = np.where(df["src"].str.startswith("data_info-"), df["src"],
                         "data_info-" + df["mangaid"] + ".pkl")
    return df


def select_galaxies(a):
    files = sorted(glob.glob(os.path.join(a.pkl_dir, "data_info-*.pkl")))
    if not files:
        raise SystemExit(f"no data_info-*.pkl files in {a.pkl_dir}")
    by_name = {os.path.basename(f): f for f in files}
    pool = list(by_name)
    if a.grid_manifest:
        man = grid_stats(a.grid_manifest)
        ok = man[(man["fill_strict"] >= a.min_fill)
                 & (man["img_coverage"] >= a.min_img_coverage)]
        pool = [s for s in ok["src"] if s in by_name]
        print(f"{len(pool)}/{len(files)} galaxies pass fill_strict >= {a.min_fill} "
              f"and img_coverage >= {a.min_img_coverage} ({a.grid_manifest})")
    if a.mangaids:
        want = {f"data_info-{m}.pkl" for m in a.mangaids}
        pick = sorted(want & set(pool))
        if len(pick) < len(want):
            print(f"[WARN] not found / cut: {sorted(want - set(pick))}")
    else:
        rng = np.random.default_rng(a.seed)
        pick = sorted(rng.choice(pool, size=min(a.n, len(pool)), replace=False))
    return [by_name[p] for p in pick]


def link_sources(files, src):
    src.mkdir(parents=True, exist_ok=True)
    for f in files:
        dst = src / os.path.basename(f)
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(os.path.abspath(f))
        except OSError:
            shutil.copy2(f, dst)


def import_shear_code(path):
    sys.path.insert(0, str(Path(path).resolve()))
    import manga_shear as ms                          # noqa: E402
    return ms


def stage_prepare(a):
    work = Path(a.workdir); work.mkdir(parents=True, exist_ok=True)
    ms = import_shear_code(a.shear_code)
    check_frame(ms)
    from convert_manga_pkl import save_grid_files

    files = select_galaxies(a)
    print(f"{len(files)} galaxies selected (seed {a.seed})")
    src = work / "src"; link_sources(files, src)
    if a.g > a.gmax:
        raise SystemExit("--g must be <= --gmax")

    g = a.g
    sets = [("ctl", ["noshear"], 0.0, 0.0)]
    sets += [(n, ["fixed", "--g1", str(g1), "--g2", str(g2)], g1, g2)
             for n, g1, g2 in [("g1p", g, 0), ("g1m", -g, 0), ("g2p", 0, g), ("g2m", 0, -g)]]
    if a.random:
        sets.append(("rnd", ["random", "--seed", str(a.shear_seed)], None, None))

    meta = {"n_galaxies": len(files), "g": g, "gmax": a.gmax, "seed": a.seed,
            "frame": "g_grid = (g1_sky, -g2_sky); grid x = West, y = North",
            "sets": {}}
    truth_rows = []

    # orig: no processing, just gridded
    grid = work / "grid_orig"
    save_grid_files(sorted(map(str, src.glob("data_info-*.pkl"))), grid,
                    npix=a.npix, checkpoint=a.checkpoint, strict_only=True,
                    overwrite=a.overwrite, verbose=False)
    meta["sets"]["orig"] = {"grid": str(grid), "g1_sky": 0.0, "g2_sky": 0.0}

    for name, mode, g1, g2 in sets:
        out, grid = work / f"sh_{name}", work / f"grid_{name}"
        truth = work / f"truth_{name}.csv"
        done = len(list(out.glob("data_info-*.pkl"))) == len(files) and truth.exists()
        if a.overwrite or not done:
            print(f"\n── manga_shear: {name} ({' '.join(mode)}) ──")
            ms.main(mode + ["--src", str(src), "--out", str(out), "--truth", str(truth),
                            "--gmax", str(a.gmax), "--jobs", str(a.jobs)]
                    + (["--noise-salt", str(a.noise_salt)] if a.noise_salt else []))
        else:
            print(f"\n── {name}: already sheared, skipping (use --overwrite to redo)")
        save_grid_files(sorted(map(str, out.glob("data_info-*.pkl"))), grid,
                        npix=a.npix, checkpoint=a.checkpoint, strict_only=True,
                        overwrite=a.overwrite or not done, verbose=False)
        meta["sets"][name] = {"grid": str(grid), "g1_sky": g1, "g2_sky": g2}
        t = pd.read_csv(truth)
        for r in t.itertuples():
            gg1, gg2 = sky_to_grid(r.g1, r.g2)
            truth_rows.append({"set": name, "mangaid": r.mangaid, "g1_sky": r.g1,
                               "g2_sky": r.g2, "g1_grid": float(gg1), "g2_grid": float(gg2)})

    pd.DataFrame(truth_rows).to_csv(work / "truth.csv", index=False)
    json.dump(meta, open(work / "sets.json", "w"), indent=1)
    print(f"\nPrepared {len(meta['sets'])} sets in {work}  (truth.csv, sets.json)")


# ═══════════════════════════════════════════════════════════════════════════════
# predict
# ═══════════════════════════════════════════════════════════════════════════════

def find_predict_script(given=None):
    """predict_manga.py: --predict_script, else next to this script or in the
    working directory, else anywhere up to three folders below either."""
    if given:
        if not Path(given).exists():
            raise SystemExit(f"--predict_script {given} does not exist")
        return str(given)
    for base in dict.fromkeys([HERE, Path.cwd()]):
        if (base / "predict_manga.py").exists():
            return str(base / "predict_manga.py")
    for base in dict.fromkeys([HERE, Path.cwd()]):
        hits = sorted(p for d in ("*", "*/*", "*/*/*")
                      for p in base.glob(f"{d}/predict_manga.py"))
        if hits:
            if len(hits) > 1:
                print(f"[NOTE] several predict_manga.py found, using {hits[0]} "
                      f"(others: {', '.join(map(str, hits[1:]))}); "
                      f"pick one with --predict_script")
            return str(hits[0])
    raise SystemExit("predict_manga.py not found next to shear_test.py or in the working "
                     "directory; pass its path with --predict_script")


def stage_predict(a, extra):
    work = Path(a.workdir)
    meta = json.load(open(work / "sets.json"))
    script = find_predict_script(a.predict_script)
    print(f"Using {script}")
    for name, s in meta["sets"].items():
        out = work / f"{PRED_PREFIX}_{name}"
        if (out / "predictions_manga.csv").exists() and not a.overwrite:
            print(f"{name}: predictions exist, skipping")
            continue
        cmd = [sys.executable, script, "--checkpoint", a.checkpoint, "--data_dir", s["grid"],
               "--outdir", str(out), "--vel_masks", "strict", "--n_train_compare", "0",
               "--n_saliency", str(a.n_saliency)] + extra
        print(f"\n── predict: {name} ──\n  {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


# ═══════════════════════════════════════════════════════════════════════════════
# compare
# ═══════════════════════════════════════════════════════════════════════════════

def load_predictions(work, meta):
    out = []
    for name in meta["sets"]:
        f = work / f"{PRED_PREFIX}_{name}" / "predictions_manga.csv"
        if not f.exists():
            print(f"[WARN] no predictions for {name} ({f})")
            continue
        p = pd.read_csv(f)
        p = p[p["vel_mask"] == "strict"]
        cols = ["mangaid", "vel_fill_frac"] + [c for v in EST.values() for c in v]
        out.append(p[cols].assign(set=name))
    return pd.concat(out, ignore_index=True)


def mean_se(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan, len(x)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), len(x)


def ols(x, y):
    """y = a + b1 x1 + b2 x2; returns coefficients and standard errors."""
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    cov = np.linalg.inv(X.T @ X) * (res @ res) / dof
    return beta, np.sqrt(np.diag(cov))


def stage_compare(a):
    work = Path(a.workdir); cmp_dir = work / COMPARE_DIR; cmp_dir.mkdir(exist_ok=True)
    meta = json.load(open(work / "sets.json"))
    truth = pd.read_csv(work / "truth.csv")
    P = load_predictions(work, meta)
    gals = sorted(set.intersection(*[set(P.loc[P["set"] == s, "mangaid"])
                                     for s in ("ctl", "g1p", "g1m", "g2p", "g2m")
                                     if s in set(P["set"])]))
    print(f"{len(gals)} galaxies with predictions in every fixed-shear set")

    def table(est, s):
        c1, c2 = EST[est]
        t = P[P["set"] == s].set_index("mangaid")
        return t.reindex(gals)[[c1, c2]].to_numpy()

    summary, per_gal, plots = [], {"mangaid": gals}, {}
    fill = P[P["set"] == "ctl"].set_index("mangaid").reindex(gals)["vel_fill_frac"]
    per_gal["vel_fill_frac_ctl"] = fill.to_numpy()

    for est in EST:
        ctl = table(est, "ctl")
        # ── Response matrix from the ±g pairs (grid frame) ─────────────────
        R = np.full((len(gals), 2, 2), np.nan)
        for j, (sp, sm) in enumerate([("g1p", "g1m"), ("g2p", "g2m")]):
            gp = np.array(sky_to_grid(meta["sets"][sp]["g1_sky"], meta["sets"][sp]["g2_sky"]))
            gm = np.array(sky_to_grid(meta["sets"][sm]["g1_sky"], meta["sets"][sm]["g2_sky"]))
            R[:, :, j] = (table(est, sp) - table(est, sm)) / (gp - gm)[j]
        for i in range(2):
            for j in range(2):
                m, se, n = mean_se(R[:, i, j])
                med = float(np.nanmedian(R[:, i, j]))
                summary.append({"estimator": est, "quantity": f"R{i+1}{j+1}",
                                "mean": m, "se": se, "median": med, "n": n,
                                "ideal": 1.0 if i == j else 0.0})
                per_gal[f"R{i+1}{j+1}_{est}"] = R[:, i, j]
        # ── Additive bias and the effect of the processing itself ──────────
        orig = table(est, "orig") if "orig" in meta["sets"] else None
        for c in range(2):
            m, se, n = mean_se(ctl[:, c])
            summary.append({"estimator": est, "quantity": f"c{c+1} (mean pred, ctl)",
                            "mean": m, "se": se, "n": n, "ideal": 0.0})
            if orig is not None:
                m, se, n = mean_se(ctl[:, c] - orig[:, c])
                summary.append({"estimator": est, "quantity": f"ctl - orig, g{c+1}",
                                "mean": m, "se": se, "n": n, "ideal": 0.0,
                                "median": float(np.sqrt(np.nanmean((ctl[:, c] - orig[:, c])**2)))})
            per_gal[f"g{c+1}_ctl_{est}"] = ctl[:, c]
        # ── Fixed-shear sets: mean paired response per set ─────────────────
        fixed = {}
        for s in ("g1p", "g1m", "g2p", "g2m"):
            d = table(est, s) - ctl
            gt = np.array(sky_to_grid(meta["sets"][s]["g1_sky"], meta["sets"][s]["g2_sky"]))
            fixed[s] = (gt, d)
        # ── Random set: paired regression on the injected grid shear ───────
        rnd = None
        if "rnd" in meta["sets"] and (P["set"] == "rnd").any():
            t = truth[truth["set"] == "rnd"].set_index("mangaid").reindex(gals)
            gt = t[["g1_grid", "g2_grid"]].to_numpy()
            d = table(est, "rnd") - ctl
            raw = table(est, "rnd")
            ok = np.isfinite(d).all(1) & np.isfinite(gt).all(1)
            for c in range(2):
                beta, se = ols(gt[ok], d[ok, c])
                summary += [
                    {"estimator": est, "quantity": f"rnd: d(g{c+1}) slope on g{c+1}",
                     "mean": beta[1 + c], "se": se[1 + c], "n": int(ok.sum()), "ideal": 1.0},
                    {"estimator": est, "quantity": f"rnd: d(g{c+1}) slope on g{2-c}",
                     "mean": beta[2 - c], "se": se[2 - c], "n": int(ok.sum()), "ideal": 0.0},
                    {"estimator": est, "quantity": f"rnd: d(g{c+1}) intercept",
                     "mean": beta[0], "se": se[0], "n": int(ok.sum()), "ideal": 0.0}]
                b_raw, se_raw = ols(gt[ok], raw[ok, c])
                summary.append({"estimator": est, "quantity": f"rnd UNPAIRED: pred g{c+1} slope",
                                "mean": b_raw[1 + c], "se": se_raw[1 + c], "n": int(ok.sum()),
                                "ideal": 1.0})
            # calibrated recovery: g_hat = <R>^-1 (pred - ctl)
            Rm = np.nanmean(R, axis=0)
            ghat = (np.linalg.inv(Rm) @ d.T).T
            for c in range(2):
                e = ghat[ok, c] - gt[ok, c]
                summary.append({"estimator": est, "quantity": f"rnd: calibrated g{c+1} rms error",
                                "mean": float(np.sqrt(np.mean(e**2))), "se": np.nan,
                                "n": int(ok.sum()),
                                "median": float(np.std(gt[ok, c])), "ideal": 0.0})
                per_gal[f"g{c+1}_true_rnd"] = gt[:, c]
                per_gal[f"d_g{c+1}_rnd_{est}"] = d[:, c]
            rnd = (gt, d, raw)
        plots[est] = dict(R=R, fixed=fixed, rnd=rnd, ctl=ctl, orig=orig)

    S = pd.DataFrame(summary)
    S.to_csv(cmp_dir / "summary.csv", index=False)
    pd.DataFrame(per_gal).to_csv(cmp_dir / "per_galaxy.csv", index=False)
    print_summary(S, meta)
    make_plots(plots, meta, cmp_dir)
    print(f"\nSaved: {cmp_dir}/summary.csv, per_galaxy.csv, response.png, controls.png")


def print_summary(S, meta):
    print(f"\n── Shear response (grid frame; |g| = {meta['g']} per component) ──")
    print("  ideal: R11 = R22 = 1, R12 = R21 = 0, c = 0, slopes 1 (own) / 0 (cross)")
    for est in S["estimator"].unique():
        print(f"\n  [{est}]")
        for r in S[S["estimator"] == est].itertuples():
            if "rms error" in r.quantity:
                print(f"    {r.quantity:<34} {r.mean:.4f}   (rms of injected g: {r.median:.4f})")
                continue
            sig = (r.mean - r.ideal) / r.se if r.se and np.isfinite(r.se) and r.se > 0 else np.nan
            print(f"    {r.quantity:<34} {r.mean:+.4f} ± {r.se:.4f}"
                  + (f"   ({sig:+.1f}σ from {r.ideal:g})" if np.isfinite(sig) else ""))


def make_plots(plots, meta, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False})
    est = "8fold" if "8fold" in plots else next(iter(plots))
    d = plots[est]; g = meta["g"]

    # ── Response: paired prediction change vs injected shear ───────────────
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    lim = 1.6 * max(meta["gmax"], g)
    for c, (axc, col) in enumerate(zip(ax[:2], (C1, C2))):
        if d["rnd"] is not None:
            gt, dd, _ = d["rnd"]
            axc.scatter(gt[:, c], dd[:, c], s=14, color=col, alpha=0.55,
                        label="random set: one galaxy each")
        for s, (gt_s, dd_s) in d["fixed"].items():
            if gt_s[c] == 0:
                continue
            m, se, _ = mean_se(dd_s[:, c])
            axc.errorbar(gt_s[c], m, yerr=se, fmt="s", ms=8, color="k", capsize=3,
                         zorder=5, label="fixed ±g: mean ± s.e." if s.endswith("p") else None)
        axc.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="ideal (slope 1)")
        axc.set_xlim(-lim, lim); axc.set_xlabel(f"injected g{c+1} (grid frame)")
        axc.set_ylabel(f"pred(g) − pred(control), g{c+1}")
        axc.set_title(f"g{c+1} response [{est}]", fontsize=10)
        axc.legend(fontsize=8, loc="upper left")
    R = d["R"]
    bins = np.linspace(np.nanpercentile(R, 2), np.nanpercentile(R, 98), 25)
    for (i, j), col, lab in [((0, 0), C1, "R11"), ((1, 1), C2, "R22")]:
        ax[2].hist(R[:, i, j], bins=bins, histtype="step", lw=2, color=col,
                   label=f"{lab}: mean {np.nanmean(R[:, i, j]):+.2f}")
    for (i, j), col, lab in [((0, 1), C1, "R12"), ((1, 0), C2, "R21")]:
        ax[2].hist(R[:, i, j], bins=bins, histtype="step", lw=1.2, ls="--", color=col,
                   label=f"{lab}: mean {np.nanmean(R[:, i, j]):+.2f}")
    ax[2].axvline(1, color="k", lw=1); ax[2].axvline(0, color="k", lw=0.8, ls=":")
    ax[2].set_xlabel("per-galaxy response (from ±g sets)"); ax[2].set_ylabel("galaxies")
    ax[2].set_title("response matrix elements (ideal: diagonal 1, off-diagonal 0)", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "response.png", dpi=140); plt.close(fig)

    # ── Controls: what the processing alone does, and unpaired vs paired ───
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    if d["orig"] is not None:
        for c, col in enumerate((C1, C2)):
            ax[0].scatter(d["orig"][:, c], d["ctl"][:, c], s=14, color=col, alpha=0.7,
                          label=f"g{c+1}")
        v = np.nanmax(np.abs(np.r_[d["orig"].ravel(), d["ctl"].ravel()])) * 1.1
        ax[0].plot([-v, v], [-v, v], "k--", lw=1)
        ax[0].set_xlabel("prediction, original pkl"); ax[0].set_ylabel("prediction, control (g = 0)")
        ax[0].set_title("effect of PSF dilation + added noise alone", fontsize=10)
        ax[0].legend(fontsize=8)
    if d["rnd"] is not None:
        gt, dd, raw = d["rnd"]
        for c, (axc, col) in enumerate(zip(ax[1:], (C1, C2))):
            axc.scatter(gt[:, c], raw[:, c], s=14, color="0.6", label="unpaired: pred(g)")
            axc.scatter(gt[:, c], dd[:, c], s=14, color=col, label="paired: pred(g) − pred(0)")
            lim = 1.6 * meta["gmax"]
            axc.plot([-lim, lim], [-lim, lim], "k--", lw=1)
            axc.set_xlabel(f"injected g{c+1} (grid frame)"); axc.set_ylabel(f"g{c+1}")
            axc.set_title(f"random set, g{c+1}: why the control is subtracted", fontsize=10)
            axc.legend(fontsize=8)
    else:
        for axc in ax[1:]:
            axc.set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / "controls.png", dpi=140); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["prepare", "predict", "compare", "all"])
    p.add_argument("--workdir", required=True)
    p.add_argument("--pkl_dir", help="folder with data_info-*.pkl (prepare)")
    p.add_argument("--shear_code", help="folder with manga_shear.py, imcom_shear.py, _vendor/")
    p.add_argument("--checkpoint", help="best_model.pt (npix for prepare; model for predict)")
    p.add_argument("--n", type=int, default=40, help="number of galaxies")
    p.add_argument("--seed", type=int, default=0, help="galaxy selection seed")
    p.add_argument("--mangaids", nargs="*", help="use these galaxies instead of a random pick")
    p.add_argument("--grid_manifest", "--grid_dir", dest="grid_manifest",
                   help="folder of *_grid_*.fits from save_grid_files (or its "
                        "grid_manifest.csv), to select galaxies by fill / coverage")
    p.add_argument("--min_fill", type=float, default=0.1)
    p.add_argument("--min_img_coverage", type=float, default=0.3)
    p.add_argument("--g", type=float, default=0.02, help="fixed shear per component")
    p.add_argument("--gmax", type=float, default=0.03, help="manga_shear G_MAX")
    p.add_argument("--random", action="store_true", help="also make a random-shear set")
    p.add_argument("--shear_seed", type=int, default=20261001)
    p.add_argument("--noise_salt", type=int, default=0)
    p.add_argument("--npix", type=int, default=None)
    p.add_argument("--jobs", type=int, default=2, help="parallel jobs for manga_shear")
    p.add_argument("--predict_script", default=None)
    p.add_argument("--n_saliency", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_known_args()


def main():
    a, extra = parse_args()
    if a.stage in ("prepare", "all"):
        for k in ("pkl_dir", "shear_code"):
            if not getattr(a, k):
                raise SystemExit(f"--{k} is required for prepare")
        stage_prepare(a)
    if a.stage in ("predict", "all"):
        if not a.checkpoint:
            raise SystemExit("--checkpoint is required for predict")
        stage_predict(a, extra)
    if a.stage in ("compare", "all"):
        stage_compare(a)


if __name__ == "__main__":
    main()