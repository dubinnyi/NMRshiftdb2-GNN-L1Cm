"""
Hyperparameter sweep for `dropout` and `weight_decay` on NMRResPredictV2
for both 1H and 13C nuclei.

Usage:
    python3.10 sweep_reg.py
    python3.10 sweep_reg.py --nuclei 1H --max_epochs 25
    python3.10 sweep_reg.py --full_grid
    python3.10 sweep_reg.py --smoke_test
"""

import argparse
import os
import re
import subprocess
import sys
import time

import pandas as pd


FOCUSED_GRID = [
    (0.01, 1e-6),
    (0.05, 1e-5),
    (0.05, 1e-4),
    (0.10, 1e-4),
    (0.10, 1e-3),
    (0.15, 1e-4),
]

FULL_DROPOUTS = [0.01, 0.05, 0.10, 0.15]
FULL_WEIGHT_DECAYS = [1e-6, 1e-4, 1e-3]


def run_trial(py_exe, nucleus, cv_i, dropout, weight_decay, max_epochs, init_std, plot_every, smoke_test=False):
    cmd = [
        py_exe,
        "respredict_pipeline.py",
        "--arch", "v2",
        "--loss", "beta_nll",
        "--nucleus", nucleus,
        "--cv_i", str(cv_i),
        "--dropout", str(dropout),
        "--weight_decay", str(weight_decay),
        "--max_epochs", str(max_epochs),
        "--eval_every", "1",
        "--plot_every", str(plot_every),
    ]
    if init_std is not None:
        cmd += ["--init_std", str(init_std)]
    if smoke_test:
        cmd += ["--smoke_test"]

    print("\n" + "#" * 78)
    print(f"[SWEEP] Nucleus={nucleus} (cv_i={cv_i}) | dropout={dropout} | weight_decay={weight_decay} | init_std={init_std}")
    print("CMD:", " ".join(cmd))
    print("#" * 78, flush=True)

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    metrics_csv = None
    best_epoch = None
    best_state = None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m_csv = re.search(r"Metrics CSV file:\s+(\S+)", line)
        if m_csv:
            metrics_csv = m_csv.group(1).strip()
        m_best = re.search(r"Restoring best model weights \(epoch (\d+),", line)
        if m_best:
            best_epoch = int(m_best.group(1))
        if ".best.state" in line:
            best_state = line.strip()

    ret = proc.wait()
    elapsed_min = (time.time() - t0) / 60.0
    if ret != 0:
        raise RuntimeError(f"Trial failed (exit={ret}) for {nucleus} dropout={dropout} wd={weight_decay}")

    row = {
        "nucleus": nucleus,
        "cv_i": cv_i,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "init_std": init_std,
        "best_epoch": best_epoch,
        "elapsed_min": round(elapsed_min, 2),
        "metrics_csv": metrics_csv or "",
        "best_state": best_state or "",
    }

    if metrics_csv and os.path.exists(metrics_csv):
        df = pd.read_csv(metrics_csv)
        if len(df) > 0:
            row["stopped_epoch"] = int(df["epoch"].max())
            if best_epoch is not None and (df["epoch"] == best_epoch).any():
                r = df[df["epoch"] == best_epoch].iloc[0]
            else:
                r = df.loc[df["te_nll_raw"].idxmin()]
                row["best_epoch"] = int(r["epoch"])

            for col in [
                "tr_mae", "te_mae", "te_mae_cal", "te_rmse",
                "tr_nll_raw", "te_nll_raw",
                "te_std_mean", "te_std_alea_mean", "te_std_epi_mean",
                "te_cov68", "te_cov95",
                "tr_nu_raw", "te_nu_raw", "d_nu_raw", "te_sc_raw",
            ]:
                if col in r.index:
                    row[col] = float(r[col])

    return row


def main():
    parser = argparse.ArgumentParser(description="Dropout & Weight Decay sweep for NMRResPredictV2")
    parser.add_argument("--nuclei", nargs="+", default=["1H", "13C"], choices=["1H", "13C"])
    parser.add_argument("--cv_1h", type=int, default=5)
    parser.add_argument("--cv_13c", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--init_std_13c", type=float, default=2.5)
    parser.add_argument("--init_std_1h", type=float, default=1.0)
    parser.add_argument("--plot_every", type=int, default=10)
    parser.add_argument("--full_grid", action="store_true",
                        help="Run full 4x3=12 grid instead of the 6-point focused grid")
    parser.add_argument("--out_csv", type=str, default="sweep_reg_results.csv")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        grid = [(0.05, 1e-4)]
    elif args.full_grid:
        grid = [(d, wd) for d in FULL_DROPOUTS for wd in FULL_WEIGHT_DECAYS]
    else:
        grid = FOCUSED_GRID

    results = []
    for nuc in args.nuclei:
        cv_i = args.cv_1h if nuc == "1H" else args.cv_13c
        init_std = args.init_std_1h if nuc == "1H" else args.init_std_13c
        for dropout, wd in grid:
            row = run_trial(
                sys.executable, nuc, cv_i, dropout, wd,
                args.max_epochs, init_std, args.plot_every, smoke_test=args.smoke_test,
            )
            results.append(row)
            if not args.smoke_test:
                pd.DataFrame(results).to_csv(args.out_csv, index=False, float_format="%.6f")
                print(f"[SWEEP] Updated {args.out_csv} ({len(results)} rows)", flush=True)

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 78)
    print("SWEEP SUMMARY (sorted by te_nll_raw ascending per nucleus)")
    print("=" * 78)
    for nuc in args.nuclei:
        sub = df_res[df_res["nucleus"] == nuc].copy()
        if "te_nll_raw" in sub.columns:
            sub = sub.sort_values("te_nll_raw", ascending=True)
        cols = [c for c in [
            "nucleus", "dropout", "weight_decay", "best_epoch", "stopped_epoch",
            "te_nll_raw", "te_mae", "te_mae_cal", "te_std_mean",
            "te_cov68", "te_cov95", "d_nu_raw", "te_sc_raw", "elapsed_min"
        ] if c in sub.columns]
        print(f"\n--- {nuc} ---")
        print(sub[cols].to_string(index=False))


if __name__ == "__main__":
    main()
