#!/usr/bin/env python3
"""
Fix CH2 proton 1H chemical shift annotations in an NMRShiftDB2 pickle dataset
and save the resulting dataset as a new cv_i split (default: --cv_i=5).

Transformations applied to every CH2 group (carbon with exactly 2 explicit H neighbors
h1 < h2 ordered by RDKit atom index):
  1) If only 1 of the 2 protons has an assigned chemical shift, copy that same shift
     to the other proton in the CH2 group.
  2) If both protons have assigned shifts and they differ (|s1 - s2| > tol), sort them
     so that h1 (smaller atom_idx) gets min(s1, s2) and h2 gets max(s1, s2)
     (or descending if --descending is passed).

All other groups (CH, CH3, heteroatom H), molecule metadata, train/test split,
and top-level pickle keys are preserved unchanged.

Recommended usage (inside the `nmrshift` conda environment so the pickle stays
100% compatible with pandas<2 and numpy<2):
    conda activate nmrshift
    python3.10 fix_1h_ch2_dataset.py
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd

from analyze_1h_ch2_ch3 import (
    DEFAULT_INFILE,
    load_pickle_compat,
    analyze_dataframe,
    merge_stats,
    print_report,
)


OUTPUT_TEMPLATE = (
    "graph_conv_many_nuc_pipeline.datasets/"
    "graph_conv_many_nuc_pipeline.data.1H.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.{max_n}.{cv_i}.mol_dict.pickle"
)


def fix_ch2_in_dataframe(df, tol=1e-6, descending=False):
    """Return a copy of `df` with CH2 proton shifts completed and sorted."""
    df_new = df.copy()
    new_values = []
    n_filled = 0
    n_sorted = 0

    for mol, val_list in zip(df["rdmol"], df["value"]):
        # Preserve original list-of-dicts structure (e.g. [shift_dict])
        if isinstance(val_list, (list, tuple)):
            shift_dict = {int(k): float(v) for k, v in val_list[0].items()}
            extra_nucs = list(val_list[1:])
        else:
            shift_dict = {int(k): float(v) for k, v in val_list.items()}
            extra_nucs = None

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:  # Carbon only
                continue
            h_idxs = sorted(nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)
            if len(h_idxs) != 2:
                continue

            h1, h2 = h_idxs[0], h_idxs[1]
            has1 = h1 in shift_dict
            has2 = h2 in shift_dict

            if has1 and not has2:
                shift_dict[h2] = shift_dict[h1]
                n_filled += 1
            elif has2 and not has1:
                shift_dict[h1] = shift_dict[h2]
                n_filled += 1
            elif has1 and has2:
                s1, s2 = shift_dict[h1], shift_dict[h2]
                if abs(s1 - s2) > tol:
                    lo, hi = (s1, s2) if s1 < s2 else (s2, s1)
                    target1, target2 = (hi, lo) if descending else (lo, hi)
                    if shift_dict[h1] != target1 or shift_dict[h2] != target2:
                        shift_dict[h1] = target1
                        shift_dict[h2] = target2
                        n_sorted += 1

        if extra_nucs is not None:
            new_values.append([shift_dict] + extra_nucs)
        else:
            new_values.append(shift_dict)

    df_new["value"] = new_values
    return df_new, n_filled, n_sorted


def main():
    parser = argparse.ArgumentParser(
        description="Fill missing CH2 partner shifts and sort split CH2 proton shifts into a new 1H pickle dataset."
    )
    parser.add_argument("--infile", type=str, default=DEFAULT_INFILE,
                        help="Source 1H .pickle dataset (default: cv_i=1)")
    parser.add_argument("--cv_i", type=int, default=5,
                        help="Target cv_i index for the output filename (default: 5)")
    parser.add_argument("--outfile", type=str, default="",
                        help="Optional explicit output path (overrides --cv_i template)")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Tolerance in ppm for considering two CH2 shifts equal (default: 1e-6)")
    parser.add_argument("--descending", action="store_true",
                        help="Sort split CH2 shifts descending (first > second) instead of ascending (default: ascending)")
    parser.add_argument("--allow-pandas2", action="store_true",
                        help="Allow saving pickle under pandas>=2 / numpy>=2 (NOT readable by pandas<2/numpy<2 in nmrshift env)")
    args = parser.parse_args()

    pd_major = int(pd.__version__.split(".")[0])
    np_major = int(np.__version__.split(".")[0])
    if (pd_major >= 2 or np_major >= 2) and not args.allow_pandas2:
        raise RuntimeError(
            f"Detected pandas={pd.__version__}, numpy={np.__version__}. "
            f"A pickle saved with pandas>=2 or numpy>=2 cannot be loaded in the `nmrshift` conda env (pandas<2, numpy<2). "
            f"Please run inside `conda activate nmrshift` (`python3.10 fix_1h_ch2_dataset.py`), "
            f"or pass --allow-pandas2 if you intentionally want a pandas2/numpy2 pickle."
        )

    if not os.path.exists(args.infile):
        raise FileNotFoundError(f"Source dataset not found: {args.infile}")

    d = load_pickle_compat(args.infile)
    max_n = int(d.get("MAX_N", 64))
    outfile = args.outfile if args.outfile else OUTPUT_TEMPLATE.format(max_n=max_n, cv_i=args.cv_i)

    print(f"Source: {args.infile}")
    print(f"Target: {outfile}")
    print(f"Sort order for split CH2: {'descending (s1 > s2)' if args.descending else 'ascending (s1 < s2)'}")

    train_fixed, tr_filled, tr_sorted = fix_ch2_in_dataframe(d["train_df"], tol=args.tol, descending=args.descending)
    test_fixed, te_filled, te_sorted = fix_ch2_in_dataframe(d["test_df"], tol=args.tol, descending=args.descending)

    print(f"\nTrain CH2 modifications: filled missing partner = {tr_filled}, swapped/sorted = {tr_sorted}")
    print(f"Test  CH2 modifications: filled missing partner = {te_filled}, swapped/sorted = {te_sorted}")

    d_out = dict(d)
    d_out["train_df"] = train_fixed
    d_out["test_df"] = test_fixed

    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    with open(outfile, "wb") as f:
        pickle.dump(d_out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved fixed dataset to: {outfile}")

    # Verify by re-reading and computing statistics
    d_check = load_pickle_compat(outfile)
    s_tr = analyze_dataframe(d_check["train_df"], tol=args.tol)
    s_te = analyze_dataframe(d_check["test_df"], tol=args.tol)
    s_all = merge_stats(s_tr, s_te)
    print_report("VERIFICATION (TOTAL TRAIN + TEST AFTER FIX)", s_all)


if __name__ == "__main__":
    main()
