#!/usr/bin/env python3
"""
Analyze 1H chemical shift annotations for CH2 and CH3 groups in an NMRShiftDB2
pickle dataset (e.g. cv_i=1).

Reports:
  - CH2 groups: counts with 0, 1, or 2 assigned shifts; among 2-shift groups,
    counts of equal vs different shifts, first > second vs first < second
    (ordered by RDKit atom index), and splitting |delta1 - delta2| distribution.
  - CH3 groups (control): counts with 0, 1, 2, or 3 assigned shifts, and counts
    of groups with identical vs different shifts.
"""

import argparse
import os
import pickle
import sys
import types
import numpy as np
import pandas as pd


DEFAULT_INFILE = (
    "graph_conv_many_nuc_pipeline.datasets/"
    "graph_conv_many_nuc_pipeline.data.1H.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.1.mol_dict.pickle"
)


def load_pickle_compat(path):
    """Load dataset pickle created with pandas 1.x even if opened under pandas 2.x."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        if "pandas.core.indexes.numeric" not in str(e):
            raise
        # Compatibility shim for unpickling pandas 1.x Int64Index/Float64Index under pandas 2.x
        mod = types.ModuleType("pandas.core.indexes.numeric")
        mod.Int64Index = pd.Index
        mod.UInt64Index = pd.Index
        mod.Float64Index = pd.Index
        mod.NumericIndex = pd.Index
        sys.modules["pandas.core.indexes.numeric"] = mod
        with open(path, "rb") as f:
            return pickle.load(f)


def analyze_dataframe(df, tol=1e-6):
    stats = {
        "n_mols": len(df),
        "n_total_h_shifts": 0,
        "implicit_h_carbons": 0,
        # CH2 stats
        "ch2_total": 0,
        "ch2_0_shifts": 0,
        "ch2_1_shift": 0,
        "ch2_1_shift_is_first": 0,
        "ch2_1_shift_is_second": 0,
        "ch2_2_shifts": 0,
        "ch2_2_equal": 0,
        "ch2_2_diff": 0,
        "ch2_first_gt_second": 0,
        "ch2_first_lt_second": 0,
        "ch2_diffs_ppm": [],
        # CH3 stats
        "ch3_total": 0,
        "ch3_0_shifts": 0,
        "ch3_1_shift": 0,
        "ch3_2_shifts": 0,
        "ch3_3_shifts": 0,
        "ch3_multi_equal": 0,
        "ch3_multi_diff": 0,
        "ch3_3_equal": 0,
        "ch3_3_diff": 0,
        "ch3_diff_examples": [],
    }

    for mol_i, (mol, val_list) in enumerate(zip(df["rdmol"], df["value"])):
        shift_dict = val_list[0] if isinstance(val_list, (list, tuple)) else val_list
        # Normalize keys to int
        shifts = {int(k): float(v) for k, v in shift_dict.items()}
        stats["n_total_h_shifts"] += len(shifts)

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:  # Carbon only
                continue
            if atom.GetNumImplicitHs() > 0:
                stats["implicit_h_carbons"] += 1

            # Explicit hydrogen neighbors sorted by RDKit atom index
            h_idxs = sorted(nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)
            n_h = len(h_idxs)

            if n_h == 2:
                stats["ch2_total"] += 1
                present = [(h_idx, shifts[h_idx]) for h_idx in h_idxs if h_idx in shifts]
                if len(present) == 0:
                    stats["ch2_0_shifts"] += 1
                elif len(present) == 1:
                    stats["ch2_1_shift"] += 1
                    if present[0][0] == h_idxs[0]:
                        stats["ch2_1_shift_is_first"] += 1
                    else:
                        stats["ch2_1_shift_is_second"] += 1
                elif len(present) == 2:
                    stats["ch2_2_shifts"] += 1
                    s1 = shifts[h_idxs[0]]  # first H (smaller atom_idx)
                    s2 = shifts[h_idxs[1]]  # second H (larger atom_idx)
                    diff = abs(s1 - s2)
                    if diff <= tol:
                        stats["ch2_2_equal"] += 1
                    else:
                        stats["ch2_2_diff"] += 1
                        stats["ch2_diffs_ppm"].append(diff)
                        if s1 > s2:
                            stats["ch2_first_gt_second"] += 1
                        else:
                            stats["ch2_first_lt_second"] += 1

            elif n_h == 3:
                stats["ch3_total"] += 1
                present_vals = [shifts[h_idx] for h_idx in h_idxs if h_idx in shifts]
                k = len(present_vals)
                if k == 0:
                    stats["ch3_0_shifts"] += 1
                elif k == 1:
                    stats["ch3_1_shift"] += 1
                elif k == 2:
                    stats["ch3_2_shifts"] += 1
                elif k == 3:
                    stats["ch3_3_shifts"] += 1

                if k >= 2:
                    spread = max(present_vals) - min(present_vals)
                    if spread <= tol:
                        stats["ch3_multi_equal"] += 1
                        if k == 3:
                            stats["ch3_3_equal"] += 1
                    else:
                        stats["ch3_multi_diff"] += 1
                        if k == 3:
                            stats["ch3_3_diff"] += 1
                        if len(stats["ch3_diff_examples"]) < 10:
                            stats["ch3_diff_examples"].append((mol_i, atom.GetIdx(), present_vals))

    return stats


def merge_stats(s1, s2):
    out = {}
    for k in s1:
        if isinstance(s1[k], list):
            out[k] = s1[k] + s2[k]
        else:
            out[k] = s1[k] + s2[k]
    return out


def pct(part, total):
    return (100.0 * part / total) if total > 0 else 0.0


def print_report(title, s):
    print("=" * 72)
    print(f" {title} (molecules: {s['n_mols']}, total 1H shifts: {s['n_total_h_shifts']})")
    if s["implicit_h_carbons"] > 0:
        print(f" WARNING: carbons with implicit Hs: {s['implicit_h_carbons']}")
    print("=" * 72)

    ch2 = s["ch2_total"]
    print(f"\n[CH2 groups] Total: {ch2}")
    print(f"  0 shifts assigned : {s['ch2_0_shifts']:6d} ({pct(s['ch2_0_shifts'], ch2):5.2f}%)")
    print(f"  1 shift  assigned : {s['ch2_1_shift']:6d} ({pct(s['ch2_1_shift'], ch2):5.2f}%)"
          f"  [first H: {s['ch2_1_shift_is_first']}, second H: {s['ch2_1_shift_is_second']}]")
    print(f"  2 shifts assigned : {s['ch2_2_shifts']:6d} ({pct(s['ch2_2_shifts'], ch2):5.2f}%)")

    c2 = s["ch2_2_shifts"]
    if c2 > 0:
        print(f"    - equal (|s1 - s2| <= tol) : {s['ch2_2_equal']:6d} ({pct(s['ch2_2_equal'], c2):5.2f}% of 2-shift CH2)")
        print(f"    - different                : {s['ch2_2_diff']:6d} ({pct(s['ch2_2_diff'], c2):5.2f}% of 2-shift CH2)")
        cd = s["ch2_2_diff"]
        if cd > 0:
            print(f"        * first > second (s1 > s2): {s['ch2_first_gt_second']:6d} ({pct(s['ch2_first_gt_second'], cd):5.2f}%)")
            print(f"        * first < second (s1 < s2): {s['ch2_first_lt_second']:6d} ({pct(s['ch2_first_lt_second'], cd):5.2f}%)")
            diffs = np.array(s["ch2_diffs_ppm"], dtype=np.float64)
            print(
                f"        * |s1 - s2| [ppm]: mean={diffs.mean():.4f}, median={np.median(diffs):.4f}, "
                f"p90={np.percentile(diffs, 90):.4f}, p95={np.percentile(diffs, 95):.4f}, max={diffs.max():.4f}"
            )
            print(
                f"        * irreducible half-split |s1 - s2|/2 [ppm]: mean={0.5*diffs.mean():.4f}, "
                f"median={0.5*np.median(diffs):.4f}, p95={0.5*np.percentile(diffs, 95):.4f}"
            )

    ch3 = s["ch3_total"]
    print(f"\n[CH3 groups (control)] Total: {ch3}")
    print(f"  0 shifts assigned : {s['ch3_0_shifts']:6d} ({pct(s['ch3_0_shifts'], ch3):5.2f}%)")
    print(f"  1 shift  assigned : {s['ch3_1_shift']:6d} ({pct(s['ch3_1_shift'], ch3):5.2f}%)")
    print(f"  2 shifts assigned : {s['ch3_2_shifts']:6d} ({pct(s['ch3_2_shifts'], ch3):5.2f}%)")
    print(f"  3 shifts assigned : {s['ch3_3_shifts']:6d} ({pct(s['ch3_3_shifts'], ch3):5.2f}%)")
    multi = s["ch3_2_shifts"] + s["ch3_3_shifts"]
    if multi > 0:
        print(f"  Among CH3 with >=2 shifts ({multi}):")
        print(f"    - all equal     : {s['ch3_multi_equal']:6d} ({pct(s['ch3_multi_equal'], multi):5.2f}%)")
        print(f"    - different     : {s['ch3_multi_diff']:6d} ({pct(s['ch3_multi_diff'], multi):5.2f}%)")
        if s["ch3_3_shifts"] > 0:
            print(f"      (among 3-shift CH3: equal={s['ch3_3_equal']}, different={s['ch3_3_diff']})")
        if s["ch3_diff_examples"]:
            print("    - sample CH3 with different shifts (mol_idx, C_idx, shifts):")
            for ex in s["ch3_diff_examples"][:5]:
                print(f"        mol={ex[0]}, C={ex[1]}, shifts={ex[2]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze CH2 and CH3 1H chemical shift statistics.")
    parser.add_argument("--infile", type=str, default=DEFAULT_INFILE,
                        help="Path to 1H dataset .pickle file")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Tolerance in ppm for considering two shifts equal (default: 1e-6)")
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        raise FileNotFoundError(f"Dataset file not found: {args.infile}")

    d = load_pickle_compat(args.infile)
    print(f"Loaded: {args.infile}")
    print(f"tgt_nucs={d.get('tgt_nucs')}, MAX_N={d.get('MAX_N')}")

    s_train = analyze_dataframe(d["train_df"], tol=args.tol)
    s_test = analyze_dataframe(d["test_df"], tol=args.tol)
    s_all = merge_stats(s_train, s_test)

    print_report("TRAIN", s_train)
    print_report("TEST", s_test)
    print_report("TOTAL (TRAIN + TEST)", s_all)


if __name__ == "__main__":
    main()
