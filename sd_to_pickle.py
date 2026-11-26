import argparse
import hashlib
import os
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

from i18n import _

# Silence noisy RDKit SDF/sanitization warnings during bulk streaming
RDLogger.DisableLog('rdApp.*')

# === DEFAULT CONFIGURATION ===
DEFAULT_SDF_FILE = "nmrshiftdb2/nmrshiftdb2withsignals_3d.sd"
DEFAULT_OUTPUT_DIR = "graph_conv_many_nuc_pipeline.datasets"
DEFAULT_OLD_13C = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "graph_conv_many_nuc_pipeline.data.13C.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle",
)
DEFAULT_OLD_1H = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "graph_conv_many_nuc_pipeline.data.1H.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.1.mol_dict.pickle",
)
DEFAULT_MAX_N = 64  # Maximum atom count INCLUDING explicit hydrogens
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_CV_I = 3
DEFAULT_SEED = 42

ALLOWED_ATOMIC_NUMS = {1, 6, 7, 8, 9, 15, 16, 17}  # H, C, N, O, F, P, S, Cl
SHIFT_BOUNDS = {
    '13C': (-20.0, 260.0),
    '1H': (-5.0, 25.0),
}
CH3_DIFF_TOL = 1e-4  # Tolerance (ppm) to treat two CH3 proton shifts as identical

# Filename template expected by respredict_pipeline.py
FILENAME_TEMPLATE = (
    "graph_conv_many_nuc_pipeline.data.{nuc}.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.{max_n}.{cv}.mol_dict.pickle"
)


def load_pickle_compat(filepath):
    """
    Load a pickled dataset dict produced by pandas 1.x / numpy 1.x even if
    running under pandas 2.x / numpy 2.x.
    """
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        pass

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'pandas.core.indexes.numeric':
                return pd.Index
            if module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core', 1)
            return super().find_class(module, name)

    with open(filepath, 'rb') as f:
        return _CompatUnpickler(f).load()


def check_pickle_write_env(allow_pandas2=False):
    """Guard against accidentally writing pandas 2.x / numpy 2.x pickles for pandas 1.x servers."""
    pd_major = int(pd.__version__.split('.')[0])
    np_major = int(np.__version__.split('.')[0])
    if (pd_major >= 2 or np_major >= 2) and not allow_pandas2:
        raise RuntimeError(
            f"Refusing to write production .pickle under pandas={pd.__version__}, numpy={np.__version__} "
            f"(incompatible with pandas<2 / numpy<2 in the 'nmrshift' conda environment). "
            f"Run inside 'conda activate nmrshift', or pass --stats_only for dry-run / --allow_pandas2 to override."
        )


def parse_spectrum_string(spec_str):
    """Parse an NMRShiftDB spectrum string '17.6;0.0Q;10|...' into {atom_idx: [shifts]}."""
    shifts_dict = {}
    if not isinstance(spec_str, str):
        spec_str = str(spec_str)
    for peak in spec_str.split('|'):
        peak = peak.strip()
        if not peak:
            continue
        parts = peak.split(';')
        if len(parts) >= 3:
            try:
                shift = float(parts[0])
                if not np.isfinite(shift):
                    continue
                for aid in parts[2].split(','):
                    aid = aid.strip()
                    if aid.isdigit():
                        idx = int(aid)
                        shifts_dict.setdefault(idx, []).append(shift)
            except ValueError:
                continue
    return shifts_dict


def get_canonical_smiles(mol):
    """
    Canonical non-isomeric connectivity SMILES (after removing explicit Hs).
    Grouping train/test by connectivity SMILES guarantees that multiple instances
    of the same molecule (even if one source has 3D/stereo tags and another does not)
    NEVER appear in both train and test.
    """
    mol_no_h = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol_no_h, isomericSmiles=False, canonical=True)


def _is_valid_3d_conformer(mol_h):
    """
    Verify that mol_h has a physically sane 3D conformer:
      - at least 1 conformer with finite coordinates;
      - at most 1 atom at the exact origin (0, 0, 0);
      - all bonded atom pairs have distance in [0.5, 3.0] A;
      - if molecule has any sp3 atom, z-coordinates must not be completely flat.
    """
    if mol_h.GetNumConformers() == 0:
        return False
    conf = mol_h.GetConformer(0)
    pos = conf.GetPositions()
    if pos.shape[0] != mol_h.GetNumAtoms() or not np.isfinite(pos).all():
        return False
    if pos.shape[0] > 1 and int(np.sum(np.linalg.norm(pos, axis=1) < 1e-6)) > 1:
        return False

    for b in mol_h.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        d = float(np.linalg.norm(pos[i] - pos[j]))
        if d < 0.5 or d > 3.0:
            return False

    has_sp3 = any(
        a.GetHybridization() == Chem.rdchem.HybridizationType.SP3
        for a in mol_h.GetAtoms()
        if a.GetAtomicNum() > 1
    )
    if has_sp3 and float(np.std(pos[:, 2])) < 1e-3:
        return False

    return True


def ensure_valid_3d(mol, stats, seed=42):
    """
    Add explicit hydrogens (preserving original atom indices 0..N_orig-1 and 3D coordinates
    when available) and validate/repair 3D conformer via ETKDGv3 + MMFF/UFF if broken.
    """
    has_conf = mol.GetNumConformers() > 0
    mol_h = Chem.AddHs(mol, addCoords=has_conf)

    if _is_valid_3d_conformer(mol_h):
        stats['geom_valid_as_is'] += 1
        return mol_h

    # Repair 3D conformer in-place (preserving exact atom indices!)
    mol_rep = Chem.Mol(mol_h)
    mol_rep.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    res = AllChem.EmbedMolecule(mol_rep, params)
    if res != 0:
        params.useRandomCoords = True
        res = AllChem.EmbedMolecule(mol_rep, params)
    if res == 0:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol_rep):
                AllChem.MMFFOptimizeMolecule(mol_rep, maxIters=200)
            else:
                AllChem.UFFOptimizeMolecule(mol_rep, maxIters=200)
        except Exception:
            pass
        if _is_valid_3d_conformer(mol_rep):
            stats['geom_repaired_etkdg'] += 1
            return mol_rep

    stats['geom_dropped_bad_3d'] += 1
    return None


def passes_molecule_filters(mol_h, max_n, stats):
    """Check single-fragment, allowed elements (HCONFSPCl), zero radicals, and atom count <= max_n."""
    if mol_h.GetNumAtoms() > max_n:
        stats['filter_too_large'] += 1
        return False
    if len(Chem.GetMolFrags(mol_h)) != 1:
        stats['filter_multi_frag'] += 1
        return False
    for a in mol_h.GetAtoms():
        if a.GetAtomicNum() not in ALLOWED_ATOMIC_NUMS:
            stats['filter_disallowed_elem'] += 1
            return False
        if a.GetNumRadicalElectrons() != 0:
            stats['filter_radical'] += 1
            return False
    return True


def clean_13c_shifts(mol_h, raw_shifts, stats):
    """
    Validate and map 13C chemical shifts to carbon atoms only, filtering out-of-range values
    and averaging duplicate peaks on the same carbon.
    """
    lo, hi = SHIFT_BOUNDS['13C']
    final_13c = {}
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        idx = atom.GetIdx()
        if idx not in raw_shifts:
            continue
        vals = [float(v) for v in raw_shifts[idx] if lo <= float(v) <= hi]
        n_oob = len(raw_shifts[idx]) - len(vals)
        if n_oob > 0:
            stats['13c_oob_peaks_dropped'] += n_oob
        if vals:
            final_13c[idx] = float(np.mean(vals))
    return final_13c


def clean_1h_shifts(mol_h, raw_shifts, stats):
    """
    Clean and assign 1H chemical shifts to explicit H atoms with full CH3/CH2 hygiene:
      - Reads BOTH heavy-atom-indexed and explicit-H-indexed signals in raw_shifts.
      - CH3 groups: if >1 shift value is specified AND they differ (> CH3_DIFF_TOL),
        DROP all protons of that CH3 group; otherwise assign the single shift to all 3 Hs.
      - CH2 groups: if 1 unique shift, assign it to BOTH Hs; if 2 distinct shifts,
        SORT ascending (s_low <= s_high) and assign to (h_first, h_second) by RDKit atom_idx;
        if >2 distinct shifts, drop the corrupted CH2 group.
      - Single-H atoms (CH, NH, OH, SH): assign if consistent, drop if conflicting.
      - Multi-H heteroatoms (NH2, NH3+, etc.): broadcast if 1 unique value, sort if 2 on NH2.
    """
    lo, hi = SHIFT_BOUNDS['1H']
    final_1h = {}

    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        h_idxs = sorted(nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)
        if not h_idxs:
            continue

        heavy_idx = atom.GetIdx()
        collected = []

        # Collect signals pointing directly to explicit H indices
        for h_idx in h_idxs:
            if h_idx in raw_shifts:
                for v in raw_shifts[h_idx]:
                    fv = float(v)
                    if lo <= fv <= hi:
                        collected.append(fv)
                    else:
                        stats['1h_oob_peaks_dropped'] += 1

        # Collect signals pointing to the parent heavy atom
        if heavy_idx in raw_shifts:
            for v in raw_shifts[heavy_idx]:
                fv = float(v)
                if lo <= fv <= hi:
                    collected.append(fv)
                else:
                    stats['1h_oob_peaks_dropped'] += 1

        if not collected:
            continue

        is_carbon = (atom.GetAtomicNum() == 6)
        n_h = len(h_idxs)
        spread = max(collected) - min(collected)

        if is_carbon and n_h == 3:
            # User rule: drop CH3 protons if >1 proton/signal is listed and they differ
            if len(collected) > 1 and spread > CH3_DIFF_TOL:
                stats['ch3_groups_dropped_conflicting'] += 1
                stats['ch3_protons_dropped_conflicting'] += 3
                continue
            val = float(np.mean(collected))
            for h_idx in h_idxs:
                final_1h[h_idx] = val
            stats['ch3_groups_kept'] += 1

        elif is_carbon and n_h == 2:
            if spread <= CH3_DIFF_TOL:
                val = float(np.mean(collected))
                final_1h[h_idxs[0]] = val
                final_1h[h_idxs[1]] = val
                if len(collected) == 1:
                    stats['ch2_groups_filled_partner'] += 1
                else:
                    stats['ch2_groups_equal'] += 1
            elif len(collected) == 2:
                s_low, s_high = sorted(collected)
                final_1h[h_idxs[0]] = s_low
                final_1h[h_idxs[1]] = s_high
                stats['ch2_groups_split_sorted'] += 1
            else:
                stats['ch2_groups_dropped_conflicting'] += 1
                continue

        elif n_h == 1:
            if spread <= CH3_DIFF_TOL:
                final_1h[h_idxs[0]] = float(np.mean(collected))
            else:
                stats['single_h_dropped_conflicting'] += 1
                continue

        else:
            # Multi-H heteroatom (e.g., NH2, NH3+, PH2)
            if spread <= CH3_DIFF_TOL:
                val = float(np.mean(collected))
                for h_idx in h_idxs:
                    final_1h[h_idx] = val
            elif n_h == 2 and len(collected) == 2:
                s_low, s_high = sorted(collected)
                final_1h[h_idxs[0]] = s_low
                final_1h[h_idxs[1]] = s_high
            else:
                stats['hetero_multi_h_dropped_conflicting'] += 1
                continue

    return final_1h


def shift_signature(canon_smi, shift_dict):
    """Exact-instance deduplication key: (canonical SMILES, sorted rounded shift values)."""
    vals = tuple(sorted(round(float(v), 3) for v in shift_dict.values()))
    return (canon_smi, vals)


def assign_split_by_smiles(canon_smi, train_split=0.8, seed=42):
    """
    Deterministic group split by canonical SMILES.
    Guarantees that ALL instances of a canonical SMILES (across both 13C and 1H,
    and across both SDF and original pickle sources) go strictly to either train or test.
    """
    h = hashlib.md5(f"{seed}:{canon_smi}".encode('utf-8')).digest()
    frac = int.from_bytes(h[:8], 'big') / float(1 << 64)
    return 'train' if frac < train_split else 'test'


def process_sdf(sdf_file, max_n, all_spectra, seed, stats,
                records_13c, records_1h, seen_sig_13c, seen_sig_1h):
    print(_("Streaming {path}...", path=sdf_file))
    supplier = Chem.ForwardSDMolSupplier(sdf_file, removeHs=False)

    for mol in tqdm(supplier, desc=_("Processing SDF")):
        stats['sdf_total_records'] += 1
        if mol is None:
            stats['sdf_rdkit_none'] += 1
            continue

        try:
            props = mol.GetPropsAsDict()
            c_keys = sorted(k for k in props if k.startswith("Spectrum 13C"))
            h_keys = sorted(k for k in props if k.startswith("Spectrum 1H"))
            if not c_keys and not h_keys:
                continue

            if not all_spectra:
                c_keys = c_keys[:1]
                h_keys = h_keys[:1]

            # Quick pre-filter before 3D check to save time
            if len(Chem.GetMolFrags(mol)) != 1:
                stats['filter_multi_frag'] += 1
                continue
            if any(a.GetAtomicNum() not in ALLOWED_ATOMIC_NUMS for a in mol.GetAtoms()):
                stats['filter_disallowed_elem'] += 1
                continue
            if any(a.GetNumRadicalElectrons() != 0 for a in mol.GetAtoms()):
                stats['filter_radical'] += 1
                continue
            tot_est = mol.GetNumAtoms() + sum(a.GetTotalNumHs(includeNeighbors=False) for a in mol.GetAtoms())
            if tot_est > max_n:
                stats['filter_too_large'] += 1
                continue

            mol_h = ensure_valid_3d(mol, stats, seed=seed)
            if mol_h is None:
                continue
            if not passes_molecule_filters(mol_h, max_n, stats):
                continue

            canon_smi = get_canonical_smiles(mol_h)

            for ck in c_keys:
                raw_13c = parse_spectrum_string(props[ck])
                final_13c = clean_13c_shifts(mol_h, raw_13c, stats)
                if not final_13c:
                    continue
                sig = shift_signature(canon_smi, final_13c)
                if sig in seen_sig_13c:
                    stats['13c_exact_dups_skipped'] += 1
                    continue
                seen_sig_13c.add(sig)
                records_13c.append({
                    'rdmol': mol_h,
                    'value': [final_13c],
                    'canon_smi': canon_smi,
                    'source': 'sdf',
                })
                stats['13c_from_sdf'] += 1

            for hk in h_keys:
                raw_1h = parse_spectrum_string(props[hk])
                final_1h = clean_1h_shifts(mol_h, raw_1h, stats)
                if not final_1h:
                    continue
                sig = shift_signature(canon_smi, final_1h)
                if sig in seen_sig_1h:
                    stats['1h_exact_dups_skipped'] += 1
                    continue
                seen_sig_1h.add(sig)
                records_1h.append({
                    'rdmol': mol_h,
                    'value': [final_1h],
                    'canon_smi': canon_smi,
                    'source': 'sdf',
                })
                stats['1h_from_sdf'] += 1

        except Exception:
            stats['sdf_exceptions'] += 1
            continue


def merge_existing_pickle(pickle_path, nucleus, max_n, seed, stats, records, seen_sigs):
    if not pickle_path or not os.path.exists(pickle_path):
        print(f"WARNING: Existing {nucleus} pickle not found (skipping): {pickle_path}")
        return

    print(f"Merging existing {nucleus} pickle: {pickle_path}")
    d = load_pickle_compat(pickle_path)
    df_all = pd.concat([d['train_df'], d['test_df']], ignore_index=True)

    for _, row in tqdm(df_all.iterrows(), total=len(df_all), desc=f"Merging old {nucleus}"):
        mol = row['rdmol']
        if mol is None:
            continue
        try:
            mol_h = ensure_valid_3d(mol, stats, seed=seed)
            if mol_h is None:
                continue
            if not passes_molecule_filters(mol_h, max_n, stats):
                continue

            raw_dict = row['value'][0]
            raw_shifts = {int(k): [float(v)] for k, v in raw_dict.items()}
            if nucleus == '13C':
                final_shifts = clean_13c_shifts(mol_h, raw_shifts, stats)
            else:
                final_shifts = clean_1h_shifts(mol_h, raw_shifts, stats)

            if not final_shifts:
                continue

            canon_smi = get_canonical_smiles(mol_h)
            sig = shift_signature(canon_smi, final_shifts)
            if sig in seen_sigs:
                stats[f'{nucleus.lower()}_exact_dups_skipped'] += 1
                continue
            seen_sigs.add(sig)
            records.append({
                'rdmol': mol_h,
                'value': [final_shifts],
                'canon_smi': canon_smi,
                'source': 'old_pickle',
            })
            stats[f'{nucleus.lower()}_from_old_pickle'] += 1
        except Exception:
            stats[f'old_{nucleus.lower()}_exceptions'] += 1
            continue


def split_and_save(records, nucleus, max_n, cv_i, output_dir, train_split, seed, stats_only=False):
    if not records:
        print(_("No data found for nucleus {nucleus}!", nucleus=nucleus))
        return set(), set(), set()

    train_rows = []
    test_rows = []
    train_smis = set()
    test_smis = set()

    for rec in records:
        smi = rec['canon_smi']
        row = {'rdmol': rec['rdmol'], 'value': rec['value']}
        if assign_split_by_smiles(smi, train_split=train_split, seed=seed) == 'train':
            train_rows.append(row)
            train_smis.add(smi)
        else:
            test_rows.append(row)
            test_smis.add(smi)

    # Strict zero-leakage assertion
    leak = train_smis & test_smis
    assert len(leak) == 0, f"FATAL: {len(leak)} canonical SMILES leaked between train and test for {nucleus}!"

    # Reproducible row shuffle within train and within test independently
    rng = np.random.RandomState(seed)
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    print(
        f"[{nucleus}] Instances: Train={len(train_rows)}, Test={len(test_rows)}, Total={len(records)} | "
        f"Unique canonical SMILES: Train={len(train_smis)}, Test={len(test_smis)}, "
        f"Total={len(train_smis | test_smis)} (Train/Test overlap=0)"
    )

    if not stats_only:
        output_dict = {
            'train_df': pd.DataFrame(train_rows),
            'test_df': pd.DataFrame(test_rows),
            'tgt_nucs': [nucleus],
            'MAX_N': max_n,
        }
        os.makedirs(output_dir, exist_ok=True)
        filename = FILENAME_TEMPLATE.format(nuc=nucleus, max_n=max_n, cv=cv_i)
        filepath = os.path.join(output_dir, filename)
        print(_("Saving {nucleus}... Train: {n_train}, Test: {n_test}",
                nucleus=nucleus, n_train=len(train_rows), n_test=len(test_rows)))
        with open(filepath, 'wb') as f:
            pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(_("-> File created: {path}", path=filepath))

    return train_smis | test_smis, train_smis, test_smis


def print_final_summary(stats, smis_13c, tr_13c, te_13c, smis_1h, tr_1h, te_1h, n_inst_13c, n_inst_1h):
    all_unique = smis_13c | smis_1h
    both_unique = smis_13c & smis_1h
    only_13c = smis_13c - smis_1h
    only_1h = smis_1h - smis_13c
    both_train = tr_13c & tr_1h
    both_test = te_13c & te_1h
    cross_leak = (tr_13c & te_1h) | (te_13c & tr_1h)

    print("\n" + "=" * 72)
    print("FINAL COMBINED DATABASE STATISTICS")
    print("=" * 72)
    print(f"SDF records scanned:              {stats['sdf_total_records']} (RDKit None: {stats['sdf_rdkit_none']})")
    print(f"3D geometry:                      valid as-is={stats['geom_valid_as_is']}, "
          f"repaired ETKDGv3={stats['geom_repaired_etkdg']}, dropped_bad_3d={stats['geom_dropped_bad_3d']}")
    print(f"Molecule filters dropped:         >MAX_N={stats['filter_too_large']}, "
          f"multi_frag={stats['filter_multi_frag']}, elem={stats['filter_disallowed_elem']}, "
          f"radical={stats['filter_radical']}")
    print("-" * 72)
    print(f"13C instances:                    total={n_inst_13c} "
          f"(from SDF={stats['13c_from_sdf']}, added from old pickle={stats['13c_from_old_pickle']}, "
          f"exact dups skipped={stats['13c_exact_dups_skipped']})")
    print(f"1H  instances:                    total={n_inst_1h} "
          f"(from SDF={stats['1h_from_sdf']}, added from old pickle={stats['1h_from_old_pickle']}, "
          f"exact dups skipped={stats['1h_exact_dups_skipped']})")
    print("-" * 72)
    print("1H CH3 / CH2 HYGIENE:")
    print(f"  CH3 groups kept (consistent):   {stats['ch3_groups_kept']}")
    print(f"  CH3 groups DROPPED (differing): {stats['ch3_groups_dropped_conflicting']} "
          f"({stats['ch3_protons_dropped_conflicting']} protons removed)")
    print(f"  CH2 groups (1 shift -> filled): {stats['ch2_groups_filled_partner']}")
    print(f"  CH2 groups (2 equal shifts):    {stats['ch2_groups_equal']}")
    print(f"  CH2 groups (2 diff -> sorted):  {stats['ch2_groups_split_sorted']}")
    print(f"  CH2 groups dropped (>2 diff):   {stats['ch2_groups_dropped_conflicting']}")
    print(f"  Single-H / hetero conflicting:  {stats['single_h_dropped_conflicting']} / {stats['hetero_multi_h_dropped_conflicting']}")
    print(f"  Out-of-range peaks dropped:     13C={stats['13c_oob_peaks_dropped']}, 1H={stats['1h_oob_peaks_dropped']}")
    print("-" * 72)
    print("UNIQUE MOLECULES (BY CANONICAL SMILES):")
    print(f"  Total unique molecules (13C U 1H):     {len(all_unique)}")
    print(f"  Unique molecules with 13C:             {len(smis_13c)} (train={len(tr_13c)}, test={len(te_13c)})")
    print(f"  Unique molecules with 1H:              {len(smis_1h)} (train={len(tr_1h)}, test={len(te_1h)})")
    print(f"  Unique molecules with 13C ONLY:        {len(only_13c)}")
    print(f"  Unique molecules with 1H  ONLY:        {len(only_1h)}")
    print(f"  Unique molecules with BOTH 1H AND 13C: {len(both_unique)} (train={len(both_train)}, test={len(both_test)})")
    print(f"  Train/Test SMILES leakage (13C/1H/cross): "
          f"{len(tr_13c & te_13c)} / {len(tr_1h & te_1h)} / {len(cross_leak)}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Build clean 3D NMRShiftDB2 train/test .pickle datasets (merging SDF + original project pickles)"
    )
    parser.add_argument("--sdf", type=str, default=DEFAULT_SDF_FILE,
                        help=f"Input NMRShiftDB2 SDF file (default: {DEFAULT_SDF_FILE})")
    parser.add_argument("--old_13c", type=str, default=DEFAULT_OLD_13C,
                        help="Original project 13C .pickle to merge")
    parser.add_argument("--old_1h", type=str, default=DEFAULT_OLD_1H,
                        help="Original project 1H .pickle to merge")
    parser.add_argument("--no_merge_old", action="store_true",
                        help="Do not merge existing project .pickle datasets")
    parser.add_argument("--first_spectrum_only", action="store_true",
                        help="Use only the first Spectrum 13C / 1H per SDF record (default: use all distinct spectra)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--cv_i", type=int, default=DEFAULT_CV_I,
                        help=f"Output dataset cv_i index (default: {DEFAULT_CV_I})")
    parser.add_argument("--max_n", type=int, default=DEFAULT_MAX_N,
                        help=f"Maximum atoms including explicit Hs (default: {DEFAULT_MAX_N})")
    parser.add_argument("--train_split", type=float, default=DEFAULT_TRAIN_SPLIT,
                        help=f"Fraction of unique canonical SMILES in train split (default: {DEFAULT_TRAIN_SPLIT})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for deterministic SMILES split & ETKDG repair (default: {DEFAULT_SEED})")
    parser.add_argument("--stats_only", action="store_true",
                        help="Run full processing and print statistics without writing .pickle files")
    parser.add_argument("--allow_pandas2", action="store_true",
                        help="Allow writing .pickle under pandas>=2 / numpy>=2 (not recommended for pandas 1.x servers)")
    args = parser.parse_args()

    if not args.stats_only:
        check_pickle_write_env(allow_pandas2=args.allow_pandas2)

    stats = Counter()
    records_13c = []
    records_1h = []
    seen_sig_13c = set()
    seen_sig_1h = set()

    if args.sdf and os.path.exists(args.sdf):
        process_sdf(
            args.sdf, args.max_n, not args.first_spectrum_only, args.seed, stats,
            records_13c, records_1h, seen_sig_13c, seen_sig_1h,
        )
    else:
        raise FileNotFoundError(f"SDF file not found: {args.sdf}")

    if not args.no_merge_old:
        merge_existing_pickle(args.old_13c, '13C', args.max_n, args.seed, stats, records_13c, seen_sig_13c)
        merge_existing_pickle(args.old_1h, '1H', args.max_n, args.seed, stats, records_1h, seen_sig_1h)

    print(_("\n--- BUILD SUMMARY ---"))
    smis_13c, tr_13c, te_13c = split_and_save(
        records_13c, '13C', args.max_n, args.cv_i, args.output_dir,
        args.train_split, args.seed, stats_only=args.stats_only,
    )
    smis_1h, tr_1h, te_1h = split_and_save(
        records_1h, '1H', args.max_n, args.cv_i, args.output_dir,
        args.train_split, args.seed, stats_only=args.stats_only,
    )

    print_final_summary(
        stats, smis_13c, tr_13c, te_13c, smis_1h, tr_1h, te_1h,
        len(records_13c), len(records_1h),
    )


if __name__ == "__main__":
    main()
