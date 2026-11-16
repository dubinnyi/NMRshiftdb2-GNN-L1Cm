import os
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from i18n import _

# === CONFIGURATION ===
SDF_FILE = "nmrshiftdb2/nmrshiftdb2withsignals.sd"
OUTPUT_DIR = "graph_conv_many_nuc_pipeline.datasets"
MAX_N = 64  # NOTE: Maximum atom count INCLUDING explicitly added hydrogens
TRAIN_SPLIT = 0.8
CV_I = 3

# Filename template expected by respredict_pipeline.py
FILENAME_TEMPLATE = "graph_conv_many_nuc_pipeline.data.{nuc}.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.{max_n}.{cv}.mol_dict.pickle"


def parse_spectrum_string(spec_str):
    """Parse an NMRShiftDB spectrum string '17.6;0.0Q;10|...' into a dict {atom_idx: [shifts]}."""
    shifts_dict = {}
    for peak in spec_str.split('|'):
        if not peak.strip():
            continue
        parts = peak.split(';')
        if len(parts) >= 3:
            try:
                shift = float(parts[0])
                atom_ids = parts[2].split(',')
                for aid in atom_ids:
                    if aid.strip().isdigit():
                        idx = int(aid.strip())
                        if idx not in shifts_dict:
                            shifts_dict[idx] = []
                        shifts_dict[idx].append(shift)
            except ValueError:
                continue
    return shifts_dict


def process_molecule(mol):
    """Add explicit H atoms and split spectral annotations into 13C and 1H records."""
    props = mol.GetPropsAsDict()

    spec_strings = {'13C': None, '1H': None}
    for prop_name, prop_val in props.items():
        if prop_name.startswith("Spectrum 13C") and spec_strings['13C'] is None:
            spec_strings['13C'] = prop_val
        elif prop_name.startswith("Spectrum 1H") and spec_strings['1H'] is None:
            spec_strings['1H'] = prop_val

    raw_shifts = {
        nuc: parse_spectrum_string(spec_strings[nuc]) if spec_strings[nuc] else {}
        for nuc in ['13C', '1H']
    }

    # Add explicit hydrogens to the molecular graph
    mol_with_hs = Chem.AddHs(mol)

    if mol_with_hs.GetNumAtoms() > MAX_N:
        return None, None

    final_13c = {}
    final_1h = {}

    for atom in mol_with_hs.GetAtoms():
        idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()

        if atomic_num == 6:  # Carbon
            if idx in raw_shifts['13C']:
                final_13c[idx] = np.mean(raw_shifts['13C'][idx])

        elif atomic_num == 1:  # Hydrogen
            neighbors = atom.GetNeighbors()
            if not neighbors:
                continue
            heavy_idx = neighbors[0].GetIdx()

            # Transfer 1H shift annotation from the parent heavy atom to the attached hydrogen
            if heavy_idx in raw_shifts['1H'] and len(raw_shifts['1H'][heavy_idx]) > 0:
                assigned_shift = raw_shifts['1H'][heavy_idx].pop(0)
                final_1h[idx] = assigned_shift

    # Structure of 'value': list of dicts (a 1-element list for single-nucleus models)
    res_13c = {'rdmol': mol_with_hs, 'value': [final_13c]} if final_13c else None
    res_1h  = {'rdmol': mol_with_hs, 'value': [final_1h]} if final_1h else None

    return res_13c, res_1h


def make_split_and_save(data, nucleus, max_n):
    if not data:
        print(_("No data found for nucleus {nucleus}!", nucleus=nucleus))
        return

    # Reproducible shuffle
    np.random.seed(42)
    np.random.shuffle(data)

    split_idx = int(len(data) * TRAIN_SPLIT)

    output_dict = {
        'train_df': pd.DataFrame(data[:split_idx]),
        'test_df': pd.DataFrame(data[split_idx:]),
        'tgt_nucs': [nucleus],  # Strictly one target nucleus per dataset file
        'MAX_N': max_n
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = FILENAME_TEMPLATE.format(nuc=nucleus, max_n=max_n, cv=CV_I)
    filepath = os.path.join(OUTPUT_DIR, filename)

    print(_("Saving {nucleus}... Train: {n_train}, Test: {n_test}",
            nucleus=nucleus, n_train=split_idx, n_test=len(data) - split_idx))
    with open(filepath, 'wb') as f:
        pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(_("-> File created: {path}", path=filepath))


def main():
    print(_("Streaming {path}...", path=SDF_FILE))
    supplier = Chem.ForwardSDMolSupplier(SDF_FILE, removeHs=False)

    data_13c = []
    data_1h = []

    for mol in tqdm(supplier, desc=_("Processing SDF")):
        if mol is None:
            continue
        try:
            r_13c, r_1h = process_molecule(mol)
            if r_13c:
                data_13c.append(r_13c)
            if r_1h:
                data_1h.append(r_1h)
        except Exception:
            continue

    print(_("\n--- BUILD SUMMARY ---"))
    make_split_and_save(data_13c, '13C', MAX_N)
    make_split_and_save(data_1h, '1H', MAX_N)


if __name__ == "__main__":
    main()
