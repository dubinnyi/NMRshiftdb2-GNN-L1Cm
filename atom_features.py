"""
Per-atom features : Featurizations that return one per atom
[in contrast to whole-molecule featurizations]
"""

import pandas as pd
import numpy as np
import sklearn.metrics
import torch
from numba import jit
import scipy.spatial
from rdkit import Chem
from util import get_nos_coords



def atom_adj_mat(mol, conformer_i, **kwargs):
    """
    OUTPUT IS ATOM_N x (adj_mat, tgt_atom, atomic_nos, dists )
    
    This is really inefficient given that we explicitly return the same adj
    matrix for each atom, and index into it
    
    Adj mat is valence number * 2
    
    
    """
    
    MAX_ATOM_N = kwargs.get('MAX_ATOM_N', 64)
    atomic_nos, coords = get_nos_coords(mol, conformer_i)
    ATOM_N = len(atomic_nos)

    atomic_nos_pad, adj = mol_to_nums_adj(mol, MAX_ATOM_N)
    

    features = np.zeros((ATOM_N,), 
                    dtype=[('adj', np.uint8, (MAX_ATOM_N, MAX_ATOM_N)), 
                           ('my_idx', np.int), 
                           ('atomicno', np.uint8, MAX_ATOM_N), 
                           ('pos', np.float32, (MAX_ATOM_N, 3,))])

    
    
    for atom_i in range(ATOM_N):
        vects = coords - coords[atom_i]
        features[atom_i]['adj'] = adj*2
        features[atom_i]['my_idx'] =  atom_i
        features[atom_i]['atomicno'] = atomic_nos_pad
        features[atom_i]['pos'][:ATOM_N] = vects
    return features

def advanced_atom_props(mol, conformer_i, **kwargs):
    import rdkit.Chem.rdPartialCharges
    pt = Chem.GetPeriodicTable()
    atomic_nos, coords = get_nos_coords(mol, conformer_i)
    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ALL, 
                     catchErrors=True)
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    ATOM_N = len(atomic_nos)
    out = np.zeros(ATOM_N, 
                   dtype=[('total_valence', np.int),  
                          ('aromatic', np.bool), 
                          ('hybridization', np.int), 
                          ('partial_charge', np.float32),
                          ('formal_charge', np.float32), 
                          ('atomicno', np.int), 
                          ('r_covalent', np.float32),
                          ('r_vanderwals', np.float32),
                          ('default_valence', np.int),
                          ('rings', np.bool, 5), 
                          ('pos', np.float32, 3)])
    
      
    for i in range(mol.GetNumAtoms()):
        a = mol.GetAtomWithIdx(i)
        atomic_num = int(atomic_nos[i])
        out[i]['total_valence'] = a.GetTotalValence()
        out[i]['aromatic'] = a.GetIsAromatic()
        out[i]['hybridization'] = a.GetHybridization()
        out[i]['partial_charge'] = a.GetProp('_GasteigerCharge')
        out[i]['formal_charge'] = a.GetFormalCharge()
        out[i]['atomicno'] = atomic_nos[i]
        out[i]['r_covalent'] =pt.GetRcovalent(atomic_num)
        out[i]['r_vanderwals'] =  pt.GetRvdw(atomic_num)
        out[i]['default_valence'] = pt.GetDefaultValence(atomic_num)
        out[i]['rings'] = [a.IsInRingSize(r) for r in range(3, 8)]
        out[i]['pos'] = coords[i]
                          
    return out

HYBRIDIZATIONS = [Chem.HybridizationType.S, 
                  Chem.HybridizationType.SP, 
                  Chem.HybridizationType.SP2, 
                  Chem.HybridizationType.SP3, 
                  Chem.HybridizationType.SP3D, 
                  Chem.HybridizationType.SP3D2]

CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]

# Exact 51 MMFF94 atom types from Williams & Jonas (Chem. Sci. 2023, ESI Table 1)
MMFF94_ATOM_TYPES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    37, 38, 39, 40, 42, 43, 44, 46, 48, 59, 62, 63, 64, 65, 66, 70, 71, 72, 74, 75, 78,
]

# Pauling electronegativities for organic NMR elements
ELECTRONEGATIVITIES = {
    1: 2.20,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    15: 2.19,
    16: 2.58,
    17: 3.16,
}

def to_onehot(x, vals):
    return [x == v for v in vals]


def _compute_3d_atom_env(mol, atomic_nos, coords, element_list):
    """
    Compute SE(3)-invariant (rotation/translation invariant) through-space 3D
    atomic environment features (11 dims for 8 elements):
      - sum_{j != i, Z_j == e} min(1 / r_ij^3, 1.0) for each element e in element_list
        (magnetic dipole / anisotropic shielding scales as 1/r^3)
      - sum_{j != i, aromatic(j)} min(1 / r_ij^3, 1.0) (ring-current proxy)
      - sum_{j != i, heteroatom(j)} min(1 / r_ij^3, 1.0) (through-space inductive/H-bond proxy)
      - inverse distance to nearest non-bonded heavy atom (steric crowding proxy)
    Returns zeros gracefully if no valid 3D coordinates exist.
    """
    atom_n = len(atomic_nos)
    n_elem = len(element_list)
    out = np.zeros((atom_n, n_elem + 3), dtype=np.float32)

    if coords is None or len(coords) != atom_n:
        return out
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    if np.max(dist) < 1e-3:
        return out

    # Compute clamped 1/r^3 kernel with zero diagonal
    safe_dist = np.where(dist > 0.5, dist, 0.5)
    inv_r3 = np.minimum(1.0 / (safe_dist ** 3), 1.0)
    np.fill_diagonal(inv_r3, 0.0)

    # Mask out zero-distance artifacts (e.g. unembedded hydrogens)
    valid_pair = dist > 0.1
    inv_r3 = inv_r3 * valid_pair

    for e_i, el in enumerate(element_list):
        el_mask = (atomic_nos == el).astype(np.float32)
        out[:, e_i] = inv_r3 @ el_mask

    arom_mask = np.array([1.0 if mol.GetAtomWithIdx(i).GetIsAromatic() else 0.0 for i in range(atom_n)], dtype=np.float32)
    hetero_mask = np.array([1.0 if int(atomic_nos[i]) not in (1, 6) else 0.0 for i in range(atom_n)], dtype=np.float32)
    out[:, n_elem] = inv_r3 @ arom_mask
    out[:, n_elem + 1] = inv_r3 @ hetero_mask

    # Nearest non-bonded heavy atom inverse distance (steric crowding / through-space contact)
    bonded = np.eye(atom_n, dtype=bool)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonded[i, j] = True
        bonded[j, i] = True
    heavy = (atomic_nos > 1)[None, :]
    nb_heavy_mask = (~bonded) & heavy & valid_pair
    for i in range(atom_n):
        d_nb = dist[i, nb_heavy_mask[i]]
        if len(d_nb) > 0:
            out[i, n_elem + 2] = float(min(1.0 / max(float(np.min(d_nb)), 0.5), 2.0))

    return out


def _compute_ch2_features(mol):
    """
    Compute 5 per-atom features specifically designed to break 2D permutation symmetry
    for CH2 protons and indicate diastereotopic context:
      0: is_ch2_h_first  - 1.0 for the first H (smaller RDKit atom_idx) of a CH2 group
      1: is_ch2_h_second - 1.0 for the second H (larger RDKit atom_idx) of a CH2 group
      2: is_ch2_in_ring  - 1.0 for either H of a cyclic CH2 group
      3: ch2_near_chiral - 1.0 for either H of a CH2 group in a molecule with >=1 chiral center
      4: ch2_heavy_asym  - 1.0 if the two non-H neighbors of the CH2 carbon are topologically distinct
    All 5 features are 0.0 for non-CH2 atoms (preserving CH3/CH/C symmetry).
    """
    atom_n = mol.GetNumAtoms()
    out = np.zeros((atom_n, 5), dtype=np.float32)

    has_chiral_center = any(
        a.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED
        for a in mol.GetAtoms()
    )
    try:
        canon_ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    except Exception:
        canon_ranks = list(range(atom_n))

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        neighbors = list(atom.GetNeighbors())
        h_idxs = sorted(nb.GetIdx() for nb in neighbors if nb.GetAtomicNum() == 1)
        if len(h_idxs) != 2:
            continue

        heavy_idxs = [nb.GetIdx() for nb in neighbors if nb.GetAtomicNum() != 1]
        in_ring = 1.0 if atom.IsInRing() else 0.0
        chiral_flag = 1.0 if has_chiral_center else 0.0
        if len(heavy_idxs) == 2:
            asym_flag = 1.0 if canon_ranks[heavy_idxs[0]] != canon_ranks[heavy_idxs[1]] else 0.0
        else:
            asym_flag = 0.0

        h1, h2 = h_idxs[0], h_idxs[1]
        out[h1] = [1.0, 0.0, in_ring, chiral_flag, asym_flag]
        out[h2] = [0.0, 1.0, in_ring, chiral_flag, asym_flag]

    return out


def feat_tensor_atom(mol, 
                     feat_atomicno = True, feat_pos=True, 
                     feat_atomicno_onehot=[1, 6, 7, 8, 9], 
                     feat_valence=True, aromatic=True, hybridization=True, 
                     
                     partial_charge=True, formal_charge=True, r_covalent=True,
                     r_vanderwals=True, default_valence=True, rings=False, 
                     total_valence_onehot=False,
                     max_ring_size=8,
                     mmff_atom_types_onehot=False,
                     electronegativity=False,
                     chirality=False,
                     assign_stereo=False,
                     extra_topo=False,
                     geom_3d_atom=False,
                     ch2_order_feat=False,
                     conf_idx = 0):

    """
    Featurize a molecule on a per-atom basis
    feat_atomicno_onehot : list of atomic numbers

    Always assume using conf_idx unless otherwise passed

    Returns an (ATOM_N x feature) float32 tensor

    NOTE: Performs NO santization or cleanup of molecule unless needed for
    optional advanced features (Gasteiger/MMFF/3D stereo).

    """

    pt = Chem.GetPeriodicTable()
    mol = Chem.Mol(mol) # copy molecule

    atomic_nos, coords = get_nos_coords(mol, conf_idx)
    ATOM_N = len(atomic_nos)    

    if assign_stereo and mol.GetNumConformers() > 0:
        try:
            Chem.rdmolops.AssignStereochemistryFrom3D(mol, confId=conf_idx)
        except Exception:
            pass

    if partial_charge:
        try:
            Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception:
            Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
            Chem.rdPartialCharges.ComputeGasteigerCharges(mol)

    mmff_p = None
    if mmff_atom_types_onehot:
        try:
            mmff_p = Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
        except Exception:
            mmff_p = None

    geom_3d_arr = None
    if geom_3d_atom:
        elem_list = feat_atomicno_onehot if feat_atomicno_onehot is not None else [1, 6, 7, 8, 9, 15, 16, 17]
        geom_3d_arr = _compute_3d_atom_env(mol, atomic_nos, coords, elem_list)

    ch2_arr = _compute_ch2_features(mol) if ch2_order_feat else None

    atom_features = []
      
    for i in range(mol.GetNumAtoms()):
        a = mol.GetAtomWithIdx(i)
        atomic_num = int(atomic_nos[i])
        atom_feature = []

        if feat_atomicno:
            atom_feature += [atomic_num]

        if feat_pos:
            atom_feature += coords[i].tolist()

        if feat_atomicno_onehot is not None : 
            atom_feature += to_onehot(atomic_num, feat_atomicno_onehot)
            
        if feat_valence:
            atom_feature += [a.GetTotalValence()]
        if total_valence_onehot:
            atom_feature +=  to_onehot(a.GetTotalValence(), range(1, 7))

        if aromatic:
            atom_feature += [a.GetIsAromatic()]

        if hybridization:
            atom_feature += to_onehot(a.GetHybridization(), HYBRIDIZATIONS)

        if partial_charge:
            try:
                gc = float(a.GetProp('_GasteigerCharge'))
            except Exception:
                gc = 0.0
            if not np.isfinite(gc):
                gc = 0.0
            gc = float(np.clip(gc, -2.0, 2.0))
            atom_feature += [gc]

        if formal_charge:
            atom_feature += to_onehot(a.GetFormalCharge(), [-1, 0, 1])

        if r_covalent:
            atom_feature += [pt.GetRcovalent(atomic_num)]
        if r_vanderwals:
            atom_feature += [pt.GetRvdw(atomic_num)]
            

        if default_valence:
            atom_feature += to_onehot(pt.GetDefaultValence(atomic_num), range(1, 7))

        if rings:
            atom_feature += [a.IsInRingSize(r) for r in range(3, max_ring_size)]

        if electronegativity:
            atom_feature += [ELECTRONEGATIVITIES.get(atomic_num, 2.5)]

        if chirality:
            ct = a.GetChiralTag()
            atom_feature += to_onehot(ct, CHIRAL_TAGS)
            atom_feature += [1.0 if ct != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0.0]

        if extra_topo:
            is_conj = any(b.GetIsConjugated() for b in a.GetBonds())
            n_h_nb = sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() == 1)
            atom_feature += [1.0 if a.IsInRing() else 0.0, 1.0 if is_conj else 0.0]
            atom_feature += to_onehot(min(n_h_nb, 3), [0, 1, 2, 3])

        if mmff_atom_types_onehot:
            if mmff_p is None:
                atom_feature += [0] * len(MMFF94_ATOM_TYPES)
            else:
                atom_feature += to_onehot(mmff_p.GetMMFFAtomType(i), MMFF94_ATOM_TYPES)

        if geom_3d_arr is not None:
            atom_feature += geom_3d_arr[i].tolist()

        if ch2_arr is not None:
            atom_feature += ch2_arr[i].tolist()

        atom_features.append(atom_feature)

    return torch.Tensor(atom_features)

