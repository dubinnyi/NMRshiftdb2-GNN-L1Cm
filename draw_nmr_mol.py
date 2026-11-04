import argparse
import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

def draw_balanced_mol(pickle_path, split_type, index, output_file, mode='shifts', size=2000):
    with open(pickle_path, 'rb') as f:
        d = pickle.load(f)
    
    df = d[f'{split_type}_df']
    mol = df.iloc[index]['rdmol']
    shifts_dict = df.iloc[index]['value'][0]
    
    mol_copy = Chem.Mol(mol)
    mol_copy.RemoveAllConformers()
    AllChem.Compute2DCoords(mol_copy)
    
    for atom in mol_copy.GetAtoms():
        idx = atom.GetIdx()
        if idx in shifts_dict:
            shift_val = shifts_dict[idx]
            if mode == 'shifts':
                label = f"{shift_val:.2f}"
            elif mode == 'indices':
                label = str(idx)
            else:
                 label = f"{idx}:{shift_val:.2f}"
            atom.SetProp('atomNote', label)

    d2d = Draw.MolDraw2DCairo(size, size)
    opts = d2d.drawOptions()
    
    # Сбалансированные настройки шрифта
    opts.annotationFontScale = 1.3 
    opts.minFontSize = 24
    opts.maxFontSize = 36
    opts.clearBackground = True
    opts.drawMolsSameScale = False
    
    d2d.DrawMolecule(mol_copy)
    d2d.FinishDrawing()
    d2d.WriteDrawingText(output_file)
    print(f"Сбалансированное изображение {size}x{size} успешно сохранено в: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw beautiful molecule with NMR shifts.")
    parser.add_argument("pickle_path", type=str)
    parser.add_argument("split", type=str, choices=["train", "test"])
    parser.add_argument("index", type=int)
    parser.add_argument("output", type=str)
    parser.add_argument("--mode", type=str, choices=["shifts", "indices", "both"], default="shifts")
    parser.add_argument("--size", type=int, default=2000)
    
    args = parser.parse_args()
    draw_balanced_mol(args.pickle_path, args.split, args.index, args.output, args.mode, args.size)
