import argparse
import pickle
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

def propagate_shifts_in_dict(mol, shifts_dict):
    """
    Распространяет химические сдвиги на эквивалентные атомы водорода
    с использованием канонических рангов симметрии.
    """
    new_shifts = shifts_dict.copy()
    
    # 1. Получаем классы симметрии атомов
    # breakTies=False гарантирует, что симметричные атомы (включая протоны одной CH3/CH2 группы
    # или симметричные половины молекулы) получат одинаковый ранг.
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    
    # 2. Группируем атомы по их рангам симметрии
    rank_to_atoms = defaultdict(list)
    for idx, rank in enumerate(ranks):
        rank_to_atoms[rank].append(idx)
        
    # 3. Распространяем сдвиги внутри каждой группы симметрии
    for rank, atom_indices in rank_to_atoms.items():
        # Ищем, есть ли в этой группе симметрии атом с уже известным сдвигом
        known_shift = None
        for idx in atom_indices:
            if idx in new_shifts:
                known_shift = new_shifts[idx]
                break
                
        # Если сдвиг найден, копируем его на все топологически эквивалентные атомы
        if known_shift is not None:
            for idx in atom_indices:
                if idx not in new_shifts:
                    new_shifts[idx] = known_shift
                    
    return new_shifts

def fix_dataset_symmetry(input_pickle, output_pickle):
    print(f"Загрузка датасета: {input_pickle}")
    with open(input_pickle, 'rb') as f:
        d = pickle.load(f)
        
    total_added = 0
    total_original = 0
    total_fixed = 0
    
    for split in ['train_df', 'test_df']:
        print(f"Обработка выборки: {split}...")
        df = d[split]
        
        # Iterating over index is safer for Pandas inplace modifications
        for idx in df.index:
            mol = df.at[idx, 'rdmol']
            original_shifts = df.at[idx, 'value'][0]
            
            new_shifts = propagate_shifts_in_dict(mol, original_shifts)
            
            total_original +=  len(original_shifts)
            total_added += (len(new_shifts) - len(original_shifts))
            total_fixed += len(new_shifts)
            df.at[idx, 'value'] = [new_shifts]
            
    print("Успех!")
    print(f"Восстановлено пропущенных сдвигов 1H по симметрии: {total_added}")
    print(f"Было до восстановления 1H                        : {total_original}")
    print(f"Стало после восстановления 1Н                    : {total_fixed}")
    
    print(f"Сохранение в: {output_pickle}")
    with open(output_pickle, 'wb') as f:
        pickle.dump(d, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix missing symmetry shifts in NMR pickle dataset.")
    parser.add_argument("input_pickle", type=str, help="Путь к оригинальному .pickle датасету")
    parser.add_argument("output_pickle", type=str, help="Путь для сохранения исправленного .pickle датасета")
    
    args = parser.parse_args()
    fix_dataset_symmetry(args.input_pickle, args.output_pickle)
