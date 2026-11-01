import torch
import argparse
import os
import pickle
import json
import nets

def main():
    parser = argparse.ArgumentParser(description="Конвертация 13C state для инициализации 1H модели (Partial Transfer Learning)")
    parser.add_argument("--state_13c", type=str, required=True, 
                        help="Путь к готовому .state файлу обученной модели 13C")
    parser.add_argument("--meta_13c", type=str, required=True, 
                        help="Путь к .meta или .json файлу от 13C (архитектуры экстрактора идентичны)")
    parser.add_argument("--output", type=str, required=True, 
                        help="Путь для сохранения нового .state файла, готового для запуска 1H")
    
    args = parser.parse_args()

    # 1. Загружаем метаданные от 13C. 
    # Архитектура экстрактора фичей (gml) у них идентична, а количество выходных ядер (OUT_DIM=1) тоже совпадает.
    print(f"Загрузка метаданных 13C из {args.meta_13c}...")
    if args.meta_13c.endswith('.json'):
        with open(args.meta_13c, 'r') as f:
            meta = json.load(f)
    else:
        with open(args.meta_13c, 'rb') as f:
            meta = pickle.load(f)
            
    net_params = meta['net_params']

    # 2. Создаем чистую модель с правильной нулевой/случайной инициализацией весов/смещений.
    # Это сгенерирует свежие веса для финальных слоев (lin_out), чтобы сеть не была смещена к значениям 13C.
    net_new = nets.GraphVertModel(**net_params)
    state_new = net_new.state_dict()

    # 3. Загружаем обученные веса 13C модели (weights_only=True убирает ворнинг/ошибку безопасности)
    print(f"Загрузка весов 13C из {args.state_13c}...")
    state_13c = torch.load(args.state_13c, map_location='cpu', weights_only=True)

    # 4. Переносим ТОЛЬКО веса графовых сверточных слоев (gml.*) и батч-нормализации на входе
    transferred_count = 0
    for k in state_new.keys():
        if (k.startswith('gml.') or k.startswith('batch_norm.')) and k in state_13c:
            state_new[k] = state_13c[k]
            transferred_count += 1

    print(f"Успешно перенесено слоев графового экстрактора (gml.*): {transferred_count}")

    # 5. Сохраняем новый комбинированный state-файл
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(state_new, args.output)
    print(f"Готово! Новый state сохранен в: {args.output}")

if __name__ == "__main__":
    main()
