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
    parser.add_argument("--meta_1h", type=str, required=True, 
                        help="Путь к .meta или .json файлу для 1H (нужен для построения архитектуры)")
    parser.add_argument("--output", type=str, required=True, 
                        help="Путь для сохранения нового .state файла, готового для запуска 1H")
    
    args = parser.parse_args()

    # 1. Загружаем метаданные для 1H, чтобы создать правильную чистую архитектуру
    print(f"Загрузка метаданных 1H из {args.meta_1h}...")
    if args.meta_1h.endswith('.json'):
        with open(args.meta_1h, 'r') as f:
            meta = json.load(f)
    else:
        with open(args.meta_1h, 'rb') as f:
            meta = pickle.load(f)
            
    net_params = meta['net_params']

    # 2. Создаем чистую модель 1H с правильной нулевой инициализацией весов/смещений
    net_1h = nets.GraphVertModel(**net_params)
    state_1h = net_1h.state_dict()

    # 3. Загружаем обученные веса 13C модели
    print(f"Загрузка весов 13C из {args.state_13c}...")
    state_13c = torch.load(args.state_13c, map_location='cpu')

    # 4. Переносим ТОЛЬКО веса графовых сверточных слоев (gml.*)
    # Финальные слои (lin_out) остаются инициализированными с нуля под 1H
    transferred_count = 0
    for k in state_1h.keys():
        if k.startswith('gml.') and k in state_13c:
            state_1h[k] = state_13c[k]
            transferred_count += 1

    print(f"Успешно перенесено слоев графового экстрактора (gml.*): {transferred_count}")

    # 5. Сохраняем новый комбинированный state-файл
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(state_1h, args.output)
    print(f"Готово! Новый протонный state сохранен в: {args.output}")

if __name__ == "__main__":
    main()
