import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import argparse
import json
import re

from graph_conv_many_nuc_util import PredModel
import nets

# === ПАРСИНГ АРГУМЕНТОВ ===
parser = argparse.ArgumentParser(description="Построение Z-score распределения для графовой модели ЯМР")
parser.add_argument("--state", type=str, required=True, 
                    help="Путь к файлу модели (.model или .state)")
parser.add_argument("--nucleus", type=str, choices=["13C", "1H"], default=None, 
                    help="Целевое ядро (автовывод из meta, если не указано)")
parser.add_argument("--cv_i", type=int, default=None, 
                    help="Индекс кросс-валидации (автовывод из meta, если не указано)")
parser.add_argument("--min_error", type=float, default=None, 
                    help="Минимальная ошибка std_reg (автовывод из meta, если не указано)")
parser.add_argument("--protocol", type=str, choices=["pretrain", "warmup", "molcalib"], default=None, 
                    help="Режим модели (автовывод из meta, если не указано)")
parser.add_argument("--out", type=str, default="", 
                    help="Явный путь к выходному графику Z-score")
parser.add_argument("--meta", type=str, default="", 
                    help="Явный путь к .meta или .json файлу (автовывод из имени state)")

args, unknown = parser.parse_known_args()
INPUT_PATH = args.state

# 1. АВТООПРЕДЕЛЕНИЕ ПУТИ К МЕТАДАННЫМ
if args.meta:
    META_FILE = args.meta
else:
    base_name = INPUT_PATH
    for ext in ['.best.state', '.state', '.model', '.best.model']:
        if base_name.endswith(ext):
            base_name = base_name[:-len(ext)]
            break
    else:
        base_name = os.path.splitext(INPUT_PATH)[0]
    
    # Убираем суффикс с номером эпохи (например, .00000000), если он есть
    base_name = re.sub(r'\.\d{8}$', '', base_name)
    
    # Ищем meta или json
    META_FILE = base_name + ".meta"
    if not os.path.exists(META_FILE) and os.path.exists(base_name + ".json"):
        META_FILE = base_name + ".json"


# 2. ЗАГРУЗКА МЕТАДАННЫХ
if not os.path.exists(META_FILE):
    raise FileNotFoundError(f"Файл метаданных не найден: {META_FILE}. Укажите его явно через --meta")

print(f"Чтение параметров из {META_FILE}...")
if META_FILE.endswith('.json'):
    with open(META_FILE, 'r') as f:
        meta_data = json.load(f)
else:
    with open(META_FILE, 'rb') as f:
        meta_data = pickle.load(f)

# 3. УМНЫЙ ПАРСИНГ ПАРАМЕТРОВ (Приоритет: аргументы CMD -> метаданные)
SPECT_SET = args.nucleus if args.nucleus is not None else meta_data.get('tgt_nucs', ['13C'])[0]
PROTOCOL = args.protocol if args.protocol is not None else meta_data.get('stage', 'warmup')

if args.min_error is not None:
    STD_REGULARIZE = args.min_error
else:
    loss_params = meta_data.get('loss_params', {})
    STD_REGULARIZE = loss_params.get('std_regularize', 0.8 if SPECT_SET == '13C' else 0.03)

if args.cv_i is not None:
    CV_I = args.cv_i
else:
    infile_path = meta_data.get('infile', '')
    try:
        parts = infile_path.split('.')
        idx = parts.index('mol_dict')
        CV_I = int(parts[idx-1])
    except:
        CV_I = 0

# Берем точный путь датасета из обучения
INFILE = meta_data.get('infile', os.path.join("graph_conv_many_nuc_pipeline.datasets", 
                       f"graph_conv_many_nuc_pipeline.data.{SPECT_SET}.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.{CV_I}.mol_dict.pickle"))

APPLY_CM = (PROTOCOL == "molcalib")

# 4. РАБОТА С ФАЙЛАМИ МОДЕЛИ
if INPUT_PATH.endswith(".state"):
    MODEL_FILE = INPUT_PATH.replace(".state", ".model")
    if not os.path.exists(MODEL_FILE):
        print(f"Обнаружен .state файл, но файл {MODEL_FILE} отсутствует.")
        print("Автоматическая сборка и сохранение недостающего .model файла...")
        try:
            net_params = meta_data['net_params']
            temp_net = nets.GraphVertModel(**net_params)
            temp_net.load_state_dict(torch.load(INPUT_PATH, map_location='cpu'))
            temp_net = temp_net.cuda()
            temp_net.eval()
            torch.save(temp_net, MODEL_FILE)
            print(f"Успешно создан файл модели: {MODEL_FILE}")
        except Exception as e:
            raise RuntimeError(f"Не удалось автоматически создать .model файл: {e}")
else:
    MODEL_FILE = INPUT_PATH

if args.out:
    OUT_FILE_Z = args.out
    base_out = os.path.splitext(OUT_FILE_Z)[0]
    OUT_FILE_STATS = base_out + "_stats.png"
else:
    base_out = os.path.splitext(MODEL_FILE)[0]
    OUT_FILE_Z = base_out + ".png"
    OUT_FILE_STATS = base_out + "_stats.png"

USE_CUDA = True
BATCH_SIZE = 16

# ================= ФУНКЦИИ ОТРИСОВКИ =================
def analyze_and_plot_z(z_values, title_prefix, axes):
    nu, loc, scale = stats.t.fit(z_values)
    print(f"[{title_prefix}] Z-score t-Student fit: nu (df) = {nu:.2f}, loc = {loc:.4f}, scale = {scale:.4f}")
    
    x = np.linspace(-8, 8, 1000)
    pdf_t = stats.t.pdf(x, nu, loc, scale)
    cdf_t = stats.t.cdf(x, nu, loc, scale)
    pdf_norm = stats.norm.pdf(x, 0, 1)
    cdf_norm = stats.norm.cdf(x, 0, 1)

    ax_hist, ax_cdf = axes
    
    ax_hist.hist(z_values, bins=200, range=(-8, 8), density=True, alpha=0.5, color='gray', label='Данные (z)')
    ax_hist.plot(x, pdf_t, 'b-', lw=2, label=f't-Стьюдент (df={nu:.1f})')
    ax_hist.plot(x, pdf_norm, 'r--', lw=2, label='Идеал: N(0,1)')
    ax_hist.set_title(f'{title_prefix}: Гистограмма z-score')
    ax_hist.set_xlabel('z = (y - mu) / sigma')
    ax_hist.set_ylabel('Плотность вероятности')
    ax_hist.set_xlim(-8, 8)
    ax_hist.grid(alpha=0.3)
    ax_hist.legend()

    x_sort = np.sort(z_values)
    y_ecdf = np.arange(1, len(x_sort) + 1) / len(x_sort)
    mask = (x_sort >= -8) & (x_sort <= 8)
    
    ax_cdf.plot(x_sort[mask], y_ecdf[mask], color='gray', lw=2, label='Эмпирическая CDF')
    ax_cdf.plot(x, cdf_t, 'b-', lw=2, label='t-Стьюдент CDF')
    ax_cdf.plot(x, cdf_norm, 'r--', lw=2, label='Идеал: N(0,1) CDF')
    ax_cdf.set_title(f'{title_prefix}: Интегральное распределение')
    ax_cdf.set_xlabel('z = (y - mu) / sigma')
    ax_cdf.set_ylabel('Вероятность (P(X < z))')
    ax_cdf.set_xlim(-8, 8)
    ax_cdf.grid(alpha=0.3)
    ax_cdf.legend()

def plot_stats(sigma_vals, cm_vals, title_prefix, axes, std_reg):
    ax_sigma, ax_cm = axes
    
    ax_sigma.hist(sigma_vals, bins=100, color='teal', alpha=0.7)
    mean_sigma = np.mean(sigma_vals)
    ax_sigma.axvline(mean_sigma, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_sigma:.3f}')
    ax_sigma.axvline(std_reg, color='black', linestyle='dotted', linewidth=2, label=f'Min Reg: {std_reg:.3f}')
    ax_sigma.set_title(f'{title_prefix}: Распределение Sigma ($\sigma$)')
    ax_sigma.set_xlabel('Предсказанная ошибка ($\sigma$) [ppm]')
    ax_sigma.set_ylabel('Частота (атомы)')
    ax_sigma.grid(alpha=0.3)
    ax_sigma.legend()

    unique_cm = np.unique(cm_vals[~np.isnan(cm_vals)])
    q_low, q_high = np.percentile(unique_cm, [1, 99]) if len(unique_cm) > 0 else (-1, 1)
    filtered_cm = unique_cm[(unique_cm > q_low) & (unique_cm < q_high)]
    
    ax_cm.hist(filtered_cm, bins=100, color='orange', alpha=0.7)
    mean_cm = np.mean(filtered_cm) if len(filtered_cm) > 0 else 0
    std_cm = np.std(filtered_cm) if len(filtered_cm) > 0 else 0
    
    ax_cm.axvline(mean_cm, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_cm:.3f}')
    ax_cm.set_title(f'{title_prefix}: Сдвиг молекулы (C_m) [Только 1-99 перцентиль]')
    ax_cm.set_xlabel('C_m [ppm]')
    ax_cm.set_ylabel('Частота (молекулы)')
    ax_cm.text(0.05, 0.95, f'Std: {std_cm:.3f}', transform=ax_cm.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_cm.grid(alpha=0.3)
    ax_cm.legend()

def main():
    print(f"--- НАСТРОЙКИ ЗАПУСКА (АВТООПРЕДЕЛЕНИЕ) ---")
    print(f"Ядро:\t\t{SPECT_SET} (cv_i: {CV_I})")
    print(f"Модель:\t\t{MODEL_FILE}")
    print(f"Датасет:\t{os.path.basename(INFILE)}")
    print(f"Мин. ошибка:\t{STD_REGULARIZE} ppm")
    print(f"Режим:\t\t{PROTOCOL} {'(C_m включен в лосс)' if APPLY_CM else '(C_m вычисляется для справки)'}")
    print(f"Сохранение Z:\t{OUT_FILE_Z}")
    print(f"-------------------------------------------\n")

    if not os.path.exists(INFILE):
        raise FileNotFoundError(f"Файл датасета не найден: {INFILE}")

    print(f"Загрузка датасета...")
    with open(INFILE, 'rb') as f:
        d = pickle.load(f)
    
    train_df = d['train_df']
    test_df = d['test_df']
    
    # Фильтрация >= 5 ядер, как при обучении
    train_df = train_df[train_df.value.map(len) >= 5]
    test_df = test_df[test_df.value.map(len) >= 5]
    print(f"Размер train: {len(train_df)} молекул. Размер test: {len(test_df)} молекул.")

    print(f"Загрузка модели...")
    model = PredModel(META_FILE, MODEL_FILE, USE_CUDA=USE_CUDA)
   
    print("\nПредсказание для тренировочной выборки (Train)...")
    with torch.no_grad():
        res_train = model.pred(train_df.rdmol.tolist(), train_df.value.tolist(), BATCH_SIZE=BATCH_SIZE, prog_bar=True)
    
    print("\nПредсказание для тестовой выборки (Test)...")
    with torch.no_grad():
        res_test = model.pred(test_df.rdmol.tolist(), test_df.value.tolist(), BATCH_SIZE=BATCH_SIZE, prog_bar=True)

    def process_predictions(res_df, std_reg, apply_cm):
        true_std = res_df['pred_std'] + std_reg
        
        # Для pretrain дисперсия sigma случайна, c_m будет неадекватным. 
        # Но чтобы графики не падали с ошибками, считаем его.
        res_df['w'] = 1.0 / (true_std ** 2)
        delta = res_df['value'] - res_df['pred_mu']
        res_df['weighted_delta'] = delta * res_df['w']
        c_m_series = res_df.groupby('m_pos')['weighted_delta'].sum() / res_df.groupby('m_pos')['w'].sum()
        res_df['computed_c_m'] = res_df['m_pos'].map(c_m_series)
        
        c_m_to_apply = res_df['computed_c_m'] if apply_cm else 0.0
        
        res_df['calib_mu'] = res_df['pred_mu'] + c_m_to_apply
        res_df['z'] = (res_df['value'] - res_df['calib_mu']) / true_std
        
        z_clean = res_df['z'].replace([np.inf, -np.inf], np.nan).dropna().values
        sigma_clean = true_std.dropna().values
        cm_clean = res_df['computed_c_m'].values
        
        return z_clean, sigma_clean, cm_clean

    print("\nПостроение графиков Z-Score...")
    z_train, sig_train, cm_train = process_predictions(res_train, STD_REGULARIZE, APPLY_CM)
    z_test, sig_test, cm_test = process_predictions(res_test, STD_REGULARIZE, APPLY_CM)

    fig_z, axes_z = plt.subplots(2, 2, figsize=(14, 10))
    analyze_and_plot_z(z_train, "Train", axes_z[:, 0]) 
    analyze_and_plot_z(z_test, "Test", axes_z[:, 1])   
    plt.tight_layout()
    fig_z.savefig(OUT_FILE_Z, dpi=300)
    print(f"Графики Z-score сохранены в: '{OUT_FILE_Z}'")

    print("\nПостроение графиков статистики (Sigma и C_m)...")
    fig_s, axes_s = plt.subplots(2, 2, figsize=(14, 10))
    plot_stats(sig_train, cm_train, "Train", axes_s[:, 0], STD_REGULARIZE)
    plot_stats(sig_test, cm_test, "Test", axes_s[:, 1], STD_REGULARIZE)
    plt.tight_layout()
    fig_s.savefig(OUT_FILE_STATS, dpi=300)
    print(f"Графики статистики сохранены в: '{OUT_FILE_STATS}'")

if __name__ == "__main__":
    main()
