import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import argparse

# Импортируем классы из скриптов проекта
from graph_conv_many_nuc_util import PredModel

# === ПАРСИНГ АРГУМЕНТОВ ===
parser = argparse.ArgumentParser(description="Построение Z-score распределения для графовой модели ЯМР")
parser.add_argument("--model", type=str, required=True, 
                    help="Путь к файлу модели (.model)")
parser.add_argument("--nucleus", type=str, choices=["13C", "1H"], default="13C", 
                    help="Целевое ядро (по умолчанию 13C)")
parser.add_argument("--cv_i", type=int, default=0, 
                    help="Индекс кросс-валидации датасета (по умолчанию 0)")
parser.add_argument("--min_error", type=float, default=None, 
                    help="Минимальная ошибка (std_reg). По умолчанию: 13C=0.8, 1H=0.08")
parser.add_argument("--protocol", type=str, choices=["warmup", "molcalib"], default="warmup", 
                    help="Режим модели: warmup (без c_m) или molcalib (с c_m)")
parser.add_argument("--out", type=str, default="", 
                    help="Явный путь к выходному графику (по умолчанию - как имя model файла с .png)")
parser.add_argument("--meta", type=str, default="", 
                    help="Явный путь к .meta файлу (по умолчанию выводится из имени model файла)")

args, unknown = parser.parse_known_args()

# --- НАСТРОЙКИ ПУТЕЙ И ПАРАМЕТРОВ ---
SPECT_SET = args.nucleus
CV_I = args.cv_i
DATASET_DIR = "graph_conv_many_nuc_pipeline.datasets"
DATASET_NAME = "nmrshiftdb_hconfspcl_nmrshiftdb"
MAT_PROPS = "aromatic"
TGT_MAX_N = 64

INFILE = os.path.join(DATASET_DIR, f"graph_conv_many_nuc_pipeline.data.{SPECT_SET}.{DATASET_NAME}.{MAT_PROPS}.{TGT_MAX_N}.{CV_I}.mol_dict.pickle")
MODEL_FILE = args.model

# Автоматическое определение META_FILE (отрезаем ".00000300.model" и добавляем ".meta")
if args.meta:
    META_FILE = args.meta
else:
    parts = MODEL_FILE.split('.')
    if len(parts) >= 3:
        META_FILE = ".".join(parts[:-2]) + ".meta"
    else:
        META_FILE = os.path.splitext(MODEL_FILE)[0] + ".meta"

# Автоматическое определение OUT_FILE
if args.out:
    OUT_FILE = args.out
else:
    OUT_FILE = os.path.splitext(MODEL_FILE)[0] + ".png"

# Определение регуляризатора
if args.min_error is not None:
    STD_REGULARIZE = args.min_error
else:
    STD_REGULARIZE = 0.8 if SPECT_SET == '13C' else 0.08

APPLY_CM = (args.protocol == "molcalib")

USE_CUDA = True
BATCH_SIZE = 16

def analyze_and_plot(z_values, title_prefix, axes):
    """
    Фитирует t-распределение и строит графики PDF и CDF
    """
    nu, loc, scale = stats.t.fit(z_values)
    print(f"[{title_prefix}] t-Student fit: nu (df) = {nu:.2f}, loc = {loc:.4f}, scale = {scale:.4f}")
    
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

def main():
    print(f"--- НАСТРОЙКИ ЗАПУСКА ---")
    print(f"Ядро:\t\t{SPECT_SET} (cv_i: {CV_I})")
    print(f"Модель:\t\t{MODEL_FILE}")
    print(f"Meta файл:\t{META_FILE}")
    print(f"Мин. ошибка:\t{STD_REGULARIZE} ppm")
    print(f"Сдвиг c_m:\t{'ВКЛЮЧЕН (molcalib)' if APPLY_CM else 'ОТКЛЮЧЕН (warmup)'}")
    print(f"Сохранение в:\t{OUT_FILE}")
    print(f"-------------------------\n")

    print(f"Загрузка датасета {INFILE}...")
    with open(INFILE, 'rb') as f:
        d = pickle.load(f)
    
    train_df = d['train_df']
    test_df = d['test_df']
    print(f"Размер train: {len(train_df)} молекул. Размер test: {len(test_df)} молекул.")

    print(f"Загрузка модели...")
    model = PredModel(META_FILE, MODEL_FILE, USE_CUDA=USE_CUDA)
   
    print("\nПредсказание для тренировочной выборки (Train)...")
    with torch.no_grad():
        res_train = model.pred(train_df.rdmol.tolist(), train_df.value.tolist(), BATCH_SIZE=BATCH_SIZE, prog_bar=True)
    
    print("\nПредсказание для тестовой выборки (Test)...")
    with torch.no_grad():
        res_test = model.pred(test_df.rdmol.tolist(), test_df.value.tolist(), BATCH_SIZE=BATCH_SIZE, prog_bar=True)

    def compute_calibrated_z(res_df, std_reg=0.8, apply_cm=True):
        """ Считает Z-score с учетом c_m и регуляризатора дисперсии """
        true_std = res_df['pred_std'] + std_reg
        
        if apply_cm:
            res_df['w'] = 1.0 / (true_std ** 2)
            delta = res_df['value'] - res_df['pred_mu']
            res_df['weighted_delta'] = delta * res_df['w']
            c_m_series = res_df.groupby('m_pos')['weighted_delta'].sum() / res_df.groupby('m_pos')['w'].sum()
            res_df['c_m'] = res_df['m_pos'].map(c_m_series)
        else:
            res_df['c_m'] = 0.0
        
        res_df['calib_mu'] = res_df['pred_mu'] + res_df['c_m']
        res_df['z'] = (res_df['value'] - res_df['calib_mu']) / true_std
        
        return res_df['z'].replace([np.inf, -np.inf], np.nan).dropna().values

    z_train = compute_calibrated_z(res_train, std_reg=STD_REGULARIZE, apply_cm=APPLY_CM)
    z_test = compute_calibrated_z(res_test, std_reg=STD_REGULARIZE, apply_cm=APPLY_CM)

    print("\nПостроение графиков и фитирование...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    analyze_and_plot(z_train, "Train", axes[:, 0]) 
    analyze_and_plot(z_test, "Test", axes[:, 1])   
    
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=300)
    print(f"Готово! Графики сохранены в файл '{OUT_FILE}'")

if __name__ == "__main__":
    main()
