import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch

# Импортируем классы из скриптов проекта
from graph_conv_many_nuc_util import PredModel

# --- НАСТРОЙКИ (замените на свои актуальные файлы из checkpoints_13C) ---
INFILE = "graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.13C.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle"

# Укажите актуальные пути (на скриншоте у вас был запуск 92183870)
META_FILE = "checkpoints_13C/good_13C.92183870.meta"
MODEL_FILE = "checkpoints_13C/good_13C.92183870.00000220.model"  # Замените 00000210 на номер вашей эпохи

USE_CUDA = True
BATCH_SIZE = 16

def analyze_and_plot(z_values, title_prefix, axes):
    """
    Фитирует t-распределение и строит графики PDF и CDF
    """
    # 1. Фитирование t-распределения Стьюдента
    # nu - степени свободы (отвечает за тяжесть хвостов), loc - сдвиг, scale - масштаб
    nu, loc, scale = stats.t.fit(z_values)
    print(f"[{title_prefix}] t-Student fit: nu (df) = {nu:.2f}, loc = {loc:.4f}, scale = {scale:.4f}")
    
    # 2. Подготовка сетки для графиков (обрезаем экстремальные выбросы для наглядности)
    x = np.linspace(-8, 8, 1000)
    pdf_t = stats.t.pdf(x, nu, loc, scale)
    cdf_t = stats.t.cdf(x, nu, loc, scale)
    
    # Идеальное стандартное нормальное распределение N(0,1)
    pdf_norm = stats.norm.pdf(x, 0, 1)
    cdf_norm = stats.norm.cdf(x, 0, 1)

    ax_hist, ax_cdf = axes
    
    # 3. Построение гистограммы (Probability Density Function)
    ax_hist.hist(z_values, bins=200, range=(-8, 8), density=True, alpha=0.5, color='gray', label='Данные (z)')
    ax_hist.plot(x, pdf_t, 'b-', lw=2, label=f't-Стьюдент (df={nu:.1f})')
    ax_hist.plot(x, pdf_norm, 'r--', lw=2, label='Идеал: N(0,1)')
    ax_hist.set_title(f'{title_prefix}: Гистограмма z-score')
    ax_hist.set_xlabel('z = (y - mu) / sigma')
    ax_hist.set_ylabel('Плотность вероятности')
    ax_hist.set_xlim(-8, 8)
    ax_hist.grid(alpha=0.3)
    ax_hist.legend()

    # 4. Построение интегрального распределения (Cumulative Distribution Function)
    x_sort = np.sort(z_values)
    y_ecdf = np.arange(1, len(x_sort) + 1) / len(x_sort)
    
    # Ограничиваем X для CDF, чтобы хвосты не сжимали весь график
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
    print(f"Загрузка датасета {INFILE}...")
    with open(INFILE, 'rb') as f:
        d = pickle.load(f)
    
    train_df = d['train_df']
    test_df = d['test_df']
    print(f"Размер train: {len(train_df)} молекул. Размер test: {len(test_df)} молекул.")

    print(f"Загрузка модели из {MODEL_FILE}...")
    model = PredModel(META_FILE, MODEL_FILE, USE_CUDA=USE_CUDA)
    
    print("\nПредсказание для тренировочной выборки (Train)...")
    with torch.no_grad():
        res_train = model.pred(train_df.rdmol.tolist(), train_df.value.tolist(), BATCH_SIZE=BATCH_SIZE, prog_bar=True)
    res_train['z'] = (res_train['value'] - res_train['pred_mu']) / res_train['pred_std']
    
    print("\nПредсказание для тестовой выборки (Test)...")
    with torch.no_grad():
        res_test = model.pred(test_df.rdmol.tolist(), test_df.value.tolist(), BATCH_SIZE=BATCH_SIZE, prog_bar=True)
    res_test['z'] = (res_test['value'] - res_test['pred_mu']) / res_test['pred_std'] 
    
    # Удаление NaN / Inf (на случай деленияFalseees_train['z'].replace([np.inf, -np.inf], np.nan).dropna().values
    z_train = res_train['z'].replace([np.inf, -np.inf], np.nan).dropna().values
    z_test = res_test['z'].replace([np.inf, -np.inf], np.nan).dropna().values

    print("\nПостроение графиков и фитирование...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    analyze_and_plot(z_train, "Train", axes[:, 0]) # Левая колонка - Train
    analyze_and_plot(z_test, "Test", axes[:, 1])   # Правая колонка - Test
    
    plt.tight_layout()
    plt.savefig("z_distribution_analysis.png", dpi=300)
    print("Готово! Графики сохранены в файл 'z_distribution_analysis.png'")
    plt.show()

if __name__ == "__main__":
    main()
