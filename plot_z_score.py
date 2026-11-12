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
import glob
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from graph_conv_many_nuc_util import PredModel
import nets

USE_CUDA = True
BATCH_SIZE = 16

parser = argparse.ArgumentParser(description="Построение Z-score распределения для графовой модели ЯМР")
parser.add_argument("--state", type=str, nargs='+', required=True, 
                    help="Путь к файлу(ам) модели. Поддерживает маски, например: checkpoints/*.state")
parser.add_argument("--nucleus", type=str, choices=["13C", "1H"], default=None, 
                    help="Целевое ядро (автовывод из meta, если не указано)")
parser.add_argument("--cv_i", type=str, default=None, 
                    help="Индекс кросс-валидации (автовывод из meta, если не указано)")
parser.add_argument("--min_error", type=float, default=None, 
                    help="Минимальная ошибка std_reg (автовывод из meta, если не указано)")
parser.add_argument("--protocol", type=str, choices=["pretrain", "warmup", "molcalib"], default=None, 
                    help="Режим модели (автовывод из meta, если не указано)")
parser.add_argument("--plots-dir", type=str, default="plots_{NUC}", 
                    help="Папка для сохранения графиков. {NUC} заменится на ядро (по умолчанию: plots_{NUC})")
parser.add_argument("--meta", type=str, default="", 
                    help="Явный путь к .meta или .json файлу (автовывод из имени state)")

def analyze_and_plot_z(z_values, title_prefix, axes, n_atoms, n_mols, filename):
    nu, loc, scale = stats.t.fit(z_values)
    
    x = np.linspace(-8, 8, 1000)
    pdf_t = stats.t.pdf(x, nu, loc, scale)
    cdf_t = stats.t.cdf(x, nu, loc, scale)
    pdf_norm = stats.norm.pdf(x, 0, 1)
    cdf_norm = stats.norm.cdf(x, 0, 1)

    ax_hist, ax_cdf = axes
    
    ax_hist.hist(z_values, bins=200, range=(-8, 8), density=True, alpha=0.5, color='gray', label='Данные ($z$)')
    ax_hist.plot(x, pdf_t, 'b-', lw=2, label=f't-Стьюдент\n($\\nu$={nu:.2f}, $\\mu$={loc:.4f}, $\\sigma$={scale:.4f})')
    ax_hist.plot(x, pdf_norm, 'r--', lw=2, label='Идеал: $\\mathcal{N}(0,1)$')
    
    ax_hist.set_title(f"{filename}\n{title_prefix}: Гистограмма z-score ({n_atoms} атомов, {n_mols} молекул)", fontsize=10)
    ax_hist.set_xlabel('$z = (y - \\mu) / \\sigma$')
    ax_hist.set_ylabel('Плотность вероятности')
    ax_hist.set_xlim(-8, 8)
    ax_hist.grid(alpha=0.3)
    ax_hist.legend(loc='upper right')

    x_sort = np.sort(z_values)
    y_ecdf = np.arange(1, len(x_sort) + 1) / len(x_sort)
    mask = (x_sort >= -8) & (x_sort <= 8)
    
    ax_cdf.plot(x_sort[mask], y_ecdf[mask], color='gray', lw=2, label='Эмпирическая CDF')
    ax_cdf.plot(x, cdf_t, 'b-', lw=2, label='t-Стьюдент CDF')
    ax_cdf.plot(x, cdf_norm, 'r--', lw=2, label='Идеал: $\\mathcal{N}(0,1)$ CDF')
    
    ax_cdf.set_title(f'{title_prefix}: Интегральное распределение', fontsize=10)
    ax_cdf.set_xlabel('$z = (y - \\mu) / \\sigma$')
    ax_cdf.set_ylabel('Вероятность ($P(X < z)$)')
    ax_cdf.set_xlim(-8, 8)
    ax_cdf.grid(alpha=0.3)
    ax_cdf.legend(loc='lower right')

def plot_stats(sigma_vals, cm_vals, title_prefix, axes, std_reg, filename):
    ax_sigma, ax_cm = axes
    
    q_high_sigma = np.percentile(sigma_vals, 99) if len(sigma_vals) > 0 else 1.0
    filtered_sigma = sigma_vals[sigma_vals < (q_high_sigma * 1.5)]
    
    ax_sigma.hist(filtered_sigma, bins=100, color='teal', alpha=0.7)
    mean_sigma = np.mean(sigma_vals)
    ax_sigma.axvline(mean_sigma, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_sigma:.3f}')
    ax_sigma.axvline(std_reg, color='black', linestyle='dotted', linewidth=2, label=f'Min Reg: {std_reg:.3f}')
    
    ax_sigma.set_title(f"{filename}\n{title_prefix}: Распределение $\\sigma$ [ppm]", fontsize=10)
    ax_sigma.set_xlabel('Предсказанная ошибка ($\\sigma$) [ppm]')
    ax_sigma.set_ylabel('Частота (атомы)')
    ax_sigma.grid(alpha=0.3)
    ax_sigma.legend(loc='upper right')

    unique_cm = np.unique(cm_vals[~np.isnan(cm_vals)])
    q_low, q_high = np.percentile(unique_cm, [1, 99]) if len(unique_cm) > 0 else (-1, 1)
    filtered_cm = unique_cm[(unique_cm > q_low) & (unique_cm < q_high)]
    
    ax_cm.hist(filtered_cm, bins=100, color='orange', alpha=0.7)
    mean_cm = np.mean(filtered_cm) if len(filtered_cm) > 0 else 0
    std_cm = np.std(filtered_cm) if len(filtered_cm) > 0 else 0
    
    ax_cm.axvline(mean_cm, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_cm:.3f}')
    ax_cm.set_title(f'{title_prefix}: Сдвиг молекулы ($C_m$) [ppm] [Только 1-99 перцентиль]', fontsize=10)
    ax_cm.set_xlabel('$C_m$ [ppm]')
    ax_cm.set_ylabel('Частота (молекулы)')
    ax_cm.text(0.05, 0.95, f'Std: {std_cm:.3f}', transform=ax_cm.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_cm.grid(alpha=0.3)
    ax_cm.legend(loc='upper right')

def process_predictions(res_df, std_reg, apply_cm, spect_set):
    if spect_set == '1H':
        mu_scale = 0.05
        std_scale = 10.0
    else:
        mu_scale = 1.0
        std_scale = 1.0

    pred_mu_real = res_df['pred_mu']
    pred_std_real = res_df['pred_std'] * (mu_scale / std_scale)
    true_std_real = pred_std_real + std_reg
    
    res_df['w'] = 1.0 / (true_std_real ** 2)
    delta = res_df['value'] - pred_mu_real
    res_df['weighted_delta'] = delta * res_df['w']
    c_m_series = res_df.groupby('m_pos')['weighted_delta'].sum() / res_df.groupby('m_pos')['w'].sum()
    res_df['computed_c_m'] = res_df['m_pos'].map(c_m_series)
    
    c_m_to_apply = res_df['computed_c_m'] if apply_cm else 0.0
    
    res_df['calib_mu'] = pred_mu_real + c_m_to_apply
    res_df['z'] = (res_df['value'] - res_df['calib_mu']) / true_std_real
    
    z_clean = res_df['z'].replace([np.inf, -np.inf], np.nan).dropna().values
    sigma_clean = true_std_real.dropna().values
    cm_clean = res_df['computed_c_m'].values
    
    n_atoms = len(res_df)
    n_mols = res_df['m_pos'].nunique()
    
    return z_clean, sigma_clean, cm_clean, n_atoms, n_mols

def process_single_state(input_path, args):
    if args.meta:
        META_FILE = args.meta
    else:
        base_name = input_path
        for ext in ['.best.state', '.state', '.model', '.best.model']:
            if base_name.endswith(ext):
                base_name = base_name[:-len(ext)]
                break
        else:
            base_name = os.path.splitext(input_path)[0]
        
        base_name = re.sub(r'\.\d{8}$', '', base_name)
        
        META_FILE = base_name + ".meta"
        if not os.path.exists(META_FILE) and os.path.exists(base_name + ".json"):
            META_FILE = base_name + ".json"

    if not os.path.exists(META_FILE):
        raise FileNotFoundError(f"Файл метаданных не найден: {META_FILE}. Укажите его явно через --meta")

    if META_FILE.endswith('.json'):
        with open(META_FILE, 'r') as f:
            meta_data = json.load(f)
    else:
        with open(META_FILE, 'rb') as f:
            meta_data = pickle.load(f)

    SPECT_SET = args.nucleus if args.nucleus is not None else meta_data.get('tgt_nucs', ['13C'])[0]
    PROTOCOL = args.protocol if args.protocol is not None else meta_data.get('stage', 'warmup')

    if args.min_error is not None:
        STD_REGULARIZE = args.min_error
    else:
        loss_params = meta_data.get('loss_params', {})
        STD_REGULARIZE = loss_params.get('std_regularize', 0.8 if SPECT_SET == '13C' else 0.01)

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

    INFILE = meta_data.get('infile', os.path.join("graph_conv_many_nuc_pipeline.datasets", 
                           f"graph_conv_many_nuc_pipeline.data.{SPECT_SET}.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.{CV_I}.mol_dict.pickle"))

    APPLY_CM = (PROTOCOL == "molcalib")

    if input_path.endswith(".state"):
        MODEL_FILE = input_path.replace(".state", ".model")
    else:
        MODEL_FILE = input_path

    # Формируем пути сохранения
    original_filename = os.path.basename(input_path)
    
    # Ищем последние 8 цифр перед расширением файла (это и есть эпоха)
    match = re.search(r'\.(\d{8})\.[^.]+$', original_filename)
    if match:
        epoch = match.group(1)
        base = original_filename[:match.start()]
    else:
        epoch = "unknown"
        base = os.path.splitext(original_filename)[0]

    plots_dir = args.plots_dir.replace("{NUC}", SPECT_SET)
    os.makedirs(plots_dir, exist_ok=True)
    
    out_name_z = f"{base}.zscore.{epoch}.png"
    out_name_stats = f"{base}.stats.{epoch}.png"
    OUT_FILE_Z = os.path.join(plots_dir, out_name_z)
    OUT_FILE_STATS = os.path.join(plots_dir, out_name_stats)

    CACHE_TRAIN = os.path.splitext(input_path)[0] + "_preds_train.csv"
    CACHE_TEST = os.path.splitext(input_path)[0] + "_preds_test.csv"

    print(f"Ядро:\t\t{SPECT_SET} (cv_i: {CV_I})")
    print(f"Режим:\t\t{PROTOCOL} {'(C_m включен в лосс)' if APPLY_CM else '(C_m вычисляется для справки)'}")

    if os.path.exists(CACHE_TRAIN) and os.path.exists(CACHE_TEST):
        print(f"[CACHE HIT] Загрузка из кеша...")
        res_train = pd.read_csv(CACHE_TRAIN)
        res_test = pd.read_csv(CACHE_TEST)
    else:
        print("[CACHE MISS] Требуется запуск инференса. Подготовка модели...")
        if not os.path.exists(INFILE):
            raise FileNotFoundError(f"Файл датасета не найден: {INFILE}")

        if not os.path.exists(MODEL_FILE):
            print("  -> Автоматическая сборка .model файла...")
            net_params = meta_data['net_params']
            temp_net = nets.GraphVertModel(**net_params)
            temp_net.load_state_dict(torch.load(input_path, map_location='cpu'))
            temp_net = temp_net.cuda() if USE_CUDA else temp_net
            temp_net.eval()
            torch.save(temp_net, MODEL_FILE)

        with open(INFILE, 'rb') as f:
            d = pickle.load(f)
        
        train_df = d['train_df']
        test_df = d['test_df']
        train_df = train_df[train_df.value.map(lambda x: len(x[0]) >= 5)]
        test_df = test_df[test_df.value.map(lambda x: len(x[0]) >= 5)]

        model = PredModel(META_FILE, MODEL_FILE, USE_CUDA=USE_CUDA)
       
        with torch.no_grad():
            print("  -> Инференс Train:")
            res_train = model.pred(train_df.rdmol.tolist(), train_df.value.tolist(), BATCH_SIZE=BATCH_SIZE, prog_bar=True)
            print("  -> Инференс Test:")
            res_test = model.pred(test_df.rdmol.tolist(), test_df.value.tolist(), BATCH_SIZE=BATCH_SIZE, prog_bar=True)
            
        res_train.to_csv(CACHE_TRAIN, index=False)
        res_test.to_csv(CACHE_TEST, index=False)
        print("  -> Кеш успешно сохранен.")

    z_train, sig_train, cm_train, n_atoms_tr, n_mols_tr = process_predictions(res_train.copy(), STD_REGULARIZE, APPLY_CM, SPECT_SET)
    z_test, sig_test, cm_test, n_atoms_te, n_mols_te = process_predictions(res_test.copy(), STD_REGULARIZE, APPLY_CM, SPECT_SET)

    fig_z, axes_z = plt.subplots(2, 2, figsize=(14, 10))
    analyze_and_plot_z(z_train, "Train", axes_z[:, 0], n_atoms_tr, n_mols_tr, original_filename) 
    analyze_and_plot_z(z_test, "Test", axes_z[:, 1], n_atoms_te, n_mols_te, original_filename)   
    plt.tight_layout()
    fig_z.savefig(OUT_FILE_Z, dpi=300)
    print(f"-> Сохранено: {OUT_FILE_Z}")

    fig_s, axes_s = plt.subplots(2, 2, figsize=(14, 10))
    plot_stats(sig_train, cm_train, "Train", axes_s[:, 0], STD_REGULARIZE, original_filename)
    plot_stats(sig_test, cm_test, "Test", axes_s[:, 1], STD_REGULARIZE, original_filename)
    plt.tight_layout()
    fig_s.savefig(OUT_FILE_STATS, dpi=300)
    print(f"-> Сохранено: {OUT_FILE_STATS}")

    plt.close(fig_z)
    plt.close(fig_s)


def main():
    args, unknown = parser.parse_known_args()
    
    state_files = []
    for path in args.state:
        expanded = glob.glob(path)
        if expanded:
            state_files.extend(expanded)
        else:
            state_files.append(path)
            
    state_files = sorted(state_files)

    total = len(state_files)
    for i, state_path in enumerate(state_files, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{total}] ОБРАБОТКА ЧЕКПОИНТА: {state_path}")
        print(f"{'='*70}")
        try:
            process_single_state(state_path, args)
        except Exception as e:
            print(f"[ОШИБКА] Не удалось обработать {state_path}: {e}")

if __name__ == "__main__":
    main()
