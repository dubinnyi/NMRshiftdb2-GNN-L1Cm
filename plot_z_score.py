import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import argparse
import json
import re
from glob import glob

import graph_conv_many_nuc_util
from graph_conv_many_nuc_util import PredModel
import nets
from i18n import _


def epoch_res_to_df(res, tgt_nucs):
    """
    Fast vectorized conversion of `run_epoch(..., return_pred=True)` dictionary output
    into a DataFrame matching `PredModel.pred` format (columns: pred_mu, pred_std,
    nuc_i, nuc, atom_idx, m_pos, value).
    """
    pred_mu = res['pred_mu']
    pred_std = res['pred_std']
    pred_mask = res['pred_mask']
    pred_truth = res['pred_truth']

    dfs = []
    for nuc_i, nuc in enumerate(tgt_nucs):
        m_pos_idx, atom_idx = np.where(pred_mask[:, :, nuc_i] > 0)
        df_nuc = pd.DataFrame({
            'pred_mu': pred_mu[m_pos_idx, atom_idx, nuc_i].astype(np.float64),
            'pred_std': pred_std[m_pos_idx, atom_idx, nuc_i].astype(np.float64),
            'nuc_i': nuc_i,
            'nuc': nuc,
            'atom_idx': atom_idx,
            'm_pos': m_pos_idx,
            'value': pred_truth[m_pos_idx, atom_idx, nuc_i].astype(np.float64),
        })
        dfs.append(df_nuc)

    if len(dfs) == 0:
        return pd.DataFrame(columns=['pred_mu', 'pred_std', 'nuc_i', 'nuc', 'atom_idx', 'm_pos', 'value'])
    return pd.concat(dfs, ignore_index=True)


def analyze_and_plot_z(z_values, title_prefix, axes, n_atoms=None, n_mols=None, filename=""):
    z_values = z_values[np.isfinite(z_values)]
    if len(z_values) < 2:
        print(_("[{title_prefix}] WARNING: z-score array is empty (model weights may contain NaN). Skipping plot.",
                title_prefix=title_prefix))
        return

    nu, loc, scale = stats.t.fit(z_values)
    print(f"[{title_prefix}] Z-score t-Student fit: nu (df) = {nu:.2f}, loc = {loc:.4f}, scale = {scale:.4f}")

    x = np.linspace(-8, 8, 1000)
    pdf_t = stats.t.pdf(x, nu, loc, scale)
    cdf_t = stats.t.cdf(x, nu, loc, scale)
    pdf_norm = stats.norm.pdf(x, 0, 1)
    cdf_norm = stats.norm.cdf(x, 0, 1)

    ax_hist, ax_cdf = axes

    in_range = (z_values >= -8) & (z_values <= 8)
    if np.any(in_range):
        ax_hist.hist(z_values[in_range], bins=200, range=(-8, 8), density=True, alpha=0.5, color='gray',
                     label=_("Data (z)"))
    ax_hist.plot(x, pdf_t, 'b-', lw=2,
                 label=_("Student-t\n(df={nu:.2f}, loc={loc:.4f}, scale={scale:.4f})",
                         nu=nu, loc=loc, scale=scale))
    ax_hist.plot(x, pdf_norm, 'r--', lw=2, label=_("Ideal: N(0,1)"))

    counts_str = (
        _(" ({n_atoms} atoms, {n_mols} molecules)", n_atoms=n_atoms, n_mols=n_mols)
        if (n_atoms is not None and n_mols is not None) else ""
    )
    prefix_str = f"{filename}\n" if filename else ""
    ax_hist.set_title(
        _("{prefix_str}{title_prefix}: Z-score Histogram{counts_str}",
          prefix_str=prefix_str, title_prefix=title_prefix, counts_str=counts_str),
        fontsize=10
    )
    ax_hist.set_xlabel('z = (y - mu) / sigma')
    ax_hist.set_ylabel(_("Probability Density"))
    ax_hist.set_xlim(-8, 8)
    ax_hist.grid(alpha=0.3)
    ax_hist.legend(loc='upper right')

    x_sort = np.sort(z_values)
    y_ecdf = np.arange(1, len(x_sort) + 1) / len(x_sort)
    mask = (x_sort >= -8) & (x_sort <= 8)

    ax_cdf.plot(x_sort[mask], y_ecdf[mask], color='gray', lw=2, label=_("Empirical CDF"))
    ax_cdf.plot(x, cdf_t, 'b-', lw=2, label=_("Student-t CDF"))
    ax_cdf.plot(x, cdf_norm, 'r--', lw=2, label=_("Ideal: N(0,1) CDF"))
    ax_cdf.set_title(_("{title_prefix}: Cumulative Distribution (CDF)", title_prefix=title_prefix), fontsize=10)
    ax_cdf.set_xlabel('z = (y - mu) / sigma')
    ax_cdf.set_ylabel(_("Probability (P(X < z))"))
    ax_cdf.set_xlim(-8, 8)
    ax_cdf.grid(alpha=0.3)
    ax_cdf.legend(loc='lower right')


def plot_stats(sigma_vals, cm_vals, title_prefix, axes, std_reg, filename=""):
    ax_sigma, ax_cm = axes

    sigma_vals = sigma_vals[np.isfinite(sigma_vals)]
    if len(sigma_vals) == 0:
        return

    q_high_sigma = np.percentile(sigma_vals, 99) if len(sigma_vals) > 0 else 1.0
    filtered_sigma = sigma_vals[sigma_vals <= (q_high_sigma * 1.5)] if len(sigma_vals) > 0 else sigma_vals

    ax_sigma.hist(filtered_sigma, bins=100, color='teal', alpha=0.7)
    mean_sigma = np.mean(sigma_vals) if len(sigma_vals) > 0 else 0.0
    ax_sigma.axvline(mean_sigma, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_sigma:.3f}')
    ax_sigma.axvline(std_reg, color='black', linestyle='dotted', linewidth=2, label=f'Min Reg: {std_reg:.3f}')

    prefix_str = f"{filename}\n" if filename else ""
    ax_sigma.set_title(
        _("{prefix_str}{title_prefix}: Sigma Distribution [ppm]",
          prefix_str=prefix_str, title_prefix=title_prefix),
        fontsize=10
    )
    ax_sigma.set_xlabel(_("Predicted Uncertainty (sigma) [ppm]"))
    ax_sigma.set_ylabel(_("Frequency (atoms)"))
    ax_sigma.grid(alpha=0.3)
    ax_sigma.legend(loc='upper right')

    unique_cm = np.unique(cm_vals[~np.isnan(cm_vals)])
    q_low, q_high = np.percentile(unique_cm, [1, 99]) if len(unique_cm) > 1 else (-1, 1)
    filtered_cm = unique_cm[(unique_cm >= q_low) & (unique_cm <= q_high)] if len(unique_cm) > 1 else unique_cm

    ax_cm.hist(filtered_cm, bins=100, color='orange', alpha=0.7)
    mean_cm = np.mean(filtered_cm) if len(filtered_cm) > 0 else 0.0
    std_cm = np.std(filtered_cm) if len(filtered_cm) > 0 else 0.0

    ax_cm.axvline(mean_cm, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_cm:.3f}')
    ax_cm.set_title(
        _("{title_prefix}: Per-Molecule Shift (C_m) [ppm] [1-99 percentile]", title_prefix=title_prefix),
        fontsize=10
    )
    ax_cm.set_xlabel('C_m [ppm]')
    ax_cm.set_ylabel(_("Frequency (molecules)"))
    ax_cm.text(0.05, 0.95, f'Std: {std_cm:.3f}', transform=ax_cm.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_cm.grid(alpha=0.3)
    ax_cm.legend(loc='upper right')


def process_predictions(res_df, std_reg, apply_cm, spect_set='13C'):
    """
    Convert predicted uncertainty values into physical ppm units (accounting for
    mu_scale / std_scale for 1H) and compute per-molecule calibration shifts C_m and Z-scores.
    """
    if spect_set == '1H':
        mu_scale = 0.05
        std_scale = 10.0
    else:
        mu_scale = 1.0
        std_scale = 1.0

    pred_mu_real = res_df['pred_mu'].astype(np.float64)
    pred_std_real = (res_df['pred_std'].astype(np.float64) + std_reg) * (mu_scale / std_scale)
    std_reg_real = std_reg * (mu_scale / std_scale)

    res_df = res_df.copy()
    res_df['w'] = 1.0 / (pred_std_real ** 2)
    delta = res_df['value'].astype(np.float64) - pred_mu_real
    res_df['weighted_delta'] = delta * res_df['w']
    c_m_series = res_df.groupby('m_pos')['weighted_delta'].sum() / res_df.groupby('m_pos')['w'].sum()
    res_df['computed_c_m'] = res_df['m_pos'].map(c_m_series)

    c_m_to_apply = res_df['computed_c_m'] if apply_cm else 0.0

    res_df['calib_mu'] = pred_mu_real + c_m_to_apply
    res_df['z'] = (res_df['value'].astype(np.float64) - res_df['calib_mu']) / pred_std_real

    z_clean = res_df['z'].replace([np.inf, -np.inf], np.nan).dropna().values
    sigma_clean = pred_std_real.dropna().values
    cm_clean = res_df['computed_c_m'].values

    n_atoms = len(res_df)
    n_mols = res_df['m_pos'].nunique()

    return z_clean, sigma_clean, cm_clean, n_atoms, n_mols, std_reg_real


def compute_and_append_epoch_metrics(res_train_df, res_test_df, epoch_i,
                                     std_reg, protocol, spect_set,
                                     cm_norm="L2", lambda_penalty=0.0,
                                     metrics_csv_path=None):
    """
    Compute comprehensive training monitoring metrics (MAE before and after C_m calibration,
    pseudo-NLL log_normal_nolog, C_m penalty, Sigma and C_m statistics, Student-t parameters
    nu, loc, scale for raw and calibrated Z-scores, df divergence d_nu, and scale ratio ratio_sc)
    and append a summary row to the cumulative CSV file every 10 epochs.
    """
    if spect_set == '1H':
        mu_scale = 0.05
        std_scale = 10.0
    else:
        mu_scale = 1.0
        std_scale = 1.0

    row = {
        'epoch': int(epoch_i),
        'protocol': protocol,
        'cm_norm': cm_norm,
        'lambda_penalty': float(lambda_penalty),
    }

    for tag, df in [('tr', res_train_df), ('te', res_test_df)]:
        if len(df) == 0:
            continue
        std_real = (df['pred_std'].astype(np.float64) + std_reg) * (mu_scale / std_scale)
        delta_raw = df['value'].astype(np.float64) - df['pred_mu'].astype(np.float64)
        w = 1.0 / (std_real ** 2)

        df_tmp = pd.DataFrame({
            'm_pos': df['m_pos'],
            'wd': delta_raw * w,
            'w': w,
        })
        cm_per_mol = df_tmp.groupby('m_pos')['wd'].sum() / df_tmp.groupby('m_pos')['w'].sum()
        atom_cm = df['m_pos'].map(cm_per_mol)
        delta_cal = delta_raw - atom_cm

        z_raw = (delta_raw / std_real).replace([np.inf, -np.inf], np.nan).dropna().values
        z_cal = (delta_cal / std_real).replace([np.inf, -np.inf], np.nan).dropna().values

        if len(z_raw) >= 2:
            nu_raw, loc_raw, sc_raw = stats.t.fit(z_raw)
        else:
            nu_raw, loc_raw, sc_raw = np.nan, np.nan, np.nan

        if len(z_cal) >= 2:
            nu_cal, loc_cal, sc_cal = stats.t.fit(z_cal)
        else:
            nu_cal, loc_cal, sc_cal = np.nan, np.nan, np.nan

        # Evaluate pseudo-NLL matching the original log_normal_nolog formulation:
        # element_wise = (y_sc - mu_sc)^2 / (2 * std_sc^2) + std_sc
        y_sc = df['value'].astype(np.float64) / mu_scale
        mu_raw_sc = df['pred_mu'].astype(np.float64) / mu_scale
        mu_cal_sc = (df['pred_mu'].astype(np.float64) + atom_cm) / mu_scale
        std_sc = (df['pred_std'].astype(np.float64) + std_reg) / std_scale

        nll_raw = float(((y_sc - mu_raw_sc) ** 2 / (2.0 * (std_sc ** 2)) + std_sc).mean())
        nll_cal = float(((y_sc - mu_cal_sc) ** 2 / (2.0 * (std_sc ** 2)) + std_sc).mean())

        cm_vals = cm_per_mol.dropna().values
        if str(cm_norm).upper() == 'L1':
            cm_norm_val = float(np.mean(np.abs(cm_vals))) if len(cm_vals) > 0 else 0.0
        else:
            cm_norm_val = float(np.mean(cm_vals ** 2)) if len(cm_vals) > 0 else 0.0

        pen_weighted = float(lambda_penalty) * cm_norm_val
        tot_loss = (nll_cal + pen_weighted) if protocol == 'molcalib' else nll_raw

        if len(cm_vals) > 1:
            q_low, q_high = np.percentile(cm_vals, [1, 99])
            cm_vals_99 = cm_vals[(cm_vals >= q_low) & (cm_vals <= q_high)]
            cm_std_99 = float(np.std(cm_vals_99)) if len(cm_vals_99) > 0 else np.nan
        else:
            cm_std_99 = float(np.std(cm_vals)) if len(cm_vals) > 0 else np.nan

        row[f'{tag}_mae'] = float(np.mean(np.abs(delta_raw)))
        row[f'{tag}_mae_cal'] = float(np.mean(np.abs(delta_cal)))
        row[f'{tag}_nll_raw'] = nll_raw
        row[f'{tag}_nll_cal'] = nll_cal
        row[f'{tag}_pen'] = pen_weighted
        row[f'{tag}_tot_loss'] = tot_loss
        row[f'{tag}_std_mean'] = float(np.mean(std_real))
        row[f'{tag}_cm_mean'] = float(np.mean(cm_vals)) if len(cm_vals) > 0 else np.nan
        row[f'{tag}_cm_std'] = float(np.std(cm_vals)) if len(cm_vals) > 0 else np.nan
        row[f'{tag}_cm_std_99'] = cm_std_99
        row[f'{tag}_nu_raw'] = float(nu_raw)
        row[f'{tag}_loc_raw'] = float(loc_raw)
        row[f'{tag}_sc_raw'] = float(sc_raw)
        row[f'{tag}_nu_cal'] = float(nu_cal)
        row[f'{tag}_loc_cal'] = float(loc_cal)
        row[f'{tag}_sc_cal'] = float(sc_cal)

    row['d_nu_cal'] = row.get('tr_nu_cal', np.nan) - row.get('te_nu_cal', np.nan)
    row['d_nu_raw'] = row.get('tr_nu_raw', np.nan) - row.get('te_nu_raw', np.nan)
    tr_sc_cal = row.get('tr_sc_cal', np.nan)
    te_sc_cal = row.get('te_sc_cal', np.nan)
    row['ratio_sc_cal'] = (te_sc_cal / tr_sc_cal) if (tr_sc_cal and np.isfinite(tr_sc_cal) and tr_sc_cal != 0) else np.nan

    if metrics_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(metrics_csv_path)), exist_ok=True)
        row_df = pd.DataFrame([row])
        write_header = not os.path.exists(metrics_csv_path)
        row_df.to_csv(metrics_csv_path, mode='a', header=write_header, index=False, float_format='%.6f')
        print(_("  [Metrics Ep {epoch_i:04d}] Appended metrics row to: {path} (te_mae_cal={te_mae_cal:.4f}, d_nu_cal={d_nu_cal:.2f}, ratio_sc_cal={ratio_sc_cal:.2f})",
                epoch_i=epoch_i, path=metrics_csv_path,
                te_mae_cal=row.get('te_mae_cal', float('nan')),
                d_nu_cal=row.get('d_nu_cal', float('nan')),
                ratio_sc_cal=row.get('ratio_sc_cal', float('nan'))))

    return row


def plot_from_dataframes(res_train, res_test, std_reg, apply_cm, spect_set,
                         out_file_z, out_file_stats, title_label=""):
    """
    Build and save Z-score and statistics (Sigma, C_m) figures directly from DataFrames
    without requiring model checkpoint files (.state / .model).
    """
    z_train, sig_train, cm_train, n_atoms_tr, n_mols_tr, std_reg_real = process_predictions(
        res_train, std_reg, apply_cm, spect_set
    )
    z_test, sig_test, cm_test, n_atoms_te, n_mols_te, _unused_std_reg = process_predictions(
        res_test, std_reg, apply_cm, spect_set
    )

    if len(z_train) < 2 or len(z_test) < 2:
        print(_("  [Plot] Skipping plot save {path}: insufficient valid data (possible NaN).", path=out_file_z))
        return

    os.makedirs(os.path.dirname(os.path.abspath(out_file_z)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_file_stats)), exist_ok=True)

    fig_z, axes_z = plt.subplots(2, 2, figsize=(14, 10))
    analyze_and_plot_z(z_train, "Train", axes_z[:, 0], n_atoms_tr, n_mols_tr, title_label)
    analyze_and_plot_z(z_test, "Test", axes_z[:, 1], n_atoms_te, n_mols_te, title_label)
    plt.tight_layout()
    fig_z.savefig(out_file_z, dpi=300)
    plt.close(fig_z)
    print(_("Z-score plots saved to: '{path}'", path=out_file_z))

    fig_s, axes_s = plt.subplots(2, 2, figsize=(14, 10))
    plot_stats(sig_train, cm_train, "Train", axes_s[:, 0], std_reg_real, title_label)
    plot_stats(sig_test, cm_test, "Test", axes_s[:, 1], std_reg_real, title_label)
    plt.tight_layout()
    fig_s.savefig(out_file_stats, dpi=300)
    plt.close(fig_s)
    print(_("Statistics plots saved to: '{path}'", path=out_file_stats))


def save_csv_and_plot_epoch(net, dl_train_eval, dl_test_eval, tgt_nucs,
                            checkpoint_prefix, epoch_i, std_reg, protocol,
                            spect_set, plots_dir=None, use_cuda=True,
                            test_res=None, cm_norm="L2", lambda_penalty=0.0):
    """
    Called at the end of every N-th training epoch (e.g. every 10 epochs):
    1) Evaluates train and test predictions in memory (without saving heavy .state/.model files);
    2) Saves lightweight prediction CSV files (*_preds_train.csv and *_preds_test.csv);
    3) Appends a summary row of metrics and Student-t parameters to <checkpoint_prefix>_metrics.csv;
    4) Generates and saves Z-score and Sigma / C_m distribution plots.
    Returns the computed metrics dictionary (metrics_row) for use in EarlyStopping.
    """
    epoch_str = f"{epoch_i:08d}"
    base_with_epoch = f"{checkpoint_prefix}.{epoch_str}"
    csv_train_path = f"{base_with_epoch}_preds_train.csv"
    csv_test_path = f"{base_with_epoch}_preds_test.csv"
    metrics_csv_path = f"{checkpoint_prefix}_metrics.csv"

    net.eval()
    with torch.no_grad():
        train_eval_res = graph_conv_many_nuc_util.run_epoch(
            net, None, None, dl_train_eval,
            pred_only=True, USE_CUDA=use_cuda,
            return_pred=True, desc=f'csv_train (ep {epoch_i})',
            progress_bar=False
        )
        if test_res is not None and 'pred_mu' in test_res:
            test_eval_res = test_res
        else:
            test_eval_res = graph_conv_many_nuc_util.run_epoch(
                net, None, None, dl_test_eval,
                pred_only=True, USE_CUDA=use_cuda,
                return_pred=True, desc=f'csv_test (ep {epoch_i})',
                progress_bar=False
            )

    res_train_df = epoch_res_to_df(train_eval_res, tgt_nucs)
    res_test_df = epoch_res_to_df(test_eval_res, tgt_nucs)

    os.makedirs(os.path.dirname(os.path.abspath(csv_train_path)), exist_ok=True)
    res_train_df.to_csv(csv_train_path, index=False)
    res_test_df.to_csv(csv_test_path, index=False)
    print(_("  [Plot/CSV Ep {epoch_i:04d}] Saved CSVs: {train_path}, {test_path}",
            epoch_i=epoch_i, train_path=csv_train_path, test_path=csv_test_path))

    metrics_row = compute_and_append_epoch_metrics(
        res_train_df, res_test_df,
        epoch_i=epoch_i,
        std_reg=std_reg,
        protocol=protocol,
        spect_set=spect_set,
        cm_norm=cm_norm,
        lambda_penalty=lambda_penalty,
        metrics_csv_path=metrics_csv_path
    )

    if plots_dir is None:
        plots_dir = f"plots_{spect_set}"
    else:
        plots_dir = plots_dir.replace("{NUC}", spect_set)
    os.makedirs(plots_dir, exist_ok=True)

    stage_model_name = os.path.basename(checkpoint_prefix)
    out_file_z = os.path.join(plots_dir, f"{stage_model_name}.zscore.{epoch_str}.png")
    out_file_stats = os.path.join(plots_dir, f"{stage_model_name}.stats.{epoch_str}.png")

    apply_cm = (protocol == "molcalib")
    plot_from_dataframes(
        res_train_df, res_test_df,
        std_reg=std_reg, apply_cm=apply_cm, spect_set=spect_set,
        out_file_z=out_file_z, out_file_stats=out_file_stats,
        title_label=f"{stage_model_name}.{epoch_str}"
    )

    return metrics_row


def find_meta_file(input_path, explicit_meta=""):
    if explicit_meta and os.path.exists(explicit_meta):
        return explicit_meta

    base_name = input_path
    for suffix in ['_preds_train.csv', '_preds_test.csv', '.best.state', '.state', '.model', '.best.model', '.csv']:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    else:
        base_name = os.path.splitext(input_path)[0]

    base_no_epoch = re.sub(r'\.\d{8}$', '', base_name)

    for cand in [base_no_epoch + ".meta", base_no_epoch + ".json", base_name + ".meta", base_name + ".json"]:
        if os.path.exists(cand):
            return cand
    return None


def resolve_csv_pair(input_path):
    """
    Given a path to one of the prediction CSVs or a checkpoint prefix,
    resolve the corresponding (csv_train, csv_test) pair.
    """
    if input_path.endswith('_preds_train.csv'):
        return input_path, input_path[:-len('_preds_train.csv')] + '_preds_test.csv'
    if input_path.endswith('_preds_test.csv'):
        return input_path[:-len('_preds_test.csv')] + '_preds_train.csv', input_path

    base = input_path
    for ext in ['.best.state', '.state', '.model', '.best.model']:
        if base.endswith(ext):
            base = base[:-len(ext)]
            break

    cand_train = base + '_preds_train.csv'
    cand_test = base + '_preds_test.csv'
    return cand_train, cand_test


def load_model_for_eval(state_path, meta):
    """
    Load a model from a .state (state_dict) or .model (serialized module) file.
    """
    obj = torch.load(state_path, map_location='cpu', weights_only=False)
    if isinstance(obj, torch.nn.DataParallel):
        net = obj.module
    elif isinstance(obj, torch.nn.Module):
        net = obj
    elif isinstance(obj, dict):
        state_dict = obj.get('state_dict', obj)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            clean_k = k[len("module."):] if k.startswith("module.") else k
            cleaned_state_dict[clean_k] = v

        net_params = meta['net_params'].copy()
        net = nets.GraphVertModel(**net_params)
        net.load_state_dict(cleaned_state_dict, strict=True)
    else:
        raise ValueError(_("Unknown object type in {path}: {obj_type}", path=state_path, obj_type=type(obj)))

    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()
    return net


def run_standalone_plot(state_or_csv_path, meta_path="", infile_override="",
                        protocol_override="", std_reg_override=None,
                        out_dir_template="plots_{NUC}", force_recalc=False):
    meta_file = find_meta_file(state_or_csv_path, meta_path)
    meta = None
    if meta_file is not None:
        print(_("Loading metadata: {path}", path=meta_file))
        if meta_file.endswith('.json'):
            meta = json.load(open(meta_file, 'r'))
        else:
            meta = pickle.load(open(meta_file, 'rb'))

    if meta is not None:
        tgt_nucs = meta.get('tgt_nucs', ['13C'])
        spect_set = tgt_nucs[0]
        std_reg = meta.get('loss_params', {}).get('std_regularize', 0.01 if spect_set == '1H' else 0.1)
        stage = meta.get('stage', '')
    else:
        spect_set = '1H' if '1H' in state_or_csv_path else '13C'
        tgt_nucs = [spect_set]
        std_reg = 0.01 if spect_set == '1H' else 0.1
        stage = ''

    if std_reg_override is not None:
        std_reg = std_reg_override

    if protocol_override:
        protocol = protocol_override
    elif stage in ['warmup', 'molcalib']:
        protocol = stage
    elif 'molcalib' in state_or_csv_path:
        protocol = 'molcalib'
    else:
        protocol = 'warmup'

    apply_cm = (protocol == 'molcalib')
    apply_str = _("YES") if apply_cm else _("NO")
    print(_("Nucleus: {spect_set} | Protocol: {protocol} (subtract C_m={apply_str}) | std_reg={std_reg}",
            spect_set=spect_set, protocol=protocol, apply_str=apply_str, std_reg=std_reg))

    csv_train_path, csv_test_path = resolve_csv_pair(state_or_csv_path)

    if os.path.exists(csv_train_path) and os.path.exists(csv_test_path) and not force_recalc:
        print(_("Found existing prediction CSV files (model evaluation not required):"))
        print(f"  Train CSV: {csv_train_path}")
        print(f"  Test  CSV: {csv_test_path}")
        res_train = pd.read_csv(csv_train_path)
        res_test = pd.read_csv(csv_test_path)
    else:
        if not os.path.exists(state_or_csv_path):
            raise FileNotFoundError(
                _("Neither prediction CSV files ({csv_path}) nor model weights file ({state_path}) were found.",
                  csv_path=csv_train_path, state_path=state_or_csv_path)
            )
        if meta is None:
            raise FileNotFoundError(
                _("A .meta or .json file is required to compute predictions from model weights ({state_path}).",
                  state_path=state_or_csv_path)
            )

        infile = infile_override if infile_override else meta['infile']
        if not os.path.exists(infile):
            local_infile = os.path.join("graph_conv_many_nuc_pipeline.datasets", os.path.basename(infile))
            if os.path.exists(local_infile):
                infile = local_infile
            else:
                raise FileNotFoundError(_("Dataset file not found: {path}", path=infile))

        print(_("Reading source dataset: {path}", path=infile))
        dataset_hparams = meta['dataset_hparams']
        ds_train, ds_test = graph_conv_many_nuc_util.make_datasets(
            {'filename': infile}, dataset_hparams, min_atoms=5
        )
        dl_train_eval = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=False, num_workers=4)
        dl_test_eval = torch.utils.data.DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=4)

        print(_("Initializing model from checkpoint: {path}", path=state_or_csv_path))
        net = load_model_for_eval(state_or_csv_path, meta)
        use_cuda = torch.cuda.is_available()

        with torch.no_grad():
            print(_("Computing predictions for training set..."))
            train_eval_res = graph_conv_many_nuc_util.run_epoch(
                net, None, None, dl_train_eval,
                pred_only=True, USE_CUDA=use_cuda,
                return_pred=True, desc='predict_train',
                progress_bar=True
            )
            print(_("Computing predictions for test set..."))
            test_eval_res = graph_conv_many_nuc_util.run_epoch(
                net, None, None, dl_test_eval,
                pred_only=True, USE_CUDA=use_cuda,
                return_pred=True, desc='predict_test',
                progress_bar=True
            )

        res_train = epoch_res_to_df(train_eval_res, tgt_nucs)
        res_test = epoch_res_to_df(test_eval_res, tgt_nucs)

        print(_("Saving results to {train_path} and {test_path}...",
                train_path=csv_train_path, test_path=csv_test_path))
        res_train.to_csv(csv_train_path, index=False)
        res_test.to_csv(csv_test_path, index=False)

    out_dir = out_dir_template.replace("{NUC}", spect_set)
    os.makedirs(out_dir, exist_ok=True)

    state_basename = os.path.basename(state_or_csv_path)
    clean_base = state_basename
    for suffix in ['_preds_train.csv', '_preds_test.csv', '.state', '.model', '.csv']:
        if clean_base.endswith(suffix):
            clean_base = clean_base[:-len(suffix)]
            break

    parts = clean_base.split('.')
    if len(parts) >= 2 and (parts[-1].isdigit() or parts[-1] == 'best'):
        epoch_tag = parts[-1]
        prefix = ".".join(parts[:-1])
        out_file_z = os.path.join(out_dir, f"{prefix}.zscore.{epoch_tag}.png")
        out_file_stats = os.path.join(out_dir, f"{prefix}.stats.{epoch_tag}.png")
    else:
        out_file_z = os.path.join(out_dir, f"{clean_base}.zscore.png")
        out_file_stats = os.path.join(out_dir, f"{clean_base}.stats.png")

    plot_from_dataframes(
        res_train, res_test,
        std_reg=std_reg, apply_cm=apply_cm, spect_set=spect_set,
        out_file_z=out_file_z, out_file_stats=out_file_stats,
        title_label=state_basename
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=_("Plot Z-score and Sigma/C_m distributions from a checkpoint or prediction CSV")
    )
    parser.add_argument("--state", type=str, required=True,
                        help=_("Path to model file (.state/.model) OR prediction CSV file (*_preds_train.csv)"))
    parser.add_argument("--meta", type=str, default="",
                        help=_("Path to .meta or .json file (optional, auto-detected if omitted)"))
    parser.add_argument("--infile", type=str, default="",
                        help=_("Path to .pickle dataset file (required only when prediction CSVs are absent)"))
    parser.add_argument("--protocol", type=str, choices=["warmup", "molcalib"], default="",
                        help=_("Explicit protocol override (if omitted, inferred from metadata or filename)"))
    parser.add_argument("--min_error", type=float, default=None,
                        help=_("Explicit std_regularize override (defaults to value in metadata)"))
    parser.add_argument("--out_dir", type=str, default="plots_{NUC}",
                        help=_("Output directory for plots (default: plots_13C or plots_1H)"))
    parser.add_argument("--force_recalc", action="store_true",
                        help=_("Force recomputation of predictions from the model even if CSVs exist"))
    args = parser.parse_args()

    run_standalone_plot(
        state_or_csv_path=args.state,
        meta_path=args.meta,
        infile_override=args.infile,
        protocol_override=args.protocol,
        std_reg_override=args.min_error,
        out_dir_template=args.out_dir,
        force_recalc=args.force_recalc
    )
