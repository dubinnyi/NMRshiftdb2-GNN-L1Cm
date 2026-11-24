import torch

import netdataio
import pickle
import copy

from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from i18n import _


class ModelTrial():
    def __init__(self, training_loader, 
                 validation_loader, hparams):
        self.learning_rate = hparams['learning_rate']
        self.my_batch_size = hparams['batch_size']

    def batch_size(self):
        return self.my_batch_size
    
    def build_model(self, hparams):
        net = nets.GraphVertExtraLinModel( g_feature_n,[INT_D] * LAYER_N, resnet=USE_RESNET, 
                                           noise=INIT_NOISE, agg_func=AGG_FUNC, GS=GS, 
                                           OUT_DIM = len(tgt_nucs))
        
    def optimizer(self, model):
        return torch.optim.Adam(model.parameters(),
                                lr=self.learning_rate)
    def loss(self):
        pass
        
    def training_metrics(self):
        return {
            'classification_Error' : function, 
            'nll_loss' : torch.nn.functional.nll_loss
        }

    def validation_metrics(self):
        pass

default_atomicno = [1, 6, 7, 8, 9, 15, 16, 17]

default_feat_vect_args = dict(feat_atomicno_onehot=default_atomicno, 
                              feat_pos=False, feat_atomicno=True,
                              feat_valence=True, aromatic=True, hybridization=True, 
                              partial_charge=False, formal_charge=True,  
                              r_covalent=False,
                              total_valence_onehot=True, 
                              r_vanderwals=False, default_valence=True, rings=True)

default_feat_mat_args = dict(feat_distances = False, 
                             feat_r_pow = None)

default_split_weights = [1, 1.5, 2, 3]

default_adj_args = dict(edge_weighted=False, 
                        norm_adj=True, add_identity=True, 
                        split_weights=default_split_weights)


DEFAULT_DATA_HPARAMS = {'feat_vect_args' : default_feat_vect_args, 
                       'feat_mat_args' : default_feat_mat_args, 
                       'adj_args' : default_adj_args}

# Rich feature preset for NMRResPredictV2:
# - Maximum chemical/topological/electronic atom features (MMFF94, Gasteiger, Pauling EN,
#   radii, 3D chirality, conjugation, H-neighbor count, ring sizes 3-8)
# - SE(3)-invariant 3D through-space 1/r^3 local environment descriptors
# - Explicit CH2 proton differentiation features (first/second H order, ring/chiral/asym context)
# - 5 topological adjacency channels (any bond + single/aromatic/double/triple, matching Williams 2023)
# - 6 SE(3)-invariant pairwise 3D spatial kernels (5 Gaussian RBF distance shells + 1/r^3)
full_feat_vect_args = dict(
    feat_atomicno_onehot=default_atomicno,
    feat_pos=False,
    feat_atomicno=False,
    feat_valence=True,
    total_valence_onehot=True,
    aromatic=True,
    hybridization=True,
    partial_charge=True,
    formal_charge=True,
    r_covalent=True,
    r_vanderwals=True,
    default_valence=True,
    rings=True,
    max_ring_size=9,
    mmff_atom_types_onehot=True,
    electronegativity=True,
    chirality=True,
    assign_stereo=True,
    extra_topo=True,
    geom_3d_atom=True,
    ch2_order_feat=True,
)

full_feat_mat_args = dict(
    feat_distances=False,
    feat_r_pow=None,
    spatial_rbf_centers=[1.5, 2.2, 3.0, 4.0, 5.5],
    spatial_rbf_sigma=0.5,
    spatial_inv_r3=True,
)

full_adj_args = dict(
    edge_weighted=False,
    norm_adj=False,
    add_identity=False,
    split_weights=default_split_weights,
    edge_bin=True,
)

FULL_DATA_HPARAMS = {
    'feat_vect_args': full_feat_vect_args,
    'feat_mat_args': full_feat_mat_args,
    'adj_args': full_adj_args,
}

def dict_combine(d1, d2):
    d1 = copy.deepcopy(d1)
    d1.update(d2)
    return d1


def make_datasets(exp_config, hparams, train_sample=0, min_atoms=0, return_idx=False):
    d = pickle.load(open(exp_config['filename'], 'rb'))
    train_df = d['train_df']
    test_df = d['test_df'] 
    
    if min_atoms > 0:
        # Count the number of annotated target nuclei in the spectrum dictionary
        train_df = train_df[train_df.value.map(lambda x: len(x[0]) >= min_atoms)]
        test_df = test_df[test_df.value.map(lambda x: len(x[0]) >= min_atoms)]

    if train_sample > 0:
        print("WARNING, subsampling training data to ", train_sample)
        train_df = train_df.sample(train_sample, random_state=0)

    tgt_nucs = d['tgt_nucs']
    MAX_N = d['MAX_N']
    datasets = {}

    for phase, df in [('train', train_df), ('test', test_df)]:
        ds = netdataio.MoleculeDatasetMulti(df.rdmol.tolist(), 
                                            df.value.tolist(),  
                                            MAX_N, len(tgt_nucs), 
                                            hparams['feat_vect_args'], 
                                            hparams['feat_mat_args'], 
                                            hparams['adj_args'],
                                            return_idx=return_idx)        
        datasets[phase] = ds

    return datasets['train'], datasets['test']


def create_checkpoint_func(every_n, filename_str):
    def checkpoint(epoch_i, net, optimizer):
        if epoch_i % every_n > 0:
            return {}
        checkpoint_filename = filename_str.format(epoch_i = epoch_i)
        t1 = time.time()
        torch.save(net.state_dict(), checkpoint_filename + ".state")
        torch.save(net, checkpoint_filename + ".model")
        t2 = time.time()
        return {'savetime' : (t2-t1)}
    return checkpoint


def has_nonfinite_parameters(net):
    """Return True if any parameter tensor in `net` contains NaN or Inf."""
    for p in net.parameters():
        if not torch.isfinite(p).all():
            return True
    return False


def run_epoch(net, optimizer, criterion, dl, 
               pred_only = False, USE_CUDA=True,
               return_pred = False, desc="train", 
              print_shapes=False, progress_bar=True,
              scaler=None, max_grad_norm=10.0):
    t1_total= time.time()

    accum_pred = []
    running_loss = 0.0
    running_nll = 0.0
    running_penalty = 0.0
    total_points = 0
    skipped_batches = 0
    total_compute_time = 0.0

    # Prefer bfloat16 on Ampere+ GPUs (same dynamic range as float32, up to 3.4e38,
    # preventing float16 overflow > 65504 on large 13C chemical shift residuals).
    amp_dtype = (
        torch.bfloat16
        if (USE_CUDA and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    
    if progress_bar:
        iterator = tqdm(enumerate(dl), total=len(dl), desc=desc)
    else:
        iterator = enumerate(dl)

    net_inner = net.module if isinstance(net, torch.nn.DataParallel) else net
    accepts_spatial_and_idx = getattr(net_inner, 'accepts_spatial_and_idx', False)

    for i_batch, batch in iterator:
        if len(batch) == 6:
            adj, vect_feat, mat_feat, vals, mask, mol_idx = batch
        else:
            adj, vect_feat, mat_feat, vals, mask = batch
            mol_idx = None

        t1 = time.time()
        if print_shapes:
            print("adj.shape=", adj.shape, "vect_feat.shape=", vect_feat.shape, 
                  "mat_feat.shape=", mat_feat.shape, "vals.shape=", vals.shape, "mask.shape=", mask.shape)

        if not pred_only:
            optimizer.zero_grad()

        adj_t = move(adj, USE_CUDA)
        vect_feat_t = move(vect_feat, USE_CUDA)
        mat_feat_t = move(mat_feat, USE_CUDA) if accepts_spatial_and_idx else None
        mask_t = move(mask, USE_CUDA)
        vals_t = move(vals, USE_CUDA)

        with torch.amp.autocast('cuda', enabled=USE_CUDA, dtype=amp_dtype):
            if accepts_spatial_and_idx:
                res = net((adj_t, vect_feat_t, mat_feat_t, mol_idx))
            else:
                res = net((adj_t, vect_feat_t))

            if return_pred:
                accum_pred_val = {}
                if isinstance(res, dict):
                    for k, v in res.items():
                        if k in ('mu', 'std', 'var', 'std_alea', 'std_epi', 'res') and torch.is_tensor(v):
                            accum_pred_val[k] = v.float().cpu().detach().numpy()
                else:
                    accum_pred_val['res'] = res.float().cpu().detach().numpy()
                accum_pred_val['mask'] = mask.float().cpu().detach().numpy()
                accum_pred_val['truth'] = vals.float().cpu().detach().numpy()

                accum_pred.append(accum_pred_val)

            if criterion is None:
                loss = None
                nll = None
                penalty = None
            else:
                loss_out = criterion(res, vals_t, mask_t)
                if isinstance(loss_out, tuple):
                    loss, nll, penalty = loss_out
                else:
                    loss = loss_out
                    nll = loss_out
                    penalty = torch.tensor(0.0, device=loss.device)

        if criterion is not None:
            # Guard against a single batch overflow poisoning the entire epoch's running_loss
            # or corrupting model parameters during backward/step.
            if not torch.isfinite(loss):
                skipped_batches += 1
                if not pred_only:
                    optimizer.zero_grad()
                t2 = time.time()
                total_compute_time += (t2 - t1)
                continue

            if not pred_only:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if max_grad_norm is not None and max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm is not None and max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                    optimizer.step()

            obs_points = float(mask.sum().item())
            running_loss += float(loss.item()) * obs_points
            running_nll += float(nll.item()) * obs_points
            running_penalty += float(penalty.item()) * obs_points
            total_points += obs_points
        else:
            total_points += float(mask.sum().item())

        t2 = time.time()
        total_compute_time += (t2-t1)
        
    t2_total = time.time()
    
    res =  {'timing' : 0.0, 
            'running_loss' : running_loss, 
            'running_nll' : running_nll,
            'running_penalty' : running_penalty,
            'total_points' : total_points, 
            'skipped_batches' : skipped_batches,
            'time' : t2_total-t1_total, 
            'compute_time' : total_compute_time, 
            }

    if return_pred:
        if len(accum_pred) > 0:
            keys = accum_pred[0].keys()
            for k in keys:
                accum_pred_v = np.vstack([a[k] for a in accum_pred])
                res[f'pred_{k}'] = accum_pred_v
        else:
            # Guard against empty dataset
            for k in ['mu', 'std', 'mask', 'truth', 'res']: 
                res[f'pred_{k}'] = np.array([])
            
    return res


def generic_runner(net, optimizer, scheduler, criterion, 
                   dl_train, dl_test, 
                   MAX_EPOCHS=1000, 
                   USE_CUDA=True, use_std=False, 
                   writer=None, validate_func = None, 
                   checkpoint_func = None,
                   plot_func = None,
                   stage='warmup',
                   max_d_nu=0.42,
                   df_patience=2,
                   min_te_nu=2.0,
                   min_epochs=None,
                   val_criterion=None):

    eval_criterion = val_criterion if val_criterion is not None else criterion

    # GradScaler is only needed for float16 (bfloat16 has full float32 exponent range)
    use_bf16 = (USE_CUDA and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.0, enabled=(USE_CUDA and not use_bf16))

    if min_epochs is None:
        min_epochs = 50 if stage == 'warmup' else 20

    # Initialize early stopping with Student-t df divergence (d_nu = df_train - df_test) monitoring
    # and a minimum base training period (min_epochs) before stopping rules become active.
    early_stopping = EarlyStopping(
        patience=20,
        min_delta=1e-4,
        stage=stage,
        max_d_nu=max_d_nu,
        df_patience=df_patience,
        min_te_nu=min_te_nu,
        min_epochs=min_epochs
    )

    # Keep a snapshot of the last known finite weights so any transient numerical spike can be recovered
    last_good_state = copy.deepcopy(net.state_dict())

    for epoch_i in tqdm(range(MAX_EPOCHS)):

        t1_total = time.time()

        net.train()
        train_res = run_epoch(net, optimizer, criterion, dl_train, 
                               pred_only = False, USE_CUDA=USE_CUDA, desc='train',
                               scaler=scaler)

        if train_res.get('skipped_batches', 0) > 0:
            print(_("  [Warning] Skipped {skipped} batches with non-finite loss in epoch {epoch_i}.",
                    skipped=train_res['skipped_batches'], epoch_i=epoch_i))

        pts = train_res['total_points']
        train_loss = float(train_res['running_loss'] / pts) if pts > 0 else float('nan')
        train_nll = float(train_res['running_nll'] / pts) if pts > 0 else float('nan')
        if val_criterion is not None:
            print(f"Epoch {epoch_i:04d} | Train beta-NLL: {train_loss:.4f} | Train NLL: {train_nll:.4f} | Time: {train_res['time']:.1f}s")
        else:
            print(f"Epoch {epoch_i:04d} | Train Loss: {train_loss:.4f} | Time: {train_res['time']:.1f}s")

        if not np.isfinite(train_loss) or has_nonfinite_parameters(net):
            # Recover from last known finite weights and reduce learning rate instead of crashing out on early epochs
            rollback_state = early_stopping.best_state if early_stopping.best_state is not None else last_good_state
            net.load_state_dict(rollback_state)
            for pg in optimizer.param_groups:
                pg['lr'] = max(pg['lr'] * 0.5, 1e-6)
            new_lr = optimizer.param_groups[0]['lr']
            print(_("\n[WARNING] Non-finite weights/loss at epoch {epoch_i}! Restoring last valid weights, reducing lr to {lr:.2e}, and continuing training.",
                    epoch_i=epoch_i, lr=new_lr))
            continue

        last_good_state = copy.deepcopy(net.state_dict())

        if writer is not None and pts > 0:
            writer.add_scalar("train_loss_total", train_res['running_loss']/pts, epoch_i)
            writer.add_scalar("train_loss_nll", train_res['running_nll']/pts, epoch_i)
            writer.add_scalar("train_c_m_sq_penalty", train_res['running_penalty']/pts, epoch_i)

        test_res = None
        val_metrics = {}
        if epoch_i % 5 == 0:
            net.eval()
            with torch.no_grad():
                test_res = run_epoch(net, optimizer, eval_criterion, dl_test, 
                                 pred_only = True, USE_CUDA=USE_CUDA, 
                                 return_pred=True, desc='validate')

            test_pts_log = test_res['total_points']
            if writer is not None and test_pts_log > 0:
                writer.add_scalar("val_loss_total", test_res['running_loss']/test_pts_log, epoch_i)
                writer.add_scalar("val_loss_nll", test_res['running_nll']/test_pts_log, epoch_i)

            if validate_func is not None:
                val_metrics = validate_func(test_res)
                if writer is not None:
                    for metric_name, metric_val in val_metrics.items():
                        if np.isfinite(metric_val):
                            writer.add_scalar(metric_name, metric_val, epoch_i)

        if checkpoint_func is not None:
            checkpoint_func(epoch_i = epoch_i, net =net, optimizer=optimizer)

        metrics_row = None
        if plot_func is not None:
            metrics_row = plot_func(epoch_i = epoch_i, net = net, test_res = test_res)

        if epoch_i % 5 == 0 and test_res is not None:
            test_pts = test_res['total_points']
            test_loss = float(test_res['running_loss'] / test_pts) if test_pts > 0 else float('nan')
            # During molcalib fine-tuning, select best_state using calibrated pattern MAE
            # (te_mae_cal / test_pattern_abs_err) because raw test NLL increases when sigma tightens.
            pattern_err_vals = [v for k, v in val_metrics.items() if k.endswith('/test_pattern_abs_err')]
            val_pattern_mae = float(pattern_err_vals[0]) if len(pattern_err_vals) > 0 else None

            early_stopping(
                val_loss=test_loss,
                model=net,
                epoch_i=epoch_i,
                val_pattern_mae=val_pattern_mae,
                metrics_row=metrics_row
            )

        if scheduler is not None:
            scheduler.step()

        if early_stopping.early_stop:
            print(_("\n--- EARLY STOPPING TRIGGERED at epoch {epoch_i} ---", epoch_i=epoch_i))
            print(_("Reason: {reason}", reason=early_stopping.stop_reason))
            break
            
    # Restore best model weights at the end of the stage
    if early_stopping.best_state is not None:
        print(_("Restoring best model weights (epoch {best_epoch}, {metric_name}: {best_loss:.4f}) before finishing stage.",
                best_epoch=early_stopping.best_epoch,
                metric_name=early_stopping.metric_name,
                best_loss=early_stopping.best_loss))
        net.load_state_dict(early_stopping.best_state)


def move(tensor, cuda=False):
    if cuda:
        if isinstance(tensor, nn.Module):
            return tensor.cuda()
        else:
            return tensor.cuda(non_blocking=True)
    else:
        return tensor.cpu()


class PredModel(object):
    def __init__(self, meta_filename, checkpoint_filename, USE_CUDA=False):
        meta = pickle.load(open(meta_filename, 'rb'))
        self.meta = meta 
        self.USE_CUDA = USE_CUDA

        if self. USE_CUDA:
            net = torch.load(checkpoint_filename, weights_only=False)
            if isinstance(net, torch.nn.Module):
                net = net.cuda()
        else:
            net = torch.load(checkpoint_filename, 
                             map_location=lambda storage, loc: storage,
                             weights_only=False)

        self.net = net
        self.net.eval()


    def pred(self, rdmols, values, BATCH_SIZE = 32, 
             debug=False, prog_bar= False):

        dataset_hparams = self.meta['dataset_hparams']
        MAX_N = self.meta.get('max_n', 32)
        USE_CUDA = self.USE_CUDA
        net_inner = self.net.module if isinstance(self.net, torch.nn.DataParallel) else self.net
        is_v2 = getattr(net_inner, 'accepts_spatial_and_idx', False)
        COMBINE_MAT_VECT = None if is_v2 else 'row'

        feat_vect_args = dataset_hparams['feat_vect_args']
        feat_mat_args = dataset_hparams['feat_mat_args']
        adj_args = dataset_hparams['adj_args']

        ds = netdataio.MoleculeDatasetMulti(rdmols, values, 
                                            MAX_N, len(self.meta['tgt_nucs']), feat_vect_args, 
                                            feat_mat_args, adj_args,  
                                            combine_mat_vect=COMBINE_MAT_VECT,
                                            return_idx=is_v2)   
        dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        results_df = []
        
        res = run_epoch(self.net, None, None, dl, 
                        pred_only = True, USE_CUDA=self.USE_CUDA,
                        return_pred = True, print_shapes=debug, desc='predict',
                        progress_bar=prog_bar)
        
        extra_pred_keys = [k for k in ('pred_std_alea', 'pred_std_epi') if k in res]
        for rd_mol_i, (rdmol, true_val) in enumerate(zip(rdmols, values)):
            for nuc_i, nuc in enumerate(self.meta['tgt_nucs']):
                true_nuc_spect = true_val[nuc_i]
                for atom_idx, true_shift in true_nuc_spect.items():
                    atom_res = {}
                    for pred_key in ['pred_mu', 'pred_std'] + extra_pred_keys:
                        atom_res[pred_key] = res[pred_key][rd_mol_i, atom_idx, nuc_i]
                    atom_res['nuc_i'] = nuc_i
                    atom_res['nuc'] = nuc
                    atom_res['atom_idx'] = atom_idx
                    atom_res['m_pos'] = rd_mol_i
                    atom_res['value'] = true_shift
                    
                    results_df.append(atom_res)
                        
        results_df = pd.DataFrame(results_df)
        return results_df
    
def rand_dict(d):
    p = {}
    for k, v in d.items():
        if isinstance(v, list):
            p[k] = v[np.random.randint(len(v))]
        else:
            p[k] = v
    return p

def create_validate_func(tgt_nucs):
    def val_func(res): 
        val = res['pred_res']
        mask = res['pred_mask']
        truth = res['pred_truth']
        res = {}
        for ni, n in enumerate(tgt_nucs):
            delta = (val[:, :, ni] - truth[:, :, ni])[mask[:, :, ni] > 0].flatten()
            res[f"{n}/test_std_err"] = np.std(delta)
            res[f"{n}/test_max_error"] = np.max(np.abs(delta))
            res[f"{n}/test_mean_abs_err"] = np.mean(np.abs(delta))
            res[f"{n}/test_abs_err_90"] = np.percentile(np.abs(delta), 90)
        return res
    return val_func

def create_uncertain_validate_func(tgt_nucs):
    def val_func(res): 
        mu = res['pred_mu']
        std = res['pred_std'] 
        mask = res['pred_mask']
        truth = res['pred_truth']
        val_res = {}
        for ni, n in enumerate(tgt_nucs):
            # 1. Global mean absolute error (MAE)
            delta = (mu[:, :, ni] - truth[:, :, ni])[mask[:, :, ni] > 0].flatten()
            mae_global = np.mean(np.abs(delta))
            
            # 2. Per-molecule pattern error (MAE_pattern) after subtracting each molecule's median shift
            mae_pattern_errors = []
            for m_idx in range(mu.shape[0]):
                m_mask = mask[m_idx, :, ni] > 0
                if np.sum(m_mask) > 0:
                    m_delta = truth[m_idx, m_mask, ni] - mu[m_idx, m_mask, ni]
                    # Optimal constant shift minimizing L1 error for a molecule is the median residual
                    m_shift = np.median(m_delta)
                    m_pattern_delta = m_delta - m_shift
                    mae_pattern_errors.extend(np.abs(m_pattern_delta))
            
            mae_pattern = np.mean(mae_pattern_errors) if len(mae_pattern_errors) > 0 else 0.0
            mae_diff = mae_global - mae_pattern
            
            masked_std = (std[:, :, ni])[mask[:, :, ni] > 0].flatten()
            
            val_res[f"{n}/test_mean_abs_err"] = mae_global
            val_res[f"{n}/test_pattern_abs_err"] = mae_pattern
            val_res[f"{n}/test_err_diff"] = mae_diff
            
            val_res[f"{n}/test_std_err"] = np.std(delta)
            val_res[f"{n}/test_max_error"] = np.max(np.abs(delta))
            val_res[f"{n}/test_abs_err_90"] = np.percentile(np.abs(delta), 90)
            
            val_res[f"{n}/std/mean"] = np.mean(masked_std)
            val_res[f"{n}/std/min"] = np.min(masked_std)
            val_res[f"{n}/std/max"] = np.max(masked_std)

        return val_res
    return val_func


class EarlyStopping:
    """
    Early stopping with automatic restoration of the best model weights, monitoring:
    1) Base training guard (min_epochs and scale <= 3.0): early stopping counters are disarmed
       until the model completes initial convergence;
    2) Student-t degrees-of-freedom divergence d_nu = df_train - df_test
       (for warmup, threshold = max_d_nu = 0.42; for molcalib, an additional +max_d_nu budget
       is added on top of initial d_nu_0, capped at 2 * max_d_nu = 0.84 < 1.0);
    3) Minimum allowed test degrees of freedom (te_nu >= min_te_nu = 2.0, ensuring finite variance
       of the test z-score distribution);
    4) Stage-appropriate quality metric (Test NLL Loss for warmup, calibrated Test MAE_cal for molcalib),
       updating best_state only while d_nu remains within the valid threshold.
    """
    def __init__(self, patience=20, min_delta=1e-4, stage='warmup',
                 max_d_nu=0.42, df_patience=2, min_te_nu=2.0, min_epochs=0):
        self.patience = patience
        self.min_delta = min_delta
        self.stage = stage
        self.max_d_nu = max_d_nu
        self.df_patience = df_patience
        self.min_te_nu = min_te_nu
        self.min_epochs = min_epochs

        self.counter = 0
        self.df_counter = 0
        self.best_loss = None
        self.best_epoch = 0
        self.early_stop = False
        self.stop_reason = ""
        self.best_state = None
        self.initial_d_nu = None
        self.d_nu_limit = max_d_nu if stage == 'warmup' else (2.0 * max_d_nu)
        self.is_overfitting_df = False
        self.metric_name = "Test MAE_cal" if stage == 'molcalib' else "Test Loss"

    def __call__(self, val_loss, model, epoch_i=0, val_pattern_mae=None, metrics_row=None):
        if not np.isfinite(val_loss):
            print(_("  [EarlyStop] WARNING: Val Loss = {val_loss} (NaN/Inf) at epoch {epoch_i}! Ignoring invalid validation step.",
                    val_loss=val_loss, epoch_i=epoch_i))
            return

        # Select target quality metric for best checkpoint tracking:
        # - warmup: val_loss (NLL)
        # - molcalib: te_mae_cal (from metrics_row on 10-epoch intervals) or val_pattern_mae
        if self.stage == 'molcalib':
            if metrics_row is not None and np.isfinite(metrics_row.get('te_mae_cal', np.nan)):
                target_val = float(metrics_row['te_mae_cal'])
            elif val_pattern_mae is not None and np.isfinite(val_pattern_mae):
                target_val = float(val_pattern_mae)
            else:
                target_val = val_loss
        else:
            target_val = val_loss

        # Check whether the model is still in its initial base convergence phase
        sc_key = 'tr_sc_cal' if self.stage == 'molcalib' else 'tr_sc_raw'
        tr_sc = metrics_row.get(sc_key, 1.0) if metrics_row is not None else 1.0
        in_base_training = (epoch_i < self.min_epochs) or (np.isfinite(tr_sc) and tr_sc > 3.0)

        # 1. Check Student-t degrees-of-freedom divergence (d_nu = tr_nu - te_nu) every 10 epochs
        # once base training has converged (epoch_i >= min_epochs, scale <= 3.0, and nu < 20).
        if metrics_row is not None and self.max_d_nu is not None and self.max_d_nu > 0:
            d_nu_key = 'd_nu_cal' if self.stage == 'molcalib' else 'd_nu_raw'
            tr_nu_key = 'tr_nu_cal' if self.stage == 'molcalib' else 'tr_nu_raw'
            te_nu_key = 'te_nu_cal' if self.stage == 'molcalib' else 'te_nu_raw'

            d_nu = metrics_row.get(d_nu_key, np.nan)
            tr_nu = metrics_row.get(tr_nu_key, np.nan)
            te_nu = metrics_row.get(te_nu_key, np.nan)

            if np.isfinite(d_nu) and np.isfinite(tr_nu) and np.isfinite(te_nu) and tr_nu < 20.0 and te_nu < 20.0:
                # Record baseline initial_d_nu for molcalib once the scale is calibrated (tr_sc <= 3.0)
                if self.initial_d_nu is None and (not np.isfinite(tr_sc) or tr_sc <= 3.0):
                    self.initial_d_nu = max(0.0, float(d_nu))
                    if self.stage == 'molcalib':
                        two_stage_cap = 2.0 * self.max_d_nu
                        if self.initial_d_nu < two_stage_cap - 0.10:
                            self.d_nu_limit = min(self.initial_d_nu + self.max_d_nu, two_stage_cap)
                        else:
                            self.d_nu_limit = self.initial_d_nu + 0.5 * self.max_d_nu
                    else:
                        self.d_nu_limit = self.max_d_nu
                    print(_("  [EarlyStop DF] Initial divergence {d_nu_key} = {d_nu:.2f} (tr_nu={tr_nu:.2f}, te_nu={te_nu:.2f}). Stage '{stage}' threshold: {limit:.2f}",
                            d_nu_key=d_nu_key, d_nu=d_nu, tr_nu=tr_nu, te_nu=te_nu,
                            stage=self.stage, limit=self.d_nu_limit))

                if in_base_training:
                    self.is_overfitting_df = False
                    self.df_counter = 0
                    print(_("  [EarlyStop Base] Epoch {epoch_i} < min_epochs ({min_epochs}) or scale={sc:.2f} > 3.0: early stopping disarmed during initial convergence.",
                            epoch_i=epoch_i, min_epochs=self.min_epochs, sc=float(tr_sc) if np.isfinite(tr_sc) else 0.0))
                else:
                    df_violated = (d_nu > self.d_nu_limit) or (te_nu < self.min_te_nu)
                    if df_violated:
                        self.is_overfitting_df = True
                        self.df_counter += 1
                        reason_detail = (
                            f"{d_nu_key}={d_nu:.2f} > {self.d_nu_limit:.2f}"
                            if d_nu > self.d_nu_limit
                            else f"{te_nu_key}={te_nu:.2f} < {self.min_te_nu:.2f}"
                        )
                        print(_("  [EarlyStop DF] Student-t overfitting threshold exceeded ({detail}, tr_nu={tr_nu:.2f}, te_nu={te_nu:.2f}). DF counter: {cnt}/{patience} (best weights locked at epoch {best_epoch}).",
                                detail=reason_detail, tr_nu=tr_nu, te_nu=te_nu,
                                cnt=self.df_counter, patience=self.df_patience, best_epoch=self.best_epoch))
                        if self.df_counter >= self.df_patience:
                            self.early_stop = True
                            self.stop_reason = _(
                                "Student-t degrees-of-freedom divergence exceeded threshold ({detail}) for {cnt} consecutive checks.",
                                detail=reason_detail, cnt=self.df_counter
                            )
                            return
                    else:
                        self.is_overfitting_df = False
                        self.df_counter = 0
                        print(_("  [EarlyStop DF] Student-t within bounds: {d_nu_key}={d_nu:.2f} <= {limit:.2f} (tr_nu={tr_nu:.2f}, te_nu={te_nu:.2f}).",
                                d_nu_key=d_nu_key, d_nu=d_nu, limit=self.d_nu_limit, tr_nu=tr_nu, te_nu=te_nu))

        # 2. Update best model weights and evaluate patience on the target metric
        if self.best_loss is None:
            self.best_loss = target_val
            self.best_epoch = epoch_i
            self.best_state = copy.deepcopy(model.state_dict())
            print(_("  [EarlyStop] Initialized at epoch {epoch_i}: {metric_name} = {best_loss:.4f}",
                    epoch_i=epoch_i, metric_name=self.metric_name, best_loss=self.best_loss))
        elif target_val < self.best_loss - self.min_delta:
            if not self.is_overfitting_df:
                print(_("  [EarlyStop] Improved at epoch {epoch_i}! {metric_name} decreased from {old_loss:.4f} to {new_loss:.4f}",
                        epoch_i=epoch_i, metric_name=self.metric_name, old_loss=self.best_loss, new_loss=target_val))
                self.best_loss = target_val
                self.best_epoch = epoch_i
                self.best_state = copy.deepcopy(model.state_dict())
                self.counter = 0
            else:
                self.counter += 1
                print(_("  [EarlyStop] {metric_name} = {target_val:.4f}, but weights NOT updated due to d_nu overfitting (kept epoch {best_epoch} with {metric_name} = {best_loss:.4f}).",
                        metric_name=self.metric_name, target_val=target_val, best_epoch=self.best_epoch, best_loss=self.best_loss))
        else:
            if in_base_training:
                self.counter = 0
            else:
                self.counter += 1
                print(_("  [EarlyStop] No improvement in {metric_name} ({target_val:.4f}). Counter: {cnt}/{patience}. Best ({best_loss:.4f}) at epoch {best_epoch}",
                        metric_name=self.metric_name, target_val=target_val, cnt=self.counter, patience=self.patience,
                        best_loss=self.best_loss, best_epoch=self.best_epoch))
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.stop_reason = _(
                        "{metric_name} did not improve for the last {epochs} epochs.",
                        metric_name=self.metric_name, epochs=self.patience * 5
                    )
