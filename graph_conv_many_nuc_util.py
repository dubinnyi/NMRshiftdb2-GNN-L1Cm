import torch

import netdataio
import pickle
import copy

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
import pandas as pd

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

def dict_combine(d1, d2):
    d1 = copy.deepcopy(d1)
    d1.update(d2)
    return d1


def make_datasets(exp_config, hparams, train_sample=0, min_atoms=0):
    d = pickle.load(open(exp_config['filename'], 'rb'))
    train_df = d['train_df']
    test_df = d['test_df'] 
    
    if min_atoms > 0:
        # Считаем именно количество реальных целевых ядер (1H) в словаре спектра
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
                                            hparams['adj_args'])        
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

def run_epoch(net, optimizer, criterion, dl, 
               pred_only = False, USE_CUDA=True,
               return_pred = False, desc="train", 
              print_shapes=False, progress_bar=True,
              scaler=None):
    t1_total= time.time()

    accum_pred = []
    running_loss = 0.0
    running_nll = 0.0
    running_penalty = 0.0
    total_points = 0
    total_compute_time = 0.0
    
    if progress_bar:
        iterator = tqdm(enumerate(dl), total=len(dl), desc=desc)
    else:
        iterator = enumerate(dl)

    for i_batch, (adj, vect_feat, mat_feat, vals, mask) in iterator:
        t1 = time.time()
        if print_shapes:
            print("adj.shape=", adj.shape, "vect_feat.shape=", vect_feat.shape, 
                  "mat_feat.shape=", mat_feat.shape, "vals.shape=", vals.shape, "mask.shape=", mask.shape)

        if not pred_only:
            optimizer.zero_grad()

        adj_t = move(adj, USE_CUDA)
        vect_feat_t = move(vect_feat, USE_CUDA)
        mat_feat_t = None
        mask_t = move(mask, USE_CUDA)
        vals_t = move(vals, USE_CUDA)

        # === ВКЛЮЧАЕМ AMP (FP16) ДЛЯ ФОРВАРДА И LOSS ===
        with torch.amp.autocast('cuda', enabled=USE_CUDA):
            res = net((adj_t, vect_feat_t))

            if return_pred:
                accum_pred_val = {}
                if isinstance(res, dict):
                    for k, v in res.items():
                        accum_pred_val[k] = res[k].float().cpu().detach().numpy()
                else:
                    accum_pred_val['res'] = res.float().cpu().detach().numpy()
                accum_pred_val['mask'] = mask.float().cpu().detach().numpy()
                accum_pred_val['truth'] = vals.float().cpu().detach().numpy()

                accum_pred.append(accum_pred_val)

            if criterion is None:
                loss = 0.0
                nll = 0.0
                penalty = 0.0
            else:
                loss_out = criterion(res, vals_t, mask_t)
                if isinstance(loss_out, tuple):
                    loss, nll, penalty = loss_out
                else:
                    loss = loss_out
                    nll = loss_out
                    penalty = torch.tensor(0.0, device=loss.device)
        # === КОНЕЦ БЛОКА AUTOCAST ===

        # === ОБРАТНОЕ РАСПРОСТРАНЕНИЕ ЧЕРЕЗ SCALER ===
        if not pred_only:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        obs_points = mask.sum()
        if criterion is not None:
            running_loss += loss.item() * obs_points
            running_nll += nll.item() * obs_points
            running_penalty += penalty.item() * obs_points
        total_points += obs_points

        t2 = time.time()
        total_compute_time += (t2-t1)
        
    t2_total = time.time()
    
    res =  {'timing' : 0.0, 
            'running_loss' : running_loss, 
            'running_nll' : running_nll,
            'running_penalty' : running_penalty,
            'total_points' : total_points, 
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
            # Защита от пустого датасета
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
                   min_te_nu=2.0):

    # === ИНИЦИАЛИЗАЦИЯ GradScaler ===
    scaler = torch.amp.GradScaler('cuda', enabled=USE_CUDA)

    # === ИНИЦИАЛИЗАЦИЯ РАННЕЙ ОСТАНОВКИ (с учетом расхождения df Стьюдента d_nu) ===
    # patience=20 означает 20 проверок подряд по 5 эпох (100 эпох) по целевой метрике;
    # max_d_nu=0.42 задает допустимый прирост разницы степеней свободы Стьюдента (df_train - df_test)
    # на один этап, так что после двух этапов (warmup -> molcalib) суммарная разница не превысит ~0.84 (< 1.0).
    early_stopping = EarlyStopping(
        patience=20,
        min_delta=1e-4,
        stage=stage,
        max_d_nu=max_d_nu,
        df_patience=df_patience,
        min_te_nu=min_te_nu
    )

    for epoch_i in tqdm(range(MAX_EPOCHS)):

        t1_total = time.time()

        net.train()
        train_res = run_epoch(net, optimizer, criterion, dl_train, 
                               pred_only = False, USE_CUDA=USE_CUDA, desc='train',
                               scaler=scaler)

        train_loss = float(train_res['running_loss'] / train_res['total_points'])
        print(f"Epoch {epoch_i:04d} | Train Loss: {train_loss:.4f} | Time: {train_res['time']:.1f}s")

        if not np.isfinite(train_loss):
            print(f"\n[ОШИБКА] Обнаружен NaN/Inf в Train Loss на эпохе {epoch_i}! Прерывание обучения и откат к лучшим весам.")
            break

        if writer is not None:
            writer.add_scalar("train_loss_total", train_res['running_loss']/train_res['total_points'], epoch_i)
            writer.add_scalar("train_loss_nll", train_res['running_nll']/train_res['total_points'], epoch_i)
            writer.add_scalar("train_c_m_sq_penalty", train_res['running_penalty']/train_res['total_points'], epoch_i)

        test_res = None
        val_metrics = {}
        if epoch_i % 5 == 0:
            net.eval()
            with torch.no_grad():
                test_res = run_epoch(net, optimizer, criterion, dl_test, 
                                 pred_only = True, USE_CUDA=USE_CUDA, 
                                 return_pred=True, desc='validate')

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
            test_loss = float(test_res['running_loss'] / test_res['total_points'])
            # На этапе molcalib в качестве целевой метрики качества для выбора лучшего чекпоинта
            # используем калиброванную ошибку паттерна (te_mae_cal / test_pattern_abs_err),
            # так как NLL на тесте искусственно растет при сжатии sigma на тренировочной выборке.
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

        # Проверка флага ранней остановки в конце эпохи
        if early_stopping.early_stop:
            print(f"\n--- ОБУЧЕНИЕ ОСТАНОВЛЕНО ДОСРОЧНО на эпохе {epoch_i} ---")
            print(f"Причина: {early_stopping.stop_reason}")
            break
            
    # === ВОССТАНОВЛЕНИЕ ЛУЧШИХ ВЕСОВ В КОНЦЕ ЦИКЛА ===
    if early_stopping.best_state is not None:
        print(f"Восстановление лучших весов модели (эпоха {early_stopping.best_epoch}, "
              f"{early_stopping.metric_name}: {early_stopping.best_loss:.4f}) перед завершением этапа.")
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
        COMBINE_MAT_VECT='row'

        feat_vect_args = dataset_hparams['feat_vect_args']
        feat_mat_args = dataset_hparams['feat_mat_args']
        adj_args = dataset_hparams['adj_args']

        ds = netdataio.MoleculeDatasetMulti(rdmols, values, 
                                            MAX_N, len(self.meta['tgt_nucs']), feat_vect_args, 
                                            feat_mat_args, adj_args,  
                                            combine_mat_vect=COMBINE_MAT_VECT)   
        dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        results_df = []
        
        res = run_epoch(self.net, None, None, dl, 
                        pred_only = True, USE_CUDA=self.USE_CUDA,
                        return_pred = True, print_shapes=debug, desc='predict',
                        progress_bar=prog_bar)
        
        for rd_mol_i, (rdmol, true_val) in enumerate(zip(rdmols, values)):
            for nuc_i, nuc in enumerate(self.meta['tgt_nucs']):
                true_nuc_spect = true_val[nuc_i]
                for atom_idx, true_shift in true_nuc_spect.items():
                    atom_res = {}
                    for pred_key in ['pred_mu', 'pred_std']:
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
            # 1. Глобальная абсолютная ошибка (MAE)
            delta = (mu[:, :, ni] - truth[:, :, ni])[mask[:, :, ni] > 0].flatten()
            mae_global = np.mean(np.abs(delta))
            
            # 2. Ошибка паттерна (MAE_pattern) через вычисление сдвига для каждой молекулы
            mae_pattern_errors = []
            for m_idx in range(mu.shape[0]):
                m_mask = mask[m_idx, :, ni] > 0
                if np.sum(m_mask) > 0:
                    m_delta = truth[m_idx, m_mask, ni] - mu[m_idx, m_mask, ni]
                    # Идеальный сдвиг, минимизирующий MAE для конкретной молекулы - это медиана ошибок
                    m_shift = np.median(m_delta)
                    m_pattern_delta = m_delta - m_shift
                    mae_pattern_errors.extend(np.abs(m_pattern_delta))
            
            mae_pattern = np.mean(mae_pattern_errors) if len(mae_pattern_errors) > 0 else 0.0
            mae_diff = mae_global - mae_pattern
            
            masked_std = (std[:, :, ni])[mask[:, :, ni] > 0].flatten()
            
            # Логируем новые и старые метрики
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
    Ранняя остановка с контролем:
    1) Расхождения степеней свободы Стьюдента d_nu = df_train - df_test
       (на этапе warmup порог = max_d_nu = 0.42; на этапе molcalib к стартовому d_nu_0
       добавляется бюджет +max_d_nu с верхним пределом 2 * max_d_nu = 0.84 < 1.0);
    2) Минимально допустимого числа степеней свободы на тесте (te_nu >= min_te_nu = 2.0,
       чтобы дисперсия распределения ошибок на тесте оставалась конечной);
    3) Целевой метрики качества (для warmup - Test NLL Loss, для molcalib - калиброванный Test MAE_cal),
       причем лучшие веса best_state обновляются только внутри допустимого коридора d_nu <= limit.
    """
    def __init__(self, patience=20, min_delta=1e-4, stage='warmup',
                 max_d_nu=0.42, df_patience=2, min_te_nu=2.0):
        self.patience = patience
        self.min_delta = min_delta
        self.stage = stage
        self.max_d_nu = max_d_nu
        self.df_patience = df_patience
        self.min_te_nu = min_te_nu

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
            self.counter += 1
            print(f"  [EarlyStop] ВНИМАНИЕ: Val Loss = {val_loss} (NaN/Inf)! Остановка обучения.")
            self.early_stop = True
            self.stop_reason = f"NaN/Inf в валидационном Loss на эпохе {epoch_i}"
            return

        # Выбираем целевую метрику для отслеживания лучшей модели:
        # - на warmup: val_loss (NLL)
        # - на molcalib: te_mae_cal (из metrics_row на кратных 10 эпохах) или val_pattern_mae
        if self.stage == 'molcalib':
            if metrics_row is not None and np.isfinite(metrics_row.get('te_mae_cal', np.nan)):
                target_val = float(metrics_row['te_mae_cal'])
            elif val_pattern_mae is not None and np.isfinite(val_pattern_mae):
                target_val = float(val_pattern_mae)
            else:
                target_val = val_loss
        else:
            target_val = val_loss

        # 1. Проверка расхождения степеней свободы Стьюдента (d_nu = tr_nu - te_nu) каждые 10 эпох.
        # Учитываем только сформировавшееся распределение Стьюдента (tr_nu < 20 и te_nu < 20),
        # чтобы на 0-й эпохе warmup белый шум инициализации (nu ~ 1000) не давал ложного срабатывания.
        if metrics_row is not None and self.max_d_nu is not None and self.max_d_nu > 0:
            d_nu_key = 'd_nu_cal' if self.stage == 'molcalib' else 'd_nu_raw'
            tr_nu_key = 'tr_nu_cal' if self.stage == 'molcalib' else 'tr_nu_raw'
            te_nu_key = 'te_nu_cal' if self.stage == 'molcalib' else 'te_nu_raw'

            d_nu = metrics_row.get(d_nu_key, np.nan)
            tr_nu = metrics_row.get(tr_nu_key, np.nan)
            te_nu = metrics_row.get(te_nu_key, np.nan)

            if np.isfinite(d_nu) and np.isfinite(tr_nu) and np.isfinite(te_nu) and tr_nu < 20.0 and te_nu < 20.0:
                if self.initial_d_nu is None:
                    self.initial_d_nu = max(0.0, float(d_nu))
                    if self.stage == 'molcalib':
                        # На втором этапе (molcalib) прибавляем бюджет max_d_nu к стартовому d_nu_0,
                        # но ограничиваем сверху величиной 2 * max_d_nu (чтобы суммарно было < 1.0 df).
                        # Если же стартовый чекпоинт уже имел высокое расхождение, даем минимальный зазор 0.5 * max_d_nu.
                        two_stage_cap = 2.0 * self.max_d_nu
                        if self.initial_d_nu < two_stage_cap - 0.10:
                            self.d_nu_limit = min(self.initial_d_nu + self.max_d_nu, two_stage_cap)
                        else:
                            self.d_nu_limit = self.initial_d_nu + 0.5 * self.max_d_nu
                    else:
                        self.d_nu_limit = self.max_d_nu
                    print(f"  [EarlyStop DF] Начальное расхождение {d_nu_key} = {d_nu:.2f} "
                          f"(tr_nu={tr_nu:.2f}, te_nu={te_nu:.2f}). Порог остановки этапа '{self.stage}': {self.d_nu_limit:.2f}")

                df_violated = (d_nu > self.d_nu_limit) or (te_nu < self.min_te_nu)
                if df_violated:
                    self.is_overfitting_df = True
                    self.df_counter += 1
                    reason_detail = (
                        f"{d_nu_key}={d_nu:.2f} > {self.d_nu_limit:.2f}"
                        if d_nu > self.d_nu_limit
                        else f"{te_nu_key}={te_nu:.2f} < {self.min_te_nu:.2f}"
                    )
                    print(f"  [EarlyStop DF] Превышен порог переобучения Стьюдента ({reason_detail}, "
                          f"tr_nu={tr_nu:.2f}, te_nu={te_nu:.2f}). "
                          f"Счетчик DF: {self.df_counter}/{self.df_patience} (лучшие веса зафиксированы на эпохе {self.best_epoch}).")
                    if self.df_counter >= self.df_patience:
                        self.early_stop = True
                        self.stop_reason = (
                            f"Расхождение степеней свободы Стьюдента превысило порог ({reason_detail}) "
                            f"{self.df_counter} проверки подряд."
                        )
                        return
                else:
                    self.is_overfitting_df = False
                    self.df_counter = 0
                    print(f"  [EarlyStop DF] Норма Стьюдента: {d_nu_key}={d_nu:.2f} <= {self.d_nu_limit:.2f} "
                          f"(tr_nu={tr_nu:.2f}, te_nu={te_nu:.2f}).")

        # 2. Обновление лучших весов и проверка терпения по целевой метрике
        if self.best_loss is None:
            self.best_loss = target_val
            self.best_epoch = epoch_i
            self.best_state = copy.deepcopy(model.state_dict())
            print(f"  [EarlyStop] Инициализация на эпохе {epoch_i}: {self.metric_name} = {self.best_loss:.4f}")
        elif target_val < self.best_loss - self.min_delta:
            if not self.is_overfitting_df:
                print(f"  [EarlyStop] Улучшение на эпохе {epoch_i}! {self.metric_name} снизился с {self.best_loss:.4f} до {target_val:.4f}")
                self.best_loss = target_val
                self.best_epoch = epoch_i
                self.best_state = copy.deepcopy(model.state_dict())
                self.counter = 0
            else:
                self.counter += 1
                print(f"  [EarlyStop] {self.metric_name} = {target_val:.4f}, но веса НЕ обновлены из-за переобучения по d_nu "
                      f"(сохранена эпоха {self.best_epoch} с {self.metric_name} = {self.best_loss:.4f}).")
        else:
            self.counter += 1
            print(f"  [EarlyStop] Нет улучшений {self.metric_name} ({target_val:.4f}). "
                  f"Счетчик: {self.counter}/{self.patience}. Лучший ({self.best_loss:.4f}) на эпохе {self.best_epoch}")
            if self.counter >= self.patience:
                self.early_stop = True
                self.stop_reason = f"{self.metric_name} не улучшался последние {self.patience * 5} эпох."
