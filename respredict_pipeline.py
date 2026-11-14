import sys
import pandas as pd
import numpy as np
import pickle
import os
from glob import glob
import json 
import time
import argparse
import re

from ruffus import * 
from tqdm import  tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors as rdMD

import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import util
import nets
from netdataio import * 
import netdataio
import itertools
import graph_conv_many_nuc_util
from graph_conv_many_nuc_util import move

# === 1. ПАРСИНГ АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ ===
parser = argparse.ArgumentParser(description="NMR GNN Training Pipeline")
parser.add_argument("--protocol", type=str, choices=["warmup", "molcalib", "auto"], default="auto", 
                    help="Режим: warmup (mu+sigma), molcalib (+c_m), auto (warmup -> molcalib)")
parser.add_argument("--weights", type=str, default="", 
                    help="Путь к файлу предобученных весов (.state или .model)")
parser.add_argument("--nucleus", type=str, choices=["13C", "1H"], default="13C", 
                    help="Целевое ядро для предсказаний")
parser.add_argument("--min_error", type=float, default=None, 
                    help="Минимальная ошибка (std_regularize) в ppm. По умолчанию: 13C=0.1, 1H=0.01")
parser.add_argument("--max_epochs", type=int, default=10000, 
                    help="Максимальное количество эпох на один этап")
parser.add_argument("--cv_i", type=int, default=0, 
                    help="Индекс кросс-валидации датасета (0, 1 и т.д.)")
parser.add_argument("--smoke_test", "--smoke-test", dest="smoke_test", action="store_true",
                    help="Быстрый прогон (2 батча, 2 эпохи) для проверки пайплайна на ошибки")
cm_group = parser.add_mutually_exclusive_group()
cm_group.add_argument("--L1Cm", action="store_true",
                      help="Использовать L1-штраф |C_m| (в ppm) для калибровки в molcalib")
cm_group.add_argument("--L2Cm", action="store_true",
                      help="Использовать L2-штраф C_m^2 (в ppm^2) для калибровки в molcalib (по умолчанию)")
parser.add_argument("--lambda_penalty", type=float, default=None,
                    help="Явный вес штрафа LAMBDA_PENALTY (если не указан, выбирается автоматически под норму и ядро)")
args, unknown = parser.parse_known_args()

SPECT_SET = args.nucleus
PROTOCOL = args.protocol
LOAD_WEIGHTS = args.weights
MAX_EPOCHS = args.max_epochs
CM_NORM = "L1" if args.L1Cm else "L2"

DATASET_DIR = "graph_conv_many_nuc_pipeline.datasets"
td = lambda x : os.path.join(DATASET_DIR, x)

TENSORBOARD_DIR = f"logs.{SPECT_SET}"
CHECKPOINT_DIR = "checkpoints"+"_"+SPECT_SET
MAT_PROPS= 'aromatic'
DATASET_NAME = 'nmrshiftdb_hconfspcl_nmrshiftdb'

if SPECT_SET == '13C':
    NUC_LOSS_SCALE = {'13C' : 1.0} 
    NUC_STD_SCALE = {'13C' : 1.0}
    STD_REGULARIZE = 0.1
else:
    NUC_LOSS_SCALE = {'1H' : 1.0/20.0}
    NUC_STD_SCALE = {'1H' : 10.0} 
    STD_REGULARIZE = 0.01

# Настройка регуляризатора (минимальной ошибки), если явно передан через CLI
if args.min_error is not None:
    STD_REGULARIZE = args.min_error

tgt_max_n = 64 
MIN_ATOMS = 5

# Автоматический выбор сбалансированного веса штрафа LAMBDA_PENALTY (для C_m в ppm),
# если он не задан явно через --lambda_penalty:
# - Для 13C (характерный сдвиг |C_m| ~ 0.8 ppm, nll_loss ~ 4.4): L1 -> 0.5, L2 -> 0.5
# - Для 1H  (характерный сдвиг |C_m| ~ 0.14 ppm, nll_loss ~ 31.0): L1 -> 20.0, L2 -> 100.0
if args.lambda_penalty is not None:
    LAMBDA_PENALTY = args.lambda_penalty
else:
    if SPECT_SET == '13C':
        LAMBDA_PENALTY = 0.5 if CM_NORM == "L1" else 0.5
    else:
        LAMBDA_PENALTY = 20.0 if CM_NORM == "L1" else 100.0

EXP_NAME = f"good_{SPECT_SET}"

net_params_base = {'init_noise' : 1e-2, 
                   'resnet' : True, 
                   'int_d' : 2048, 
                   'layer_n' : 10, 
                   'GS' : 4, 
                   'agg_func' : 'goodmax', 
                   'force_lin_init' : True, 
                   'g_feature_n' : -1, 
                   'resnet_out' : True, 
                   'out_std' : True, 
                   'graph_dropout' : 0.0, 
                   'resnet_d': 128, 
                   'OUT_DIM' : 1, 
}

opt_params_base = {'amsgrad' : False, 
                   'lr' : 1e-4,
                   'weight_decay' : 1e-6, 
                   'scheduler_gamma' : 0.95, 
                   'eps' : 1e-8, 
                   'scheduler_step_size' : 10}


def load_pretrained_checkpoint(weights_path, net_params):
    """
    Универсальная загрузка чекпоинта (.state или .model), созданного
    как оригинальным кодом авторов, так и промежуточными или best-сохранениями.
    При наличии рядом .json/.meta файла синхронизирует архитектурные параметры (например, agg_func).
    """
    base = weights_path
    for ext in ['.best.state', '.best.model', '.state', '.model']:
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    base_no_epoch = re.sub(r'\.\d{8}$', '', base)

    for cand in [base_no_epoch + ".json", base + ".json", base_no_epoch + ".meta", base + ".meta"]:
        if os.path.exists(cand):
            try:
                meta = json.load(open(cand, 'r')) if cand.endswith('.json') else pickle.load(open(cand, 'rb'))
                if 'net_params' in meta:
                    for k in ['int_d', 'layer_n', 'GS', 'agg_func', 'resnet', 'resnet_out', 'resnet_d', 'out_std']:
                        if k in meta['net_params'] and net_params.get(k) != meta['net_params'][k]:
                            print(f"  [Checkpoint Meta] Обновление net_params['{k}']: {net_params.get(k)} -> {meta['net_params'][k]}")
                            net_params[k] = meta['net_params'][k]
                break
            except Exception as e:
                print(f"  [Checkpoint Meta] Не удалось прочитать {cand}: {e}")

    net = nets.GraphVertModel(**net_params)
    obj = torch.load(weights_path, map_location='cpu', weights_only=False)

    if isinstance(obj, torch.nn.DataParallel):
        state_dict = obj.module.state_dict()
    elif isinstance(obj, torch.nn.Module):
        state_dict = obj.state_dict()
    elif isinstance(obj, dict):
        state_dict = obj.get('state_dict', obj)
    else:
        raise ValueError(f"Неизвестный формат чекпоинта: {type(obj)}")

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k[len("module."):] if k.startswith("module.") else k
        cleaned_state_dict[clean_k] = v

    net.load_state_dict(cleaned_state_dict, strict=True)
    return net


def params():
    CV_I = args.cv_i
    infile = td('graph_conv_many_nuc_pipeline.data.{}.{}.{}.{:d}.{:d}.mol_dict.pickle'.format(SPECT_SET, DATASET_NAME, 
                                                                                              MAT_PROPS, tgt_max_n, CV_I))
#    seed = 1234                                                                                              
    seed = 4321
    outfile = f"respredict_pipeline_{SPECT_SET}.{seed}.{CV_I}.out"
    yield infile, outfile, seed

@mkdir(CHECKPOINT_DIR)
@files(params)
def train(infile, outfile, seed):
    print("infile:", infile)
    np.random.seed(seed)

    d = pickle.load(open(infile, 'rb'))
    tgt_nucs = d['tgt_nucs']
    MAX_N = d['MAX_N']
    print("TGT_NUCS=", tgt_nucs)
    print("output is", outfile)
    USE_CUDA = True

    mu_scale = []
    std_scale = []
    for tn in tgt_nucs:
        for k, v in NUC_LOSS_SCALE.items():
            if k in tn:
                mu_scale.append(v)
        for k, v in NUC_STD_SCALE.items():
            if k in tn:
                std_scale.append(v)
                
    assert len(mu_scale) == len(tgt_nucs)
    assert len(std_scale) == len(tgt_nucs)
    


    # -----------------------------------------------------
    # Настройка Dataloader (фильтрация по мин. числу атомов)
    # -----------------------------------------------------
    BATCH_SIZE = 16
    dataset_hparams = graph_conv_many_nuc_util.DEFAULT_DATA_HPARAMS
    ds_train, ds_test = graph_conv_many_nuc_util.make_datasets({'filename' : infile}, 
                                                               dataset_hparams, 
                                                               min_atoms=MIN_ATOMS)

    g_feature_n = ds_test[0][1].shape[-1]

    if args.smoke_test:
        print("\n" + "!"*50)
        print("!!! ВКЛЮЧЕН РЕЖИМ SMOKE TEST (2 батча, 2 эпохи) !!!")
        print("!"*50 + "\n")
        smoke_n = min(2 * BATCH_SIZE, len(ds_train))
        smoke_test_n = min(2 * BATCH_SIZE, len(ds_test))
        ds_train = torch.utils.data.Subset(ds_train, range(smoke_n))
        ds_test = torch.utils.data.Subset(ds_test, range(smoke_test_n))
                                                               
    print(f"Размер выборки (>= {MIN_ATOMS} целевых ядер {SPECT_SET}): Train={len(ds_train)}, Test={len(ds_test)}")
    
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)

    net_params = net_params_base.copy()
    net_params['g_feature_n'] = g_feature_n
    net_params['OUT_DIM'] =  len(tgt_nucs)

    # === 2. СОЗДАНИЕ СЕТИ И ЗАГРУЗКА ПРЕДОБУЧЕННЫХ ВЕСОВ ===
    if LOAD_WEIGHTS and os.path.exists(LOAD_WEIGHTS):
        print(f"Loading pre-trained weights from {LOAD_WEIGHTS}...")
        net = load_pretrained_checkpoint(LOAD_WEIGHTS, net_params)
    else:
        if LOAD_WEIGHTS:
            print(f"WARNING: Checkpoint {LOAD_WEIGHTS} not found! Initializing from scratch.")
        net = nets.GraphVertModel(**net_params)

    net = move(net, USE_CUDA)

    loss_config = {'std_regularize' : STD_REGULARIZE, 
                   'mu_scale' : mu_scale, 
                   'std_scale' : std_scale,
                   'cm_norm' : CM_NORM,
                   'lambda_penalty' : LAMBDA_PENALTY}

    mu_scale_t = move(torch.Tensor(loss_config['mu_scale']), USE_CUDA)
    std_scale_t = move(torch.Tensor(loss_config['std_scale']), USE_CUDA)
    
    # Определяем, какие этапы запускать
    if PROTOCOL == "auto":
        protocols_to_run = ["warmup", "molcalib"]
    else:
        protocols_to_run = [PROTOCOL]

    # === 3. ПОСЛЕДОВАТЕЛЬНЫЙ ЗАПУСК ЭТАПОВ ===
    for stage_protocol in protocols_to_run:
        opt_params = opt_params_base.copy()
        
        stage_max_epochs = MAX_EPOCHS

        print("\n" + "="*60)
        if stage_protocol == 'molcalib':
            print(f"=== RUNNING STAGE: MOLCALIB (FINE-TUNING, norm={CM_NORM}, lambda={LAMBDA_PENALTY}) ===")
            opt_params['lr'] = 1e-5
            criterion = nets.NormUncertainCalibrationLoss(mu_scale_t, std_scale_t,
                                                          std_regularize=loss_config['std_regularize'],
                                                          lambda_penalty=LAMBDA_PENALTY,
                                                          cm_norm=CM_NORM)

        else: # warmup
            print("=== RUNNING STAGE: WARMUP (STANDARD) ===")
            opt_params['lr'] = 1e-4
            criterion = nets.NormUncertainLoss(mu_scale_t, std_scale_t,
                                               std_regularize=loss_config['std_regularize'])
            stage_max_epochs = 300

        if args.smoke_test:
            stage_max_epochs = 2
        print("="*60 + "\n")

        validate_func = graph_conv_many_nuc_util.create_uncertain_validate_func(tgt_nucs)

        optimizer = torch.optim.Adam(net.parameters(), lr=opt_params['lr'], 
                                     amsgrad=opt_params['amsgrad'], 
                                     eps=opt_params['eps'], 
                                     weight_decay=opt_params['weight_decay'])
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=opt_params['scheduler_step_size'], 
                                                    gamma=opt_params['scheduler_gamma'])

        STAGE_MODEL_NAME = "{}_{}.{:08d}".format(EXP_NAME, stage_protocol, int(time.time() % 1e8))
        checkpoint_filename = os.path.join(CHECKPOINT_DIR, STAGE_MODEL_NAME + ".{epoch_i:08d}")
        print("Checkpoint template:", checkpoint_filename)
        checkpoint_func = graph_conv_many_nuc_util.create_checkpoint_func(50, checkpoint_filename)
                            
        writer = SummaryWriter("{}/{}".format(TENSORBOARD_DIR, STAGE_MODEL_NAME))

        metadata = {'dataset_hparams' : dataset_hparams, 
                    'net_params' : net_params, 
                    'opt_params' : opt_params, 
                    'infile' : infile, 
                    'tgt_nucs' : tgt_nucs, 
                    'max_n' : MAX_N,
                    'loss_params' : loss_config,
                    'stage': stage_protocol}

        json.dump(metadata, open(os.path.join(CHECKPOINT_DIR, STAGE_MODEL_NAME + ".json"), 'w'), indent=4)
        pickle.dump(metadata, open(os.path.join(CHECKPOINT_DIR, STAGE_MODEL_NAME + ".meta"), 'wb'))

        # Запуск обучения для текущего этапа (встроенная функция EarlyStopping прервет цикл при необходимости)
        graph_conv_many_nuc_util.generic_runner(net, optimizer, scheduler, criterion, 
                                                dl_train, dl_test, 
                                                MAX_EPOCHS=stage_max_epochs, 
                                                USE_CUDA=USE_CUDA, writer=writer, 
                                                validate_func= validate_func, 
                                                checkpoint_func= checkpoint_func)
                                                
        # После выхода из generic_runner веса сети сохраняются в памяти для следующего этапа.
        best_state_path = os.path.join(CHECKPOINT_DIR, f"{STAGE_MODEL_NAME}.best.state")
        best_model_path = os.path.join(CHECKPOINT_DIR, f"{STAGE_MODEL_NAME}.best.model")

        # Если ранее была применена обертка DataParallel для нескольких GPU,
        # нужно сохранять внутренний модуль (net.module), а не саму обертку.
        if isinstance(net, torch.nn.DataParallel):
            torch.save(net.module.state_dict(), best_state_path)
            torch.save(net.module, best_model_path)
        else:
            torch.save(net.state_dict(), best_state_path)
            torch.save(net, best_model_path)
        print(f"Финальные веса этапа {stage_protocol} сохранены :\n  {best_state_path}\n  {best_model_path}")
        # --------------------------

    pickle.dump({'net_params' : net_params, 
                 'opt_params' : opt_params, 
                 'loss_params' : loss_config}, 
                open(outfile, 'wb'))

if __name__ == "__main__":
    import glob
    # Принудительно удаляем старые маркеры выполнения Ruffus перед запуском
    for old_out_file in glob.glob(f"respredict_pipeline_{SPECT_SET}.*.out"):
        try:
            os.remove(old_out_file)
        except OSError:
            pass
            
    pipeline_run([train])
