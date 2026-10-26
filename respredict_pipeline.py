import sys
import pandas as pd
import numpy as np
import pickle
import os
from glob import glob
import json 
import time
import argparse

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
                    help="Режим обучения: warmup, molcalib, или auto (warmup -> molcalib)")
parser.add_argument("--weights", type=str, default="", 
                    help="Путь к файлу предобученных весов (.state)")
parser.add_argument("--nucleus", type=str, choices=["13C", "1H"], default="13C", 
                    help="Целевое ядро для предсказаний")
parser.add_argument("--min_error", type=float, default=None, 
                    help="Минимальная ошибка (std_regularize) в ppm. По умолчанию: 13C=0.8, 1H=0.08")
parser.add_argument("--max_epochs", type=int, default=10000, 
                    help="Максимальное количество эпох на один этап")
parser.add_argument("--cv_i", type=int, default=0, 
                    help="Индекс кросс-валидации датасета (0, 1 и т.д.)")
args, unknown = parser.parse_known_args()

SPECT_SET = args.nucleus
PROTOCOL = args.protocol
LOAD_WEIGHTS = args.weights
MAX_EPOCHS = args.max_epochs

# Настройка регулярзиатора (минимальной ошибки)
if args.min_error is not None:
    STD_REGULARIZE = args.min_error
else:
    STD_REGULARIZE = 0.8 if SPECT_SET == '13C' else 0.03

DATASET_DIR = "graph_conv_many_nuc_pipeline.datasets"
td = lambda x : os.path.join(DATASET_DIR, x)

TENSORBOARD_DIR = f"logs.{SPECT_SET}"
CHECKPOINT_DIR = "checkpoints"+"_"+SPECT_SET
MAT_PROPS= 'aromatic'
DATASET_NAME = 'nmrshiftdb_hconfspcl_nmrshiftdb'

if SPECT_SET == '13C':
    NUC_LOSS_SCALE = {'13C' : 1.0} 
    NUC_STD_SCALE = {'13C' : 1.0}
    STD_REGULARIZE = 0.8
else:
    NUC_LOSS_SCALE = {'1H' : 1.0}
    NUC_STD_SCALE = {'1H' : 1.0} 
    STD_REGULARIZE = 0.03

tgt_max_n = 64 
MIN_ATOMS = 5
LAMBDA_PENALTY = 1.0  # Вес L1-якоря для c_m

EXP_NAME = f"good_{SPECT_SET}"

net_params_base = {'init_noise' : 1e-4, 
                   'resnet' : True, 
                   'int_d' : 2048, 
                   'layer_n' : 10, 
                   'GS' : 4, 
                   'agg_func' : 'mean', 
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
    BATCH_SIZE = 128
    dataset_hparams = graph_conv_many_nuc_util.DEFAULT_DATA_HPARAMS

    ds_train, ds_test = graph_conv_many_nuc_util.make_datasets({'filename' : infile}, 
                                                               dataset_hparams, 
                                                               min_atoms=MIN_ATOMS)
                                                               
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)

    net_params = net_params_base.copy()
    net_params['g_feature_n'] = ds_test[0][1].shape[-1]
    net_params['OUT_DIM'] =  len(tgt_nucs)

    net = nets.GraphVertModel(**net_params)

    # === 2. ЗАГРУЗКА ПРЕДОБУЧЕННЫХ ВЕСОВ ===
    if LOAD_WEIGHTS and os.path.exists(LOAD_WEIGHTS):
        print(f"Loading pre-trained weights from {LOAD_WEIGHTS}...")
        net.load_state_dict(torch.load(LOAD_WEIGHTS))
    elif LOAD_WEIGHTS:
        print(f"WARNING: Checkpoint {LOAD_WEIGHTS} not found!")

    net = move(net, USE_CUDA)

    loss_config = {'std_regularize' : STD_REGULARIZE, 
                   'mu_scale' : mu_scale, 
                   'std_scale' : std_scale}

    mu_scale_t = move(torch.Tensor(loss_config['mu_scale']), USE_CUDA)
    std_scale_t = move(torch.Tensor(loss_config['std_scale']), USE_CUDA)
    
    # Определяем, какие этапы запускать
    protocols_to_run = ["warmup", "molcalib"] if PROTOCOL == "auto" else [PROTOCOL]

    # === 3. ПОСЛЕДОВАТЕЛЬНЫЙ ЗАПУСК ЭТАПОВ ===
    for stage_protocol in protocols_to_run:
        opt_params = opt_params_base.copy()
        
        print("\n" + "="*60)
        if stage_protocol == 'molcalib':
            print("=== RUNNING STAGE: MOLCALIB (FINE-TUNING) ===")
            opt_params['lr'] = 1e-5  # Снижаем LR для дообучения
            criterion = nets.NormUncertainCalibrationLoss(mu_scale_t, 
                                                          std_scale_t,
                                                          std_regularize = loss_config['std_regularize'],
                                                          lambda_penalty = LAMBDA_PENALTY)
        else:
            print("=== RUNNING STAGE: WARMUP (STANDARD) ===")
            opt_params['lr'] = 1e-4
            criterion = nets.NormUncertainLoss(mu_scale_t, 
                                               std_scale_t,
                                               std_regularize = loss_config['std_regularize'])
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
                                                MAX_EPOCHS=MAX_EPOCHS, 
                                                USE_CUDA=USE_CUDA, writer=writer, 
                                                validate_func= validate_func, 
                                                checkpoint_func= checkpoint_func)
                                                
        # После выхода из generic_runner веса сети сохраняются в памяти для следующего этапа.
        best_model_path = os.path.join(CHECKPOINT_DIR, f"{STAGE_MODEL_NAME}.best.state")

        # Если ранее была применена обертка DataParallel для нескольких GPU,
        # нужно сохранять внутренний модуль (net.module), а не саму обертку.
        if isinstance(net, torch.nn.DataParallel):
            torch.save(net.module.state_dict(), best_model_path)
        else:
            torch.save(net.state_dict(), best_model_path)

        print(f"Финальные веса этапа {stage_protocol} сохранены в: {best_model_path}")
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
