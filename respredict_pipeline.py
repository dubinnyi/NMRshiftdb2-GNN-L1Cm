import sys
import pandas as pd
import numpy as np
import pickle
import os
import shutil
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
import plot_z_score
from i18n import _

# === 1. COMMAND-LINE ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description=_("NMR GNN Training Pipeline"))
parser.add_argument("--protocol", "--stage", dest="protocol", type=str, choices=["warmup", "molcalib", "auto"], default="auto", 
                    help=_("Training mode: warmup (mu+sigma), molcalib (+c_m), auto (warmup -> molcalib)"))
parser.add_argument("--weights", "--checkpoint", dest="weights", type=str, default="", 
                    help=_("Path to pre-trained weights file (.state or .model)"))
parser.add_argument("--nucleus", "--nuc", dest="nucleus", type=str, choices=["13C", "1H"], default="13C", 
                    help=_("Target nucleus for chemical shift prediction"))
parser.add_argument("--min_error", type=float, default=None, 
                    help=_("Minimum uncertainty floor (std_regularize) in ppm. Default: 13C=0.1, 1H=0.01"))
parser.add_argument("--max_epochs", type=int, default=10000, 
                    help=_("Maximum number of epochs per stage"))
parser.add_argument("--cv_i", type=int, default=0, 
                    help=_("Dataset cross-validation split index (0, 1, etc.)"))
parser.add_argument("--smoke_test", "--smoke-test", dest="smoke_test", action="store_true",
                    help=_("Quick smoke test run (2 batches, 2 epochs) to verify pipeline integrity"))
cm_group = parser.add_mutually_exclusive_group()
cm_group.add_argument("--L1Cm", action="store_true",
                      help=_("Use L1 penalty |C_m| (in ppm) for molecule calibration in molcalib"))
cm_group.add_argument("--L2Cm", action="store_true",
                      help=_("Use L2 penalty C_m^2 (in ppm^2) for molecule calibration in molcalib (default)"))
parser.add_argument("--lambda_penalty", type=float, default=None,
                    help=_("Explicit LAMBDA_PENALTY weight (if omitted, selected automatically by norm and nucleus)"))
parser.add_argument("--lr", type=float, default=None,
                    help=_("Explicit learning rate. Default: warmup=1e-4, molcalib=3e-5"))
parser.add_argument("--max_d_nu", type=float, default=0.42,
                    help=_("Allowed per-stage increase in Student-t df divergence (df_train - df_test). Default: 0.42 (two-stage cap <= 0.84)"))
args, unknown = parser.parse_known_args()

# Support key=value arguments passed without leading dashes (e.g. checkpoint=... stage=... nuc=...)
for u in unknown:
    u_clean = u.lstrip("-")
    if "=" in u_clean:
        k, v = u_clean.split("=", 1)
        if k in ("protocol", "stage"):
            args.protocol = v
        elif k in ("weights", "checkpoint", "state"):
            args.weights = v
        elif k in ("nucleus", "nuc"):
            args.nucleus = v
        elif k == "cv_i":
            args.cv_i = int(v)
        elif k == "max_epochs":
            args.max_epochs = int(v)
        elif k == "min_error":
            args.min_error = float(v)
        elif k == "lambda_penalty":
            args.lambda_penalty = float(v)
        elif k == "lr":
            args.lr = float(v)
        elif k in ("max_d_nu", "d_nu"):
            args.max_d_nu = float(v)
    elif u not in ("--",):
        print(_("WARNING: Unrecognized command-line argument: '{arg}'", arg=u))

if args.protocol == "molcalib" and not args.weights:
    raise ValueError(
        _("ERROR: Stage 'molcalib' requires a pre-trained checkpoint path via --weights <path> or --checkpoint <path> (or use --protocol auto for the full warmup -> molcalib cycle).")
    )

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

# Override uncertainty regularizer (minimum error floor) if explicitly passed via CLI
if args.min_error is not None:
    STD_REGULARIZE = args.min_error

tgt_max_n = 64 
MIN_ATOMS = 5

# Automatic selection of soft LAMBDA_PENALTY weight (for C_m in ppm),
# anchoring the global reference zero without forcing the GNN to memorize per-spectrum offsets:
# - For 13C (typical shift |C_m| ~ 0.8 ppm, nll_loss ~ 4.0): L1 -> 0.1, L2 -> 0.2
# - For 1H  (typical shift |C_m| ~ 0.12 ppm, nll_loss ~ 4.0): L1 -> 1.0, L2 -> 5.0
if args.lambda_penalty is not None:
    LAMBDA_PENALTY = args.lambda_penalty
else:
    if SPECT_SET == '13C':
        LAMBDA_PENALTY = 0.1 if CM_NORM == "L1" else 0.2
    else:
        LAMBDA_PENALTY = 1.0 if CM_NORM == "L1" else 5.0

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
    Universal loader for checkpoints (.state or .model) created either by the original
    authors' code or by intermediate/best stage saves.
    If a companion .json/.meta file exists, synchronizes architecture hyperparameters (e.g. agg_func).
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
                            print(_("  [Checkpoint Meta] Updating net_params['{key}']: {old_val} -> {new_val}",
                                    key=k, old_val=net_params.get(k), new_val=meta['net_params'][k]))
                            net_params[k] = meta['net_params'][k]
                break
            except Exception as e:
                print(_("  [Checkpoint Meta] Failed to read {path}: {err}", path=cand, err=e))

    net = nets.GraphVertModel(**net_params)
    obj = torch.load(weights_path, map_location='cpu', weights_only=False)

    if isinstance(obj, torch.nn.DataParallel):
        state_dict = obj.module.state_dict()
    elif isinstance(obj, torch.nn.Module):
        state_dict = obj.state_dict()
    elif isinstance(obj, dict):
        state_dict = obj.get('state_dict', obj)
    else:
        raise ValueError(_("Unknown checkpoint format: {obj_type}", obj_type=type(obj)))

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
    # Configure DataLoaders (filtering by minimum atom count)
    # -----------------------------------------------------
    BATCH_SIZE = 16
    dataset_hparams = graph_conv_many_nuc_util.DEFAULT_DATA_HPARAMS
    ds_train, ds_test = graph_conv_many_nuc_util.make_datasets({'filename' : infile}, 
                                                               dataset_hparams, 
                                                               min_atoms=MIN_ATOMS)

    g_feature_n = ds_test[0][1].shape[-1]

    if args.smoke_test:
        print("\n" + "!"*50)
        print(_("!!! SMOKE TEST MODE ENABLED (2 batches, 2 epochs) !!!"))
        print("!"*50 + "\n")
        smoke_n = min(2 * BATCH_SIZE, len(ds_train))
        smoke_test_n = min(2 * BATCH_SIZE, len(ds_test))
        ds_train = torch.utils.data.Subset(ds_train, range(smoke_n))
        ds_test = torch.utils.data.Subset(ds_test, range(smoke_test_n))
                                                               
    print(_("Dataset size (>= {min_atoms} target {spect_set} nuclei): Train={n_train}, Test={n_test}",
            min_atoms=MIN_ATOMS, spect_set=SPECT_SET, n_train=len(ds_train), n_test=len(ds_test)))
    
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)

    # Unshuffled DataLoaders for fast CSV export and distribution plotting every 10 epochs
    dl_train_eval = DataLoader(ds_train, batch_size=64, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)
    dl_test_eval = DataLoader(ds_test, batch_size=64, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)

    net_params = net_params_base.copy()
    net_params['g_feature_n'] = g_feature_n
    net_params['OUT_DIM'] =  len(tgt_nucs)

    # === 2. MODEL INSTANTIATION & PRE-TRAINED WEIGHTS LOADING ===
    if LOAD_WEIGHTS:
        if not os.path.exists(LOAD_WEIGHTS):
            raise FileNotFoundError(_("ERROR: Specified checkpoint file not found: {path}", path=LOAD_WEIGHTS))
        print(_("Loading pre-trained weights from {path}...", path=LOAD_WEIGHTS))
        net = load_pretrained_checkpoint(LOAD_WEIGHTS, net_params)
    else:
        net = nets.GraphVertModel(**net_params)

    net = move(net, USE_CUDA)

    loss_config = {'std_regularize' : STD_REGULARIZE, 
                   'mu_scale' : mu_scale, 
                   'std_scale' : std_scale,
                   'cm_norm' : CM_NORM,
                   'lambda_penalty' : LAMBDA_PENALTY}

    mu_scale_t = move(torch.Tensor(loss_config['mu_scale']), USE_CUDA)
    std_scale_t = move(torch.Tensor(loss_config['std_scale']), USE_CUDA)
    
    if PROTOCOL == "auto":
        protocols_to_run = ["warmup", "molcalib"]
    else:
        protocols_to_run = [PROTOCOL]

    smoke_stage_names = []

    # === 3. SEQUENTIAL STAGE EXECUTION ===
    for stage_protocol in protocols_to_run:
        opt_params = opt_params_base.copy()
        
        stage_max_epochs = MAX_EPOCHS

        print("\n" + "="*60)
        if stage_protocol == 'molcalib':
            opt_params['lr'] = args.lr if args.lr is not None else 3e-5
            print(f"=== RUNNING STAGE: MOLCALIB (FINE-TUNING, norm={CM_NORM}, lambda={LAMBDA_PENALTY}, lr={opt_params['lr']}, max_d_nu={args.max_d_nu}) ===")
            criterion = nets.NormUncertainCalibrationLoss(mu_scale_t, std_scale_t,
                                                          std_regularize=loss_config['std_regularize'],
                                                          lambda_penalty=LAMBDA_PENALTY,
                                                          cm_norm=CM_NORM)

        else: # warmup
            opt_params['lr'] = args.lr if args.lr is not None else 1e-4
            print(f"=== RUNNING STAGE: WARMUP (STANDARD, lr={opt_params['lr']}, max_d_nu={args.max_d_nu}) ===")
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
        smoke_stage_names.append(STAGE_MODEL_NAME)
        checkpoint_base_prefix = os.path.join(CHECKPOINT_DIR, STAGE_MODEL_NAME)
        checkpoint_filename = checkpoint_base_prefix + ".{epoch_i:08d}"
        print("Checkpoint template:", checkpoint_filename)
        print("Metrics CSV file:   ", f"{checkpoint_base_prefix}_metrics.csv")
        checkpoint_func = graph_conv_many_nuc_util.create_checkpoint_func(50, checkpoint_filename)

        # Callback to export prediction CSVs, log epoch metrics, and generate plots every 10 epochs
        def plot_every_10_epochs(epoch_i, net, test_res=None,
                                 _prefix=checkpoint_base_prefix, _stage=stage_protocol):
            if epoch_i % 10 != 0:
                return None
            return plot_z_score.save_csv_and_plot_epoch(
                net=net,
                dl_train_eval=dl_train_eval,
                dl_test_eval=dl_test_eval,
                tgt_nucs=tgt_nucs,
                checkpoint_prefix=_prefix,
                epoch_i=epoch_i,
                std_reg=loss_config['std_regularize'],
                protocol=_stage,
                spect_set=SPECT_SET,
                plots_dir=f"plots_{SPECT_SET}",
                use_cuda=USE_CUDA,
                test_res=test_res,
                cm_norm=CM_NORM,
                lambda_penalty=LAMBDA_PENALTY if _stage == 'molcalib' else 0.0
            )

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

        graph_conv_many_nuc_util.generic_runner(net, optimizer, scheduler, criterion, 
                                                dl_train, dl_test, 
                                                MAX_EPOCHS=stage_max_epochs, 
                                                USE_CUDA=USE_CUDA, writer=writer, 
                                                validate_func= validate_func, 
                                                checkpoint_func= checkpoint_func,
                                                plot_func=plot_every_10_epochs,
                                                stage=stage_protocol,
                                                max_d_nu=args.max_d_nu)
        writer.close()

        best_state_path = os.path.join(CHECKPOINT_DIR, f"{STAGE_MODEL_NAME}.best.state")
        best_model_path = os.path.join(CHECKPOINT_DIR, f"{STAGE_MODEL_NAME}.best.model")

        # If wrapped in DataParallel for multi-GPU, save the underlying module (net.module)
        if isinstance(net, torch.nn.DataParallel):
            torch.save(net.module.state_dict(), best_state_path)
            torch.save(net.module, best_model_path)
        else:
            torch.save(net.state_dict(), best_state_path)
            torch.save(net, best_model_path)
        print(_("Final weights for stage {stage} saved to:\n  {state_path}\n  {model_path}",
                stage=stage_protocol, state_path=best_state_path, model_path=best_model_path))

    if args.smoke_test:
        print(_("\n[SMOKE TEST] Test completed without errors. Automatically cleaning up temporary files..."))
        for s_name in smoke_stage_names:
            for pattern in [
                os.path.join(CHECKPOINT_DIR, f"{s_name}*"),
                os.path.join(f"plots_{SPECT_SET}", f"{s_name}*"),
            ]:
                for fpath in glob(pattern):
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass
            tb_log_dir = os.path.join(TENSORBOARD_DIR, s_name)
            if os.path.exists(tb_log_dir):
                shutil.rmtree(tb_log_dir, ignore_errors=True)
        print(_("[SMOKE TEST] All temporary checkpoints, CSVs, plots, and TensorBoard logs removed.\n"))

    pickle.dump({'net_params' : net_params, 
                 'opt_params' : opt_params, 
                 'loss_params' : loss_config}, 
                open(outfile, 'wb'))

if __name__ == "__main__":
    # Remove stale Ruffus marker files before running the pipeline
    for old_out_file in glob(f"respredict_pipeline_{SPECT_SET}.*.out"):
        try:
            os.remove(old_out_file)
        except OSError:
            pass
            
    pipeline_run([train])

    if args.smoke_test:
        for out_file in glob(f"respredict_pipeline_{SPECT_SET}.*.out"):
            try:
                os.remove(out_file)
            except OSError:
                pass
