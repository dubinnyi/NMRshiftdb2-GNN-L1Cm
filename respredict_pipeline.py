import sys
import pandas as pd
import numpy as np
import pickle
import os
import copy
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
parser.add_argument("--arch", type=str, choices=["v2", "legacy"], default="v2",
                    help=_("Model architecture: v2 (lightweight Williams-2023 polynomial GNN + 3D RBF + CH2 features + multi-head beta-NLL) or legacy (original GraphVertModel). Default: v2"))
parser.add_argument("--protocol", "--stage", dest="protocol", type=str, choices=["warmup", "molcalib", "auto"], default=None, 
                    help=_("Training mode: warmup (mu+sigma), molcalib (+c_m), auto (warmup -> molcalib). Default: warmup for beta_nll, auto for nolog"))
parser.add_argument("--loss", "--loss_type", dest="loss_type", type=str, choices=["beta_nll", "nolog"], default="beta_nll",
                    help=_("Uncertainty loss function: beta_nll (Seitzer et al., ICLR 2022) or nolog (legacy Jonas & Kuhn). Default: beta_nll"))
parser.add_argument("--beta", type=float, default=0.5,
                    help=_("Beta exponent in [0, 1] for beta-NLL loss (0=standard NLL, 0.5=default, 1=MSE mean gradient)"))
parser.add_argument("--mixture_n", type=int, default=5,
                    help=_("Number of probabilistic bootstrap readout heads in v2 architecture (default: 5)"))
parser.add_argument("--mixture_obs", type=int, default=4,
                    help=_("Number of heads trained per molecule in v2 bootstrap subsampling (default: 4)"))
parser.add_argument("--int_d", type=int, default=256,
                    help=_("Hidden GNN width for v2 architecture (default: 256, lightweight as in Williams & Jonas 2023)"))
parser.add_argument("--layer_n", type=int, default=8,
                    help=_("Number of GNN message-passing layers for v2 architecture (default: 8)"))
parser.add_argument("--batch_size", type=int, default=None,
                    help=_("Training batch size (default: 32 for v2, 16 for legacy)"))
parser.add_argument("--no_3d", action="store_true",
                    help=_("Disable 3D spatial features in v2 (ablation: pure 2D + rich chemical + CH2 features)"))
parser.add_argument("--init_std", type=float, default=None,
                    help=_("Initial predicted std (in ppm) for beta-NLL softplus variance head. Default: 13C=5.0, 1H=1.0"))
parser.add_argument("--max_std", type=float, default=None,
                    help=_("Maximum predicted std clamp (in ppm) for beta-NLL variance head. Default: 13C=100.0, 1H=10.0"))
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
                    help=_("Explicit learning rate. Default: v2=5e-4, legacy warmup=1e-4, molcalib=3e-5"))
parser.add_argument("--max_d_nu", type=float, default=0.42,
                    help=_("Allowed per-stage increase in Student-t df divergence (df_train - df_test). Default: 0.42 (two-stage cap <= 0.84)"))
parser.add_argument("--min_epochs", type=int, default=None,
                    help=_("Minimum number of base training epochs before early stopping is armed (default: warmup=50, molcalib=20)"))
args, unknown = parser.parse_known_args()

# Support key=value arguments passed without leading dashes (e.g. checkpoint=... stage=... nuc=...)
for u in unknown:
    u_clean = u.lstrip("-")
    if "=" in u_clean:
        k, v = u_clean.split("=", 1)
        if k == "arch":
            args.arch = v
        elif k in ("protocol", "stage"):
            args.protocol = v
        elif k in ("loss", "loss_type"):
            args.loss_type = v
        elif k == "beta":
            args.beta = float(v)
        elif k == "mixture_n":
            args.mixture_n = int(v)
        elif k == "mixture_obs":
            args.mixture_obs = int(v)
        elif k == "int_d":
            args.int_d = int(v)
        elif k == "layer_n":
            args.layer_n = int(v)
        elif k == "batch_size":
            args.batch_size = int(v)
        elif k == "init_std":
            args.init_std = float(v)
        elif k == "max_std":
            args.max_std = float(v)
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
        elif k == "min_epochs":
            args.min_epochs = int(v)
    elif u not in ("--",):
        print(_("WARNING: Unrecognized command-line argument: '{arg}'", arg=u))

ARCH = args.arch
LOSS_TYPE = args.loss_type
BETA = float(args.beta)
if ARCH == "v2" and LOSS_TYPE != "beta_nll":
    raise ValueError(
        _("ERROR: --arch v2 (NMRResPredictV2) requires --loss beta_nll. Use --arch legacy --loss nolog for the original Jonas & Kuhn model.")
    )
if LOSS_TYPE == "beta_nll" and not (0.0 <= BETA <= 1.0):
    raise ValueError(_("ERROR: --beta must be in [0.0, 1.0], got {beta}", beta=BETA))

if args.protocol is None:
    args.protocol = "warmup" if LOSS_TYPE == "beta_nll" else "auto"

if LOSS_TYPE == "beta_nll" and args.protocol != "warmup":
    raise ValueError(
        _("ERROR: --loss beta_nll is currently implemented for stage 'warmup' only. Pass --protocol warmup (or omit --protocol), or use --loss nolog for legacy molcalib/auto.")
    )

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


def load_pretrained_checkpoint(weights_path, net_params, net_name="GraphVertModel"):
    """
    Universal loader for checkpoints (.state or .model) created either by the original
    authors' code, beta-NLL runs, or NMRResPredictV2 runs.
    If a companion .json/.meta file exists, synchronizes architecture hyperparameters.
    """
    base = weights_path
    for ext in ['.best.state', '.best.model', '.state', '.model']:
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    base_no_epoch = re.sub(r'\.\d{8}$', '', base)

    detected_net_name = net_name
    for cand in [base_no_epoch + ".json", base + ".json", base_no_epoch + ".meta", base + ".meta"]:
        if os.path.exists(cand):
            try:
                meta = json.load(open(cand, 'r')) if cand.endswith('.json') else pickle.load(open(cand, 'rb'))
                if 'net_name' in meta:
                    detected_net_name = meta['net_name']
                if 'net_params' in meta:
                    for k in ['int_d', 'layer_n', 'GS', 'agg_func', 'resnet', 'resnet_out', 'resnet_d', 'out_std',
                              'var_mode', 'init_var', 'min_var', 'max_var', 'spatial_dim', 'spatial_hidden',
                              'mixture_n', 'mixture_num_obs_per', 'dropout', 'resnet_depth']:
                        if k in meta['net_params'] and net_params.get(k) != meta['net_params'][k]:
                            print(_("  [Checkpoint Meta] Updating net_params['{key}']: {old_val} -> {new_val}",
                                    key=k, old_val=net_params.get(k), new_val=meta['net_params'][k]))
                            net_params[k] = meta['net_params'][k]
                break
            except Exception as e:
                print(_("  [Checkpoint Meta] Failed to read {path}: {err}", path=cand, err=e))

    if detected_net_name == "NMRResPredictV2":
        net = nets.NMRResPredictV2(**net_params)
    else:
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
    if ARCH == 'v2':
        BATCH_SIZE = args.batch_size if args.batch_size is not None else 32
        dataset_hparams = copy.deepcopy(graph_conv_many_nuc_util.FULL_DATA_HPARAMS)
        if args.no_3d:
            dataset_hparams['feat_vect_args']['geom_3d_atom'] = False
            dataset_hparams['feat_mat_args']['spatial_rbf_centers'] = None
            dataset_hparams['feat_mat_args']['spatial_inv_r3'] = False
        return_idx = True
    else:
        BATCH_SIZE = args.batch_size if args.batch_size is not None else 16
        dataset_hparams = graph_conv_many_nuc_util.DEFAULT_DATA_HPARAMS
        return_idx = False

    ds_train, ds_test = graph_conv_many_nuc_util.make_datasets({'filename' : infile}, 
                                                               dataset_hparams, 
                                                               min_atoms=MIN_ATOMS,
                                                               return_idx=return_idx)

    sample0 = ds_test[0]
    adj_channels = sample0[0].shape[0]
    g_feature_n = sample0[1].shape[-1]
    spatial_dim = 0 if (ARCH != 'v2' or args.no_3d) else sample0[2].shape[-1]

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

    min_std_ppm = float(STD_REGULARIZE)
    init_std_ppm = float(args.init_std) if args.init_std is not None else (5.0 if SPECT_SET == '13C' else 1.0)
    max_std_ppm = float(args.max_std) if args.max_std is not None else (100.0 if SPECT_SET == '13C' else 10.0)

    if ARCH == 'v2':
        net_name = 'NMRResPredictV2'
        net_params = {
            'g_feature_n': g_feature_n,
            'int_d': args.int_d,
            'layer_n': args.layer_n,
            'GS': adj_channels,
            'OUT_DIM': len(tgt_nucs),
            'spatial_dim': spatial_dim,
            'spatial_hidden': 64,
            'dropout': 0.01,
            'resnet_d': 128,
            'resnet_depth': 3,
            'mixture_n': args.mixture_n,
            'mixture_num_obs_per': min(args.mixture_obs, args.mixture_n),
            'min_var': min_std_ppm ** 2,
            'init_var': init_std_ppm ** 2,
            'max_var': max_std_ppm ** 2,
        }
    else:
        net_name = 'GraphVertModel'
        net_params = net_params_base.copy()
        net_params['g_feature_n'] = g_feature_n
        net_params['OUT_DIM'] = len(tgt_nucs)
        if LOSS_TYPE == 'beta_nll':
            net_params['var_mode'] = 'softplus_var'
            net_params['min_var'] = min_std_ppm ** 2
            net_params['init_var'] = init_std_ppm ** 2
            net_params['max_var'] = max_std_ppm ** 2

    # === 2. MODEL INSTANTIATION & PRE-TRAINED WEIGHTS LOADING ===
    if LOAD_WEIGHTS:
        if not os.path.exists(LOAD_WEIGHTS):
            raise FileNotFoundError(_("ERROR: Specified checkpoint file not found: {path}", path=LOAD_WEIGHTS))
        print(_("Loading pre-trained weights from {path}...", path=LOAD_WEIGHTS))
        net = load_pretrained_checkpoint(LOAD_WEIGHTS, net_params, net_name=net_name)
    else:
        if net_name == 'NMRResPredictV2':
            net = nets.NMRResPredictV2(**net_params)
        else:
            net = nets.GraphVertModel(**net_params)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"Model: {net_name} | Atom features: {g_feature_n} | Adj channels: {adj_channels} | "
          f"3D spatial channels: {spatial_dim} | Parameters: {n_params / 1e6:.2f}M")

    net = move(net, USE_CUDA)

    loss_config = {'loss_type' : LOSS_TYPE,
                   'beta' : BETA if LOSS_TYPE == 'beta_nll' else 0.0,
                   'std_regularize' : STD_REGULARIZE, 
                   'mu_scale' : mu_scale, 
                   'std_scale' : std_scale,
                   'cm_norm' : CM_NORM,
                   'lambda_penalty' : LAMBDA_PENALTY}
    if LOSS_TYPE == 'beta_nll':
        loss_config['min_var'] = net_params['min_var']
        loss_config['init_var'] = net_params['init_var']
        loss_config['max_var'] = net_params['max_var']

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
        stage_min_epochs = args.min_epochs if args.min_epochs is not None else (50 if stage_protocol == 'warmup' else 20)
        val_criterion = None

        print("\n" + "="*60)
        if stage_protocol == 'molcalib':
            opt_params['lr'] = args.lr if args.lr is not None else 3e-5
            print(f"=== RUNNING STAGE: MOLCALIB (FINE-TUNING, norm={CM_NORM}, lambda={LAMBDA_PENALTY}, lr={opt_params['lr']}, max_d_nu={args.max_d_nu}, min_epochs={stage_min_epochs}) ===")
            criterion = nets.NormUncertainCalibrationLoss(mu_scale_t, std_scale_t,
                                                          std_regularize=loss_config['std_regularize'],
                                                          lambda_penalty=LAMBDA_PENALTY,
                                                          cm_norm=CM_NORM)

        else: # warmup
            default_warmup_lr = 5e-4 if ARCH == 'v2' else 1e-4
            opt_params['lr'] = args.lr if args.lr is not None else default_warmup_lr
            if LOSS_TYPE == 'beta_nll':
                heads_info = (
                    f", heads={net_params['mixture_n']} (obs={net_params['mixture_num_obs_per']})"
                    if ARCH == 'v2' else ""
                )
                print(
                    f"=== RUNNING STAGE: WARMUP ({net_name}, BETA-NLL, beta={BETA}{heads_info}, "
                    f"min_std={np.sqrt(net_params['min_var']):.4g} ppm, "
                    f"init_std={np.sqrt(net_params['init_var']):.4g} ppm, "
                    f"max_std={np.sqrt(net_params['max_var']):.4g} ppm, "
                    f"lr={opt_params['lr']}, max_d_nu={args.max_d_nu}, min_epochs={stage_min_epochs}) ==="
                )
                criterion = nets.BetaNLLLoss(beta=BETA, min_var=net_params['min_var'])
                # Evaluate validation & early stopping with exact unweighted Gaussian NLL (beta=0.0)
                val_criterion = nets.BetaNLLLoss(beta=0.0, min_var=net_params['min_var'])
            else:
                print(f"=== RUNNING STAGE: WARMUP (STANDARD, lr={opt_params['lr']}, max_d_nu={args.max_d_nu}, min_epochs={stage_min_epochs}) ===")
                criterion = nets.NormUncertainLoss(mu_scale_t, std_scale_t,
                                                   std_regularize=loss_config['std_regularize'])
            stage_max_epochs = min(MAX_EPOCHS, 300)

        if args.smoke_test:
            stage_max_epochs = 2
            stage_min_epochs = 0
        print("="*60 + "\n")

        validate_func = graph_conv_many_nuc_util.create_uncertain_validate_func(tgt_nucs)

        optimizer = torch.optim.Adam(net.parameters(), lr=opt_params['lr'], 
                                     amsgrad=opt_params['amsgrad'], 
                                     eps=opt_params['eps'], 
                                     weight_decay=opt_params['weight_decay'])
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=opt_params['scheduler_step_size'], 
                                                    gamma=opt_params['scheduler_gamma'])

        if LOSS_TYPE == 'beta_nll':
            beta_tag = f"b{str(BETA).replace('.', 'p')}"
            arch_tag = "v2_" if ARCH == 'v2' else ""
            STAGE_MODEL_NAME = "{}_{}_{}betanll_{}.{:08d}".format(EXP_NAME, stage_protocol, arch_tag, beta_tag, int(time.time() % 1e8))
        else:
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
                lambda_penalty=LAMBDA_PENALTY if _stage == 'molcalib' else 0.0,
                loss_type=LOSS_TYPE,
                beta=BETA if LOSS_TYPE == 'beta_nll' else 0.0
            )

        writer = SummaryWriter("{}/{}".format(TENSORBOARD_DIR, STAGE_MODEL_NAME))

        metadata = {'dataset_hparams' : dataset_hparams, 
                    'net_name' : net_name,
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
                                                max_d_nu=args.max_d_nu,
                                                min_epochs=stage_min_epochs,
                                                val_criterion=val_criterion)
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
