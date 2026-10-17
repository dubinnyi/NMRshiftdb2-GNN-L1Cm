import numpy as np
import pickle
import pandas as pd
from tqdm import  tqdm
from rdkit import Chem
import pickle
import os

from glob import glob

import time
import util
from util import move
import nets

from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors as rdMD

import torch
from torch import nn
from tensorboardX import SummaryWriter

from tqdm import  tqdm
import netdataio
import itertools

class Model(object):
    def __init__(self, meta_filename, checkpoint_filename, USE_CUDA=False):

        meta = pickle.load(open(meta_filename, 'rb'))


        self.meta = meta 


        self.USE_CUDA = USE_CUDA

        if self. USE_CUDA:
            net = torch.load(checkpoint_filename)
        else:
            net = torch.load(checkpoint_filename, 
                             map_location=lambda storage, loc: storage)

        self.net = net
        self.net.eval()


    def pred(self, rdmols, values, BATCH_SIZE = 32, debug=False):

        data_feat = self.meta['data_feat']
        MAX_N = self.meta.get('MAX_N', 32)

        USE_CUDA = self.USE_CUDA

        COMBINE_MAT_VECT='row'


        feat_vect_args = self.meta['feat_vect_args']
        feat_mat_args = self.meta['feat_mat_args']
        adj_args = self.meta['adj_args']

        ds = netdataio.MoleculeDatasetMulti(rdmols, values, 
                                            MAX_N, len(self.meta['tgt_nucs']), feat_vect_args, 
                                            feat_mat_args, adj_args,  
                                            combine_mat_vect=COMBINE_MAT_VECT)   
        dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


        allres = []
        alltrue = []
        results_df = []
        m_pos = 0
        for i_batch, (adj, vect_feat, mat_feat, vals, mask) in enumerate(tqdm(dl)):

            if debug:
                print(adj.shape, vect_feat.shape, mat_feat.shape, vals.shape, masks.shape)
            adj_t = move(adj, USE_CUDA)

            args = [adj_t]

            if 'vect' in data_feat:
                vect_feat_t = move(vect_feat, USE_CUDA)
                args.append(vect_feat_t)

            if 'mat' in data_feat:
                mat_feat_t = move(mat_feat, USE_CUDA)
                args.append(mat_feat_t)

            mask_t = move(mask, USE_CUDA)
            vals_t = move(vals, USE_CUDA)

            res = self.net(args)
            std_out = False
            if isinstance(res, tuple):
                std_out = True
                
            if std_out:
                y_est = res[0].detach().cpu().numpy()
                std_est = res[1].detach().cpu().numpy()
            else:
                y_est = res.detach().cpu().numpy()
                std_est = np.zeros_like(y_est)
            for m, v, v_est, std_est in zip(mask.numpy(), vals.numpy(), y_est, std_est):
                for nuc_i, nuc in enumerate(self.meta['tgt_nucs']):
                    m_n = m[:, nuc_i]
                    v_n = v[:, nuc_i]
                    v_est_n = v_est[:, nuc_i]
                    std_est_n = std_est[:, nuc_i]
                    for i in np.argwhere(m_n > 0).flatten():
                        results_df.append({
                            'i_batch' : i_batch, 'atom_idx' : i, 
                            'm_pos' : m_pos,
                            'nuc_i' : nuc_i, 
                            'nuc' : nuc, 
                            'std_out' : std_out, 
                            'value' : v_n[i], 
                            'est' : float(v_est_n[i]), 
                            'std' : float(std_est_n[i]), 
                            
                        })
                m_pos += 1

        results_df = pd.DataFrame(results_df)
        return results_df
