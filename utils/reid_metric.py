# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric
import os
import logging

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes', cfg=None): # <-- Pass cfg
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.cfg = cfg  # <-- Store cfg
        self.logger = logging.getLogger("reid_baseline.inference") # <-- Add logger

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()

        # --- ADD THIS BLOCK TO SAVE DATA ---
        if self.cfg is not None:
            save_path = os.path.join(self.cfg.OUTPUT_DIR, "test_data.npz")
            try:
                np.savez(save_path, 
                         distmat=distmat, 
                         q_pids=q_pids, 
                         g_pids=g_pids, 
                         q_camids=q_camids, 
                         g_camids=g_camids)
                self.logger.info(f"Saved test data (distmat, pids, camids) to {save_path}")
            except Exception as e:
                self.logger.warning(f"Could not save test_data.npz: {e}")
        else:
            self.logger.warning("CFG object not passed to R1_mAP, cannot save test_data.npz")
        # ------------------------------------

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes', cfg=None): # <-- Pass cfg
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.cfg = cfg # <-- Store cfg
        self.logger = logging.getLogger("reid_baseline.inference") # <-- Add logger

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        
        # --- ADD THIS BLOCK TO SAVE DATA ---
        if self.cfg is not None:
            save_path = os.path.join(self.cfg.OUTPUT_DIR, "test_data.npz")
            try:
                np.savez(save_path, 
                         distmat=distmat, 
                         q_pids=q_pids, 
                         g_pids=g_pids, 
                         q_camids=q_camids, 
                         g_camids=g_camids)
                self.logger.info(f"Saved reranking test data to {save_path}")
            except Exception as e:
                self.logger.warning(f"Could not save reranking test_data.npz: {e}")
        else:
            self.logger.warning("CFG object not passed to R1_mAP_reranking, cannot save test_data.npz")
        # ------------------------------------
        
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP
