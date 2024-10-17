import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import os

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"
class SSLModel(nn.Module):
    def __init__(self, device, cp_path, out_dim):
        super().__init__()
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([
                                                                                 cp_path])
        self.model = model[0]
        self.model = self.model.to(device)
        self.out_dim = out_dim
        self.freeze = False

    def extract_feat(self, input_data):
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb

    def forward(self, input_data):
        return self.extract_feat(input_data)

    def frozen(self):
        logging.info("Freezing the model")
        for param in self.model.parameters():
            param.requires_grad = False
        self.freeze = True

    def unfrozen(self):
        logging.info("Unfreezing the model")
        for param in self.model.parameters():
            param.requires_grad = True
        self.freeze = False
        
class DropoutForMC(nn.Module):
    """Dropout layer for Bayesian model
    THe difference is that we do dropout even in eval stage
    """
    def __init__(self, p, dropout_flag=True):
        super(DropoutForMC, self).__init__()
        self.p = p
        self.flag = dropout_flag
        return
        
    def forward(self, x):
        return torch.nn.functional.dropout(x, self.p, training=self.flag)

class BackEnd(nn.Module):
    """Back End Wrapper
    """
    def __init__(self, input_dim, out_dim, num_classes, 
                 dropout_rate, dropout_flag=True):
        super(BackEnd, self).__init__()

        self.in_dim = input_dim
        self.out_dim = out_dim
        self.num_class = num_classes
        self.m_mcdp_rate = dropout_rate
        self.m_mcdp_flag = dropout_flag
        self.m_frame_level = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.LeakyReLU(),
            torch.nn.Dropout(self.m_mcdp_rate),            
            nn.Linear(self.in_dim, self.in_dim),
            nn.LeakyReLU(),
            torch.nn.Dropout(self.m_mcdp_rate),            
            nn.Linear(self.in_dim, self.out_dim),
            nn.LeakyReLU(),
            torch.nn.Dropout(self.m_mcdp_rate)
        )
        self.m_utt_level = nn.Linear(self.out_dim, self.num_class)
        
        return

    def forward(self, feat):
        feat_ = self.m_frame_level(feat)
        feat_utt = feat_.mean(1)
        
        logits = self.m_utt_level(feat_utt)
        return logits, feat_utt

class Model(nn.Module):
    def __init__(self, ssl_cpkt_path, device, out_dim=1024):
        super().__init__()
        self.device = device

        self.ssl_model = SSLModel(self.device, ssl_cpkt_path, out_dim)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.selu = nn.SELU(inplace=True)
        
        self.loss_CE = nn.CrossEntropyLoss()
        self.backend = BackEnd(128, 128, 2, 0.5, False)
        
        self.sim_metric_seq = lambda mat1, mat2: torch.bmm(
            mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)

        
    def _forward(self, x):


        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)
        feats = x
        x = nn.ReLU()(x)
    
        output, emb = self.backend(x)
        output = F.log_softmax(output, dim=1)

        return output
    
    def forward(self, x_big):
            
        return self._forward(x_big)