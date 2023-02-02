"""
pl paras for baseline model
"""
import torch
import torch.nn as nn

from ..pipeline.data_aug import get_transform,get_three_channels
from ..pipeline.metrics import create_metrics
from .baseline import BaselinePreTrain
from ..pipeline.pl_para import PL_Para

baseline_pl_para = PL_Para()
#----> model and loss
baseline_pl_para.model_define = BaselinePreTrain # e.g.  BaselinePreTrain (from ..model.baseline import)
baseline_pl_para.criterion = nn.BCEWithLogitsLoss
baseline_pl_para.opt_func = torch.optim.Adam
baseline_pl_para.sch_func = torch.optim.lr_scheduler.ExponentialLR
#---->optimizer and loss
baseline_pl_para.max_epoch = 5
baseline_pl_para.patience = 3
#num_workers = 8,

#----> optimizer paras
baseline_pl_para.init_lr = 0.0005
baseline_pl_para.opt_paras = {"weight_decay": 0.0,}
    
#----> scheduler paras
# for ExponentialLR
baseline_pl_para.sch_para = {"gamma": 0.9,}

#---->data loader
baseline_pl_para.batch_size = 32           # for train
baseline_pl_para.shuffle = True
baseline_pl_para.num_workers = 8

# define data augmentation
baseline_pl_para.get_transform=get_transform
baseline_pl_para.post_processing=get_three_channels
baseline_pl_para.additional_info = ['age', 'implant'] # should be number if you want to feed into network

#---->evaluate metric
baseline_pl_para.create_metrics = create_metrics # need define in models part or import create_cFscore 
baseline_pl_para.eval_cFscore_beta = 1.0