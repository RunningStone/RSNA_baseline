"""
pl paras for baseline model
"""
import torch
import torch.nn as nn

from ..pipeline.data_aug import get_transform,get_three_channels
from ..pipeline.metrics import create_metrics,create_cFscore

from ..pipeline.pl_para import PL_Para

baseline_pl_para = PL_Para(
    #----> model and loss
    model_define = None, # e.g.  BaselinePreTrain (from ..model.baseline import)
    criterion = nn.BCEWithLogitsLoss,
    opt_func = torch.optim.Adam,
    sch_func = torch.optim.lr_scheduler.ExponentialLR,
    #---->optimizer and loss
    max_epoch = 5,
    patience = 3,
    #num_workers = 8,

    #----> optimizer paras
    init_lr = 0.0005,
    opt_paras = {"weight_decay": 0.0,},
    
    #----> scheduler paras
    # for ExponentialLR
    sch_para = {"gamma": 0.9,},

    #---->data loader
    batch_size = 32,           # for train
    shuffle = True,
    num_workers = 8,

    # define data augmentation
    get_transform=get_transform,
    post_processing=get_three_channels,
    additional_info = ['age', 'implant'], # should be number if you want to feed into network

    #---->evaluate metric
    create_metrics = create_cFscore # need define in models part or import create_cFscore 
)