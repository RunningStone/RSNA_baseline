import attr 

import torch
import torch.nn as nn

from .data_aug import get_transform,get_three_channels

@attr.s
class PL_Para:

    #----> model and loss
    model_define = None # e.g.  BaselinePreTrain (from ..model.baseline import)
    criterion = nn.BCEWithLogitsLoss
    opt_func = torch.optim.Adam
    sch_func = torch.optim.lr_scheduler.ReduceLROnPlateau
    #---->optimizer and loss
    max_epoch = 5
    patience = 3
    num_workers = 8

    #----> optimizer paras
    init_lr = 0.0005
    opt_paras = {"weight_decay": 0.0,}
    
    #----> scheduler paras
    sch_para = {"patience": 1, # 1 model not improving until lr is decreasing
                "factor": 0.4, # by how much the lr is decreasing
                "verbose": True,
                "mode": "max",}

    #---->data loader
    batch_size = 32           # for train
    shuffle = True
    num_workers = 8

    # define data augmentation
    get_transform:callable=get_transform
    three_channels_fn:callable=get_three_channels
    additional_info = ['laterality', 'view', 'age', 'implant']