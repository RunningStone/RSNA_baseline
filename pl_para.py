import attr 

@attr.s
class PL_Para:
    max_epoch = 5
    patience = 3
    num_workers = 8
    init_lr = 0.0005
    weight_decay = 0.0
    lr_patience = 1            # 1 model not improving until lr is decreasing
    lr_factor = 0.4            # by how much the lr is decreasing

    batch_size = 32           # for train