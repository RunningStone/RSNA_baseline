
csv_path = '/RSNA/data/train.csv'
slide_root='/RSNA/data/train'
#######################################################
# init cohort
#######################################################
from .pipeline.cohort import Cohort
cohort=Cohort(cohort_csv=csv_path,slide_root=slide_root)

# for k fold cross validation
cohort.init_k_fold(K=4)
cohort.get_K_th_fold(0)
#######################################################
# init wandb
#######################################################

#######################################################
# init model
#######################################################
#----> general pytorch lightning protocol parameters in this package
import torch
#from RSNA_baseline.pipeline.pl_para import PL_Para
from RSNA_baseline.models.pl_config import baseline_pl_para 

#----> each algorithm define model in a folder
from RSNA_baseline.models.baseline import BaselinePreTrain
from RSNA_baseline.models.baseline_para import BaselinePara
from RSNA_baseline.models.pl_baseline import pl_baseline


#----> init pl paras
#pl_para you can define model,loss,optimizer,
# scheduler and relevent parameters
pl_para = baseline_pl_para
pl_para.model_define = BaselinePreTrain
pl_para.sch_func = torch.optim.lr_scheduler.ExponentialLR
pl_para.sch_para = {"gamma": 0.9,}
pl_para.batch_size = 8           # for train
pl_para.num_workers = 2
pl_para.additional_info = [ 'age', 'implant',"machine_id"]
#----> init model parameters
model_para = BaselinePara()
model_para.backbone = "resnet50"
model_para.backbone_dim = 2048
model_para.no_columns = len(pl_para.additional_info)
model_para.column_out_dim = 100


#----> init model
pl_model = pl_baseline(pl_para, model_para)
#######################################################
# init dataloader
#######################################################
from RSNA_baseline.pipeline.dataset import RSNADataset
from torch.utils.data import DataLoader
trainset = RSNADataset(cohort=cohort,is_train=True,additional_info=pl_para.additional_info)
trainloader = DataLoader(trainset, 
                        batch_size=pl_para.batch_size, 
                        shuffle=pl_para.shuffle, 
                        num_workers=pl_para.num_workers)

valset = RSNADataset(cohort=cohort,is_train=False,additional_info=pl_para.additional_info)
valloader = DataLoader(valset, 
                        batch_size=pl_para.batch_size, 
                        shuffle=pl_para.shuffle, 
                        num_workers=pl_para.num_workers)
#######################################################
# init trainer
#######################################################

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from copy import deepcopy
import torch
import gc

# trainer config
trainer = pl.Trainer(
    accelerator='gpu', 
    precision=32,
    default_root_dir='.log_causal_lm',
    max_epochs=10,
    callbacks=[
        EarlyStopping(monitor='dev_loss',patience=2), # 監測dev_loss的變化，超過兩次沒有改進就停止
        ModelCheckpoint(monitor='dev_loss',filename='{epoch}-{dev_loss:.2f}',save_last=True),
    ]
)


for param in pl_model.model.features.parameters():
    param.requires_grad = False

trainer.fit(pl_model,trainloader,valloader)