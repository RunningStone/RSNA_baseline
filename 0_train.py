
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
from .pipeline.dataset import RSNADataset
from torch.utils.data import DataLoader
dataset = RSNADataset(cohort=cohort,is_train=True)
data_loader = DataLoader(dataset, 
                        batch_size=pl_para.batch_size, 
                        shuffle=pl_para.shuffle, 
                        num_workers=pl_para.num_workers)
#######################################################
# init trainer
#######################################################
