
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
from .pipeline.pl_para import PL_Para
#----> each algorithm define model in a folder
from .models.baseline import BaselinePreTrain
from .models.baseline_para import BaselinePara
from .models.pl_baseline import pl_baseline

#----> init model parameters
model_para = BaselinePara()
#----> init paras
#pl_para you can define model,loss,optimizer,
# scheduler and relevent parameters
pl_para = PL_Para(model_define = BaselinePreTrain)


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
