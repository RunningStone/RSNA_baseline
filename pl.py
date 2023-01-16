
import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl


from .model_para import BaselinePara
from .pl_para import PL_Para
from .model import BaselinePreTrain
from .utils import create_metrics


class pl_baseline(pl.LightningModule):
    def __init__(self,pl_para:PL_Para, model_para: BaselinePara):
        super().__init__()
        self.pl_para = pl_para
        self.model_para = model_para
        self.model = BaselinePreTrain(model_para)
        self.criterion = nn.BCEWithLogitsLoss()
        self.model_para = model_para
        self.save_hyperparameters()

    def metrics(self):
        metrics = create_metrics(self.model_para.output_size)

        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

    def pre_process(self, batch):
        x, y = batch
        y = y.unsqueeze(1).float()
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self.pre_process(batch)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = self.pre_process(batch)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss,'Y_hat',y_hat,'label',y)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.pl_para.init_lr, 
                                     weight_decay=self.pl_para.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', 
                                      patience=self.pl_para.lr_patience, verbose=True, factor=self.pl_para.lr_factor)
        
        return optimizer, scheduler

    def validation_epoch_end(self, val_step_outputs):
        max_probs = torch.cat([x['Y_hat'] for x in val_step_outputs])
        target = torch.cat([x['label'] for x in val_step_outputs],dim=0)
        
        #---->
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)
    
