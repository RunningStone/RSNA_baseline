
import torch 
import torch.nn as nn


import pytorch_lightning as pl


from .pl_para import PL_Para
from .metrics import create_metrics


class pl_base(pl.LightningModule):
    """
    base class for model implementation
    """
    def __init__(self,pl_para:PL_Para, model_para):
        super().__init__()
        self.pl_para = pl_para
        self.model_para = model_para
        self.model = self.pl_para.model_define(model_para)
        self.criterion = self.pl_para.criterion()
        self.model_para = model_para
        #self.save_hyperparameters()

        self.metrics()

    def metrics(self):
        metrics = self.pl_para.create_metrics(self.model_para.output_size)

        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

    def pre_process(self, batch):
        img,meta, y = batch
        y = y.unsqueeze(1).float()
        return img,meta, y

    def training_step(self, batch, batch_idx):
        img,meta, y = self.pre_process(batch)
        results_dict = self.model(img,meta)
        y_prob = results_dict['Y_prob']

        loss = self.criterion(y_prob, y)
        self.log('train_loss', loss,logger = True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        img,meta, y = self.pre_process(batch)
        results_dict = self.model(img,meta)
        y_prob = results_dict['Y_prob']
        loss = self.criterion(y_prob, y)
        self.log('val_loss', loss,logger = True)
        return {'val_loss': loss,'Y_prob':y_prob,'label':y, 'Y_hat':results_dict['Y_hat']}
    
    def configure_optimizers(self):
        opt_para_dict = self.pl_para.opt_paras
        opt_para_dict['lr'] = self.pl_para.init_lr
        optimizer = self.pl_para.opt_func(self.model.parameters(), **opt_para_dict)
        scheduler = self.pl_para.sch_func(optimizer=optimizer, **self.pl_para.sch_para)
        
        return [optimizer], [scheduler]

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        self.log('avg_train_loss', avg_loss,on_epoch = True, logger = True)

        
    def validation_epoch_end(self, val_step_outputs):
        max_probs = torch.cat([x['Y_hat'] for x in val_step_outputs])
        target = torch.cat([x['label'] for x in val_step_outputs],dim=0)
        
        #---->
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)
    
