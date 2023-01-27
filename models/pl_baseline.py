from ..pipeline.pl_base import pl_base

# do modification here
import torch

class pl_baseline(pl_base):
    def __init__(self,pl_para, model_para):
        super().__init__(pl_para, model_para)
        pass

    def configure_optimizers(self):
        return super().configure_optimizers()

    def training_step(self, batch, batch_idx):
        # change training step here
        return super().training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        # change validation step here
        return super().validation_step(batch, batch_idx)
    
    def training_epoch_end(self, training_step_outputs):
        # change training epoch end here
        return super().training_epoch_end(training_step_outputs)

    def validation_epoch_end(self, val_step_outputs):
        # change validation epoch end here
        max_probs = torch.cat([x['Y_hat'] for x in val_step_outputs])
        target = torch.cat([x['label'] for x in val_step_outputs],dim=0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs],dim=0)
        
        #---->
        # valid_metrics only include cFscore
        # and cFscore need three input, prob, pred, target
        # can not directly use 
        self.log_dict( self.valid_metrics(probs, max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)