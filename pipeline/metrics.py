import torch
import torchmetrics
from torchmetrics import Metric

def create_metrics(n_classes,):
            #---->Metrics
        if n_classes > 2: 
            metrics_template = torchmetrics.MetricCollection([torchmetrics.Accuracy(task="multiclass",
                                                                            num_classes = n_classes,
                                                                           average='micro'),
                                                    
                                                     torchmetrics.CohenKappa(task="multiclass",num_classes = n_classes),
                                                     torchmetrics.F1Score(task="multiclass",num_classes = n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(task="multiclass",average = 'macro',
                                                                         num_classes = n_classes),
                                                     torchmetrics.Precision(task="multiclass",average = 'macro',
                                                                            num_classes = n_classes),
                                                     torchmetrics.Specificity(task="multiclass",average = 'macro',
                                                                            num_classes = n_classes)])
        else : 
            metrics_template = torchmetrics.MetricCollection([torchmetrics.Accuracy(task="binary",
                                                                            num_classes = 2,
                                                                           average = 'micro'),
                                                    torchmetrics.AUROC(task="binary",num_classes = n_classes, average = 'macro'),
                                                     torchmetrics.CohenKappa(task="binary",num_classes = 2),
                                                     torchmetrics.F1Score(task="binary",num_classes = 2,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(task="binary",average = 'macro',
                                                                         num_classes = 2),
                                                     torchmetrics.Precision(task="binary",average = 'macro',
                                                                            num_classes = 2)])
        return metrics_template


###################################################################
#     for cFscore as metrics
##################################################################

class pF1Score(torchmetrics.Metric):
    """
    Only for binary classification
    from paper: https://aclanthology.org/2020.eval4nlp-1.9.pdf 
    pfbeta should be:
    pfbeta = 2 \cdot \frac{c_precisiont \cdot c_recall}{c_precision + c_recall}
    c_precision = \frac{c_tp}{c_tp + c_fp}
    c_recall = \frac{c_tp}{tp + fn}

    c_tp = \sum_{i=1}^{n} M(x_i,C_j) where sign(y_i==C_j) 
    c_fp = \sum_{i=1}^{n} M(x_i,C_j) where sign(y_i!=C_j)
        where M(x_i,C_j) is prob score of x_i in class C_j
    
    base kaggle implementation to create torchmetrics class
    https://www.kaggle.com/code/sohier/probabilistic-f-score/comments
    """
    def __init__(self,beta:float=1.0):
        super().__init__()
        self.beta = beta
        self.add_state("ctp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("cfp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("y_true_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def _input_format(self,probs, preds, target):
        # format check for input data
        assert preds.shape == target.shape and preds.shape == probs.shape
        probs = probs.clip(0, 1)
        return probs, preds, target

    def update(self, probs: torch.Tensor, preds: torch.Tensor, target: torch.Tensor):
        # probs for probability of positive class [B x C]
        # preds for predicted class which is 0 or 1 [B x 1]
        # target for true class which is 0 or 1 [B x 1]
        probs, preds, target = self._input_format(probs, preds, target)
        
        # only for binary classification [Bx2] or [Bx1] -> [B]
        probs = probs[:,-1]

        # get three values in loop part of base kaggle implementation
        self.y_true_count = torch.sum(target == 1)

        # ctp for label ==1 in probability score
        self.ctp = probs[target==1].sum()

        # cfp for label ==0 in probability score
        self.cfp = probs[target==0].sum()

    def compute(self):
        beta_squared = self.beta * self.beta
        c_precision = self.ctp / (self.ctp + self.cfp)
        c_recall = self.ctp / self.y_true_count
        if (c_precision > 0 and c_recall > 0):
            result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
            return result
        else:
            return 0.0


def create_cFscore(n_classes,):
    metrics_template = torchmetrics.MetricCollection([pF1Score])
    return metrics_template



