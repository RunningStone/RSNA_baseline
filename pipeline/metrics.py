import torchmetrics
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

