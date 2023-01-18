import torchmetrics
def create_metrics(n_classes,):
            #---->Metrics
        if n_classes > 2: 
            metrics_template = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = n_classes,
                                                                           average='micro'),
                                                    
                                                     torchmetrics.CohenKappa(num_classes = n_classes),
                                                     torchmetrics.F1Score(num_classes = n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = n_classes),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = n_classes),
                                                     torchmetrics.Specificity(average = 'macro',
                                                                            num_classes = n_classes)])
        else : 
            metrics_template = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = 2,
                                                                           average = 'micro'),
                                                    torchmetrics.AUROC(num_classes = n_classes, average = 'macro'),
                                                     torchmetrics.CohenKappa(num_classes = 2),
                                                     torchmetrics.F1Score(num_classes = 2,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = 2),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = 2)])
        return metrics_template

