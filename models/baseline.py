# PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor

from torch.optim.lr_scheduler import ReduceLROnPlateau

# pre-trained models
import timm
from ..models.baseline_para import BaselinePara

class BaselinePreTrain(nn.Module):
    def __init__(self,model_para: BaselinePara):
        super().__init__()
        """
        in:
            model_para: BaselinePara, parameters for the model

        simple model with pre-trained backbone
        img:  backbone  |
                        |-> classifior -> output
        csv: two linears|
        """
        #----> feature net
        self.backbone = model_para.backbone
        self.backbone_dim = model_para.backbone_dim
        self.pre_trained = model_para.pre_trained
        self.ckpt_path = model_para.ckpt_path
        
        #----> csv net
        self.no_columns = model_para.no_columns
        self.column_out_dim = model_para.column_out_dim

        #----> classification net
        self.output_size = model_para.output_size


        
        # Define Feature part (IMAGE)
        self.get_encoder() # 1000 neurons out
        # (metadata)
        self.csv = nn.Sequential(nn.Linear(self.no_columns, self.column_out_dim),
                                 nn.BatchNorm1d(self.column_out_dim),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2))
        
        # Define Classification part
        self.classification = nn.Linear(self.backbone_dim \
                                 + self.column_out_dim, self.output_size)

    def get_encoder(self):
        self.features = timm.create_model(self.backbone, \
                            pretrained=self.pre_trained, num_classes=0)
        if not self.pre_trained and self.ckpt_path is not None:
            self.features.load_state_dict(torch.load(self.ckpt_path))
        
    def forward(self, image, meta):
        
        # Image CNN
        image = self.features(image)
  
        # CSV FNN
        meta = self.csv(meta)

        # Concatenate layers from image with layers from csv_data
        image_meta_data = torch.cat((image, meta), dim=1)

        # CLASSIF
        out = self.classification(image_meta_data)

        return out