"""

"""
import pydicom
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from .data_aug import get_transform,get_three_channels
from .cohort import Cohort
class RSNADataset(Dataset):
    
    def __init__(self, 
                cohort:Cohort, 
                is_train=True,
                get_transform:callable=get_transform,
                three_channels_fn:callable=get_three_channels,
                additional_info = ['laterality', 'view', 'age', 'implant'],):
        '''Initialize the dataset.
        Args:
            df: pandas dataframe with the data from cohort instance
            get_transform: callable function to get the transformation in data_aug.py
            is_train: bool to indicate if the dataset is for training or validation
            additional_info: list of strings indicates csv columns to be added as additional info
        '''
        self.cohort = cohort
        self.is_train = is_train
        self.df = self.cohort.set_train_val(is_train=is_train)

        # Data Augmentation (custom for each dataset type)
        self.transform = get_transform(is_train=is_train)
        self.three_channels_fn = three_channels_fn
        
        self.additional_info = additional_info
            
    def __len__(self):
        return len(self.df)
    
    def get_additional_info(self,idx:int):
        if self.additional_info is not None and len(self.additional_info) > 0:
            add_data = np.array(self.cohort.df.iloc[idx][self.additional_info].values, 
                                dtype=np.float32)
            return add_data
        else:
            return None

    def get_label(self,idx:int):
        return self.cohort.df['cancer'][idx]

    def __getitem__(self, index):
        '''Take each row in batcj at a time.'''
        
        # Select path and read image
        image = self.cohort.read_slide(self.df,index)
        # Apply transforms
        transf_image = self.transform(image=image)['image']
        transf_image = self.three_channels_fn(transf_image) # Change image from 1 channel (B&W) to 3 channels

        # additional .csv information
        add_data = self.get_additional_info(index)
        
        # label
        label = self.get_label(index)
        return transf_image, add_data, label

