"""
Log in cohort and organize data
show meta data
"""
from os import listdir
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm

class Cohort:
    def __init__(self,cohort_csv:str,slide_root:str,file_type:str='dcm'):
        """
        cohort_csv:str: csv file for cohort
        """
        self.cohort_csv = cohort_csv
        self.df = pd.read_csv(self.cohort_csv)
        

        self.slide_root = slide_root

        # file type
        assert file_type in ['dcm','png']       
        self.file_type = file_type

        self.loc_list = None
        self.kfold = None
        self.get_filelist()
    ##################################################
    #       read slide related 
    ##################################################
    def load_patient_slides(self, patient_id:str):
        """
        give a patient id and return a list of dicom data instance
        """
        patient_path = self.slide_root + '/' + patient_id
        return [pydicom.dcmread(patient_path + '/' + file) for file in listdir(patient_path)]
    
    def get_filelist(self):
        """
        extract data list from cohort csv file
        [{"path":xx,'laterality':xx, 'view', 'age', 'implant'},...]
        """
        base_path = self.slide_root + "/"
        all_paths = []
        for k in tqdm(range(len(self.df))):
            row = self.df.iloc[k, :]
            if self.file_type == 'dcm':
                f_name = f"{base_path}/{str(row.patient_id)}/{str(row.image_id)}.dcm"
            elif self.file_type == 'png':
                f_name = f"{base_path}/{str(row.patient_id)}_{str(row.image_id)}.png"
            else:
                raise ValueError(f"file type {self.file_type} not supported..")
            all_paths.append(f_name)
        self.df["path"] = all_paths
        self.loc_list = all_paths

        # copy the df for split train and test
        self.all_df = self.df.copy()
        print("Cohort read filelist done!")

    def read_slide(self,df:pd.DataFrame,idx:int):
        """
        for a given index, read the slide and return the image
        in:
            idx:int: index of the slide
        return: 
            image: np.array
        """
        image_path = df.iloc[idx]['path']
        if self.file_type == 'png':
            image = np.array(Image.open(image_path))
        elif self.file_type == 'dcm':
            image = pydicom.dcmread(image_path).pixel_array.astype(np.float32)
        return image

    def init_k_fold(self,K:int = 4):
        self.k = K
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
        self.kfold = KFold(n_splits=K, shuffle=True, random_state=2023)

    def get_K_th_fold(self,K:int):
        """
        get K-th split of the dataset 
        """
        train_index, val_index = next(iter(self.kfold.split(self.all_df)))
        self.train_df = pd.DataFrame(self.all_df.iloc[train_index])
        self.val_df = pd.DataFrame(self.all_df.iloc[val_index])

    def set_train_val(self,is_train:bool):
        if self.kfold is not None:
            if is_train:
                return self.train_df
            else:
                return self.val_df
        else:
            return self.df

        
        

    