"""
Log in cohort and organize data
show meta data
"""
from os import listdir
import numpy as np
import pandas as pd
import pydicom
import torch

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from tqdm import tqdm

class Cohort:
    def __init__(self,cohort_csv:str,slide_root:str):
        """
        cohort_csv:str: csv file for cohort
        """
        self.cohort_csv = cohort_csv
        self.df = pd.read_csv(self.cohort_csv)

        self.slide_root = slide_root

    ##################################################
    #       read slide related 
    ##################################################
    def load_patient_slides(self, patient_id:str):
        """
        give a patient id and return a list of dicom data instance
        """
        patient_path = self.slide_root + '/' + patient_id
        return [pydicom.dcmread(patient_path + '/' + file) for file in listdir(patient_path)]
    
    def add_path_in_df(self):
        """
        extract data list from cohort csv file
        [{"path":xx,'laterality':xx, 'view', 'age', 'implant'},...]
        """
        base_path = self.slide_root + "/"
        all_paths = []
        for k in tqdm(range(len(self.df))):
            row = self.df.iloc[k, :]
            all_paths.append(base_path 
                            + str(row.patient_id) 
                            + "/" + str(row.image_id) 
                            + ".dcm")
        self.df["path"] = all_paths

    