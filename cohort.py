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

    ##################################################
    #      cohort meta related or EDA process
    ##################################################
    def show_base_meta(self):
        """
        show basic meta data of cohort
        """
        df = self.df
        print("Show head of cohort csv file:\n")
        print(df.head())
        print("Show N/A values count:\n")
        print(df.isna().sum())

        # for RSNABreast dataset
        num_patients = df['patient_id'].nunique()
        min_patient_age = int(df['age'].min())
        max_patient_age = int(df['age'].max())
        groupby_id = df.groupby('patient_id')['cancer'].apply(lambda x: x.unique()[0])
        n_negative = (groupby_id == 0).sum()
        n_positive = (groupby_id == 1).sum()

        print(f"There are {num_patients} different patients in the train set.\n")
        print(f"The younger patient is {min_patient_age} years old.")
        print(f"The older patient is {max_patient_age} years old.\n")
        print(f"There are {n_negative} patients negative to breast cancer. Ratio = {n_negative / num_patients}")
        print(f"There are {n_positive} patients positive to breast cancer. Ratio = {n_positive / num_patients}")


    def show_slide_meta(self):
        data = self.df

        print("Show slide meta data:\n")
        print(f"laterality: {data['laterality'].value_counts()}")
        print(f"implant: {data['implant'].value_counts()}")
        print(f"difficult_negative_case: {data['difficult_negative_case'].value_counts()}")
        print(f"view: {data['view'].value_counts()}")
        print(f"density: {data['density'].value_counts()}")
        print(f"site_id: {data['site_id'].value_counts()}")
        print(f"site: {data['machine_id'].value_counts()}")

        n_images_per_patient = data['patient_id'].value_counts()
        plt.figure(figsize=(16, 6))
        sns.countplot(n_images_per_patient, palette='Reds_r')
        plt.title("Number of images taken per patients")
        plt.xlabel('Number of images taken')
        plt.ylabel('Count of patients')
        plt.show()

    def show_age_distribution(self):
        data = self.df
        ages = data.groupby('patient_id')['age'].apply(lambda x: x.unique()[0])
        cancer_ages = data[data['cancer'] == 1].groupby('patient_id')['age'].apply(lambda x: x.unique()[0])
        no_cancer_ages = data[data['cancer'] == 0].groupby('patient_id')['age'].apply(lambda x: x.unique()[0])

        print("Age distribution of the patients:\n")
        print("Mean:", ages.mean())
        print("Std:", ages.std())
        print("Q1:", ages.quantile(0.25))
        print("Median:", ages.median())
        print("Q3:", ages.quantile(0.75))
        print("Mode:", ages.mode()[0])

        plt.figure(figsize=(16, 10))

        plt.subplot(1, 2, 1)
        sns.histplot(ages, bins=63, color='orange', kde=True)
        plt.title("All the patient")
        plt.xlim(33, 89)

        plt.subplot(2, 2, 2)
        sns.histplot(cancer_ages, bins=51, color='red', kde=True)
        plt.title("Patients with cancer")
        plt.xlim(33, 89)

        plt.subplot(2, 2, 4)
        sns.histplot(no_cancer_ages, bins=63, color='green', kde=True)
        plt.title("Patients without cancer")
        plt.xlim(33, 89)

        plt.suptitle("Age distribution of the patients")
        plt.show()

    def show_target_meta(self):
        data = self.df
        biopsy_counts = data.groupby('cancer')['biopsy'].value_counts().unstack().fillna(0)
        biopsy_perc = biopsy_counts.transpose() / biopsy_counts.sum(axis=1)

        fig, ax = plt.subplots(1, 5, figsize=(10, 10))
        sns.countplot(data['cancer'], palette='Greens', ax=ax[0])
        sns.heatmap(biopsy_perc, square=True, annot=True, fmt='.1%', cmap='Blues', ax=ax[1])
        ax[0].set_title("Number of images showing cancer")
        ax[1].set_title("Percentage of images\nresulting in a biopsy")

        sns.countplot(data[data['cancer'] == True]['invasive'], ax=ax[2], palette='Reds')
        sns.countplot(data[data['cancer'] == False]['BIRADS'], order=[0, 1, 2], ax=ax[3], palette='Blues')
        sns.countplot(data[data['cancer'] == True]['BIRADS'], order=[0, 1, 2], ax=ax[4], palette='Blues')
        ax[2].set_title("Count of invasive cancer images")
        ax[3].set_title("BIRADS for healthy images")
        ax[4].set_title("BIRADS for cancer images")
        plt.show()

    ##################################################
    #      slide meta related 
    ##################################################
    def show_slide(self,slide:pydicom.dataset.FileDataset):
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        im = ax[0].imshow(slide.pixel_array, cmap='bone')
        ax[0].grid(False)
        fig.colorbar(im, ax=ax[0])
        sns.histplot(slide.pixel_array.flatten(), ax=ax[1], bins=50)
        plt.show()
    
    def show_patient_slides(self,patient_id:str):
        plt.figure(figsize=(22, 8))
        scans = self.load_patient_slides(patient_id)
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(scans[i].pixel_array, cmap='bone')
            plt.grid(False)
        plt.suptitle("All scans of patient 13095")
        plt.show()

    def show_machine_diff_in_slide(self):
        from scipy.stats import mode
        data = self.df
        modes, rows, cols = [], [], []
        machine_ids = data['machine_id'].unique()
        for m_id in machine_ids:
            m_id_modes, m_id_rows, m_id_cols = [], [], []
            print(f"Counting Machine id {m_id}:\n")
            patient_ids = data[data['machine_id'] == m_id]['patient_id'].unique()
            for n in range(50):
                try:
                    scan = self.load_patient_slides( str(patient_ids[n]))[0]
                    m_id_modes.append(mode(scan.pixel_array.flatten())[0][0])
                    m_id_rows.append(scan.Rows)
                    m_id_cols.append(scan.Columns)
                except IndexError:
                    break
            modes.append(m_id_modes)
            rows.append(m_id_rows)
            cols.append(m_id_cols)
        
        medians = [np.median(x) for x in modes]
        stds = [np.std(x) for x in modes]
        rows = [np.mean(x) for x in rows]
        cols = [np.mean(x) for x in cols]
        df = pd.DataFrame(data={'Machine ID': machine_ids, 'Mode (median)': medians, 'Mode (std)': stds, 'Rows (mean)': rows, 'Cols (mean)': cols})
        count_df=df.astype(int).set_index('Machine ID').T
        print(count_df)

        plt.figure(figsize=(22, 8))
        for i, m_id in enumerate(machine_ids):
            patient_ids = data[data['machine_id'] == m_id]['patient_id'].unique()
            scan = self.load_patient_slides(str(patient_ids[0]))[0] # Load first scan of first patient
            plt.subplot(2, 5, i+1)
            plt.imshow(scan.pixel_array, cmap='bone')
            plt.title(f"Machine {m_id}")
            plt.colorbar()
            plt.grid(False)
        plt.show()

    def show_implant_diff(self):
        data = self.df
        m_id_implants = data[data['implant'] == 1]['machine_id'].unique()
        print(f"Scans showing implents are from machines {m_id_implants}\n")
        patient_ids = data[data['implant'] == 1]['patient_id'].unique()

        # Display scans showing implants
        plt.figure(figsize=(22, 8))
        for i in range(10):
            scan = self.load_patient_slides(str(patient_ids[i]))[0] # Load first scan of the patient
            plt.subplot(2, 5, i+1)
            plt.imshow(scan.pixel_array, cmap='bone')
            plt.title(f"Patient {patient_ids[i]}")
            plt.grid(False)
        plt.show()

    def display_cancer_or_not(self,cancer:bool=True):
        """
        show 10 random scans of cancer or not cancer
        """
        data = self.df
        cancer_scans = data[data['cancer'] == int(cancer)].sample(frac=1, random_state=0)
        plt.figure(figsize=(22, 10))
        for i in range(10):
            patient = str(cancer_scans.iloc[i][['patient_id']][0])
            file = str(cancer_scans.iloc[i][['image_id']][0]) + '.dcm'
            file_loc = self.slide_root + '/' + patient + '/' + file
            scan = pydicom.dcmread(file_loc)
            plt.subplot(2, 5, i+1)
            plt.imshow(scan.pixel_array, cmap='bone')
            plt.title(f"Patient {patient}\nScan {file}")
            plt.grid(False)
        plt.suptitle(f"Cancer = {cancer}")
        plt.show()