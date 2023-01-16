# RNSAB_base
Basic code for RNSA Breast

code structure are designed as following:

cohort: from csv to dataframe (can add file list)
dataset: pytorch dataset module include data aug 

model and model_para define transfer learning model and related parameters

pl and pl_para define optimizer, scheduler, training steps with pytorch-lightning format

train file define training process(TO-DO)

