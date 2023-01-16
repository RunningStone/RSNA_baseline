import attr 

@attr.s
class BaselinePara:
    backbone:str='resnet50'
    pre_trained:bool=True 
    backbone_dim:int=1024
    no_columns:int=None
    column_out_dim:int=500
    output_size:int=1
    ckpt_path:str=None