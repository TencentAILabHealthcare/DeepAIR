import os
import tempfile
import io
import glob
import h5py
import pickle
import shutil

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans

from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='talk',palette='bright')
import umap

# moving_folder = ''
#%%
def isNaN(num):
    return num != num

def df_to_input_feature_extraction(df,
                                    ID = 'ID', 
                                    CDR3b = 'TRB_cdr3',
                                    b_vgene = 'TRB_v_gene',
                                    b_jgene = 'TRB_j_gene', 
                                    CDR3a = 'TRA_cdr3',
                                    a_vgene = 'TRA_v_gene',
                                    a_jgene = 'TRA_j_gene'
                                ):
    """ 
    convert a dataframe into a dictionary of inputs 
    """
    cols = [ID, b_vgene,b_jgene, a_vgene, a_jgene, CDR3b, CDR3a, CDR3b+'_splited', CDR3a+'_splited']
    # check nan
    for item in cols:
        if df[item].isnull().values.any():
            print(f'Nan in the col {item}')
   
    x = {c:df[c].values for c in cols}
    return x

def generate_AF2_feature_on_the_fly(input_data_frame, 
                                    AF2_Feature_Info,                                     
                                    CDR3b = 'TRB_cdr3', 
                                    b_vgene = 'TRB_v_gene',
                                    b_jgene = 'TRB_j_gene',
                                    CDR3a = 'TRA_cdr3', 
                                    a_vgene = 'TRA_v_gene',
                                    a_jgene = 'TRA_j_gene',
                                    task_name = '*',
                                    ID = 'ID', 
                                ):
    
    if AF2_Feature_Info:
        
        data_num = len(input_data_frame)
        seq_max_len = AF2_Feature_Info['seq_max_len']
        fea_dim = AF2_Feature_Info['fea_dim']
        AF2_feature_file = AF2_Feature_Info['feature_file']
        beta_feature = np.zeros((data_num, seq_max_len, fea_dim))
        alpha_feature = np.zeros((data_num, seq_max_len, fea_dim))

        i=0        
        for index, row in input_data_frame.iterrows():
               
            if ID:
                current_ID = row[ID]
            else:
                current_ID = '*'
            
            print('-'*30)
            print(f'{index}_{row[ID]}_{row[CDR3b]}')
            print('-'*30)
            
            current_CDR3b_ID_note = f'{row[CDR3b]}_{row[b_vgene].replace("/", "-")}_{row[b_jgene].replace("/", "-")}_{current_ID}_{task_name}.npy'
            print(current_CDR3b_ID_note)
            CDR3b_File = glob.glob(os.path.join(AF2_feature_file, "*", current_CDR3b_ID_note))[0]
            CDR3b_Feature = get_AF2_feature(CDR3b_File)
            
            # moving_file = os.path.join(moving_folder, '/'.join(CDR3b_File.split('/')[-2:]))
            # os.makedirs(os.path.dirname(moving_file), exist_ok=True)            
            # shutil.copy(CDR3b_File, moving_file)
            
            beta_feature[i] = CDR3b_Feature                             
                
            if ID:
                current_ID = row[ID]
            else:
                current_ID = '*'
                
            print('-'*30)
            print(f'{index}_{row[ID]}_{row[CDR3b]}')
            print('-'*30)
            
            current_CDR3a_ID_note = f'{row[CDR3a]}_{row[a_vgene].replace("/", "-")}_{row[a_jgene].replace("/", "-")}_{current_ID}_{task_name}.npy'  
            print(current_CDR3a_ID_note)
            CDR3a_File = glob.glob(os.path.join(AF2_feature_file, "*", current_CDR3a_ID_note))[0]
            CDR3a_Feature = get_AF2_feature(CDR3a_File) 

            # moving_file = os.path.join(moving_folder, '/'.join(CDR3a_File.split('/')[-2:]))
            # os.makedirs(os.path.dirname(moving_file), exist_ok=True)
            # shutil.copy(CDR3a_File, moving_file)
            
            alpha_feature[i] = CDR3a_Feature          

            i+=1        
    else:
        beta_feature = None
        alpha_feature = None
            
    return beta_feature, alpha_feature

def get_AF2_feature(feature_file):
    feature = np.load(feature_file)
    return feature    

#%%
def violinplot(input_DF, save_path):
    sns.set(style="whitegrid")
    sns.set(rc = {'figure.figsize':(8,8)})
    ax = sns.violinplot(data=input_DF, width=0.8)
    os.makedirs(os.path.dirname(save_path),exist_ok=True)    
    plt.xticks(rotation=60)
    # plt.legend(frameon=False)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)   
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
    
def MIL_head(input, MIL_Method):
    """
    MIL Head 

    Args:
        input (numpy.array): input receptor level prediction
        MIL_Method ([int, 'average_pooling', 'majority_voting']]): different MIL heads

    Returns:
        float: prediction probability
    """
    input = input[np.logical_not(np.isnan(input))]
    N = input.shape[0]
    input_reverse_sort = abs(np.sort(-input))
    if isinstance(MIL_Method, int):
        assert (MIL_Method>=0) and (MIL_Method<=100)
        return input_reverse_sort[int(N*MIL_Method/(float(100)))]
    elif MIL_Method == 'average_pooling':
        return np.mean(input_reverse_sort)
    elif MIL_Method == 'majority_voting':
        return np.sum(input_reverse_sort>=0.5)/float(N)
    else:
        raise ValueError(f'please check the MIL_Method: {MIL_Method}')

def seq_modafication(seq):
    seq = " ".join("".join(seq.split()))
    return seq

def print_model_summary(string, model_summary_file='modelsummary.txt'):
    with open(model_summary_file,'w+') as f:
        print(string, file=f)

def save_dict(dictionary, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dictionary, f)

def load_dict(file_name):        
    with open(file_name, 'rb') as f:
        loaded_dict = pickle.load(f)
        return loaded_dict