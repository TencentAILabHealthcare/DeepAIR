    
import os
import pandas as pd
import numpy as np
import pickle
import argparse
#%%
def get_AF2_feature(feature_file):
    with open(feature_file,'rb') as f:
        feature = pickle.load(f)
    return feature['representations']['structure_module']

def parse_AF2_data(AIR_file_path, 
                    AF2_feature_folder, 
                    output_folder, 
                    max_len = 40,
                    feature_dimentation = 384,
                    ID = 'ID', 
                    CDR3b = 'Heavy_cdr3', 
                    CDR3a = 'Light_cdr3', 
                    b_vgene = 'Heavy_v_gene',
                    b_jgene = 'Heavy_j_gene',
                    a_vgene = 'Light_v_gene',
                    a_jgene = 'Light_j_gene',
                    cdr3b_begin = 'Heavy_cdr3_begin', 
                    cdr3b_end = 'Heavy_cdr3_end',
                    cdr3a_begin = 'Light_cdr3_begin',
                    cdr3a_end = 'Light_cdr3_end',
                    af_pkl_file = 'result_model_1_ptm_pred_0.pkl',
                    task_name = '*'
                    ):
    
    os.makedirs(output_folder, exist_ok=True)
    
    AIR_df = pd.read_csv(AIR_file_path)
    
    for index, row in AIR_df.iterrows():
        print('-'*30)
        print(f'{index}_{row[CDR3b]}')
        print('-'*30)
                    
        current_CDR3b_ID_note = f'{row[CDR3b]}_{row[ID]}'
        current_CDR3a_ID_note = f'{row[CDR3a]}_{row[ID]}'
        
        current_cdr3b_begin = row[cdr3b_begin]
        current_cdr3b_end = row[cdr3b_end]
        current_cdr3a_begin = row[cdr3a_begin]
        current_cdr3a_end = row[cdr3a_end]

        # B chain
        CDR3b_Feature = np.zeros((max_len, feature_dimentation))
        feature_file = os.path.join(AF2_feature_folder, current_CDR3b_ID_note, af_pkl_file)
        CDR3b_Feature[0:(current_cdr3b_end-current_cdr3b_begin)] = get_AF2_feature(feature_file)[current_cdr3b_begin:current_cdr3b_end]
        
        # A chain
        CDR3a_Feature = np.zeros((max_len, feature_dimentation))
        feature_file = os.path.join(AF2_feature_folder, current_CDR3a_ID_note, af_pkl_file)
        CDR3a_Feature[0:(current_cdr3a_end-current_cdr3a_begin)] = get_AF2_feature(feature_file)[current_cdr3a_begin:current_cdr3a_end]            
        
        output_CDR3b_ID_note = f'{row[CDR3b]}_{row[b_vgene].replace("/", "-")}_{row[b_jgene].replace("/", "-")}_{row[ID]}_{task_name}.npy'
        output_CDR3a_ID_note = f'{row[CDR3a]}_{row[a_vgene].replace("/", "-")}_{row[a_jgene].replace("/", "-")}_{row[ID]}_{task_name}.npy'   
        np.save(os.path.join(output_folder, output_CDR3b_ID_note), CDR3b_Feature)
        np.save(os.path.join(output_folder, output_CDR3a_ID_note), CDR3a_Feature)     
        
        print(output_CDR3b_ID_note)
        print(CDR3b_Feature)
        print(output_CDR3a_ID_note)
        print(CDR3a_Feature)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepAIR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # data options
    parser.add_argument('--AIR_file_path', 
                        type=str, 
                        default='./AB_Combine_Drop_Duplicates_example.csv',
                        help='path to the vdj file'
                    )
    parser.add_argument('--AF2_feature_folder',
                        type=str,
                        default='./output',
                        help='path to the AF2 feature folder'
                    )
    parser.add_argument('--output_folder', 
                        type=str, 
                        default='./AB_Combine_Drop_Duplicates_CDR3Region_example.csv', 
                        help='path to the output table'
                    )
    parser.add_argument('--ID', type=str, default='ID', help='ID column name')
    parser.add_argument('--Chain1_cdr3', type=str, default='Heavy_cdr3', help='The cdr3 region of BCR Heavy chain/TCR Beta chain')
    parser.add_argument('--Chain2_cdr3', type=str, default='Light_cdr3', help='The cdr3 region of BCR Light chain/TCR Alpha chain')
    parser.add_argument('--Chain1_v_gene', type=str, default='Heavy_v_gene', help='The v gene of BCR Heavy chain/TCR Beta chain')
    parser.add_argument('--Chain1_j_gene', type=str, default='Heavy_j_gene', help='The J gene of BCR Heavy chain/TCR Beta chain')
    parser.add_argument('--Chain2_v_gene', type=str, default='Light_v_gene', help='The v gene of BCR Light chain/TCR Alpha chain')
    parser.add_argument('--Chain2_j_gene', type=str, default='Light_j_gene', help='The J gene of BCR Light chain/TCR Alpha chain')
    parser.add_argument('--task_name', type=str, default='CoV-AbDab', help='the task name (dataset) as a part of note for the AF2 feature')
    args = parser.parse_args()   

    AIR_file_path = args.AIR_file_path
    AF2_feature_folder = args.AF2_feature_folder
    output_folder = args.output_folder
    ID = args.ID
    Chain1_cdr3 = args.Chain1_cdr3
    Chain2_cdr3 = args.Chain2_cdr3
    Chain1_v_gene = args.Chain1_v_gene
    Chain1_j_gene = args.Chain1_j_gene
    Chain2_v_gene = args.Chain2_v_gene
    Chain2_j_gene = args.Chain2_j_gene
    task_name = args.task_name
    
    parse_AF2_data(AIR_file_path, 
                    AF2_feature_folder, 
                    output_folder, 
                    max_len = 40,
                    feature_dimentation = 384,
                    ID = ID, 
                    CDR3b = Chain1_cdr3, 
                    CDR3a = Chain2_cdr3, 
                    b_vgene = Chain1_v_gene,
                    b_jgene = Chain1_j_gene,
                    a_vgene = Chain2_v_gene,
                    a_jgene = Chain2_j_gene,
                    cdr3b_begin = f'{Chain1_cdr3}_begin', 
                    cdr3b_end = f'{Chain1_cdr3}_end',
                    cdr3a_begin = f'{Chain2_cdr3}_begin',
                    cdr3a_end = f'{Chain2_cdr3}_end',
                    task_name = task_name
                    )