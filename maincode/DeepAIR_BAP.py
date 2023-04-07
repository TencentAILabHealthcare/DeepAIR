import os
import sys
import logging
import numpy as np
import pandas as pd
import argparse

sys.path.insert(0, os.path.abspath('../DeepAIR'))

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from transformers import TFBertModel, BertTokenizer

from config import DeepAIR_BAP_saved_model_dict
from deepair.modelling.regression import RegressionModelWithProcessor
from deepair.utility.utility import df_to_input_feature_extraction
from deepair.utility.utility import generate_AF2_feature_on_the_fly
from DeepAIR_Transformer_Feature_Extraction import generate_transformer_feature_on_the_fly
#%%
def one_hot_decode(encoded_data, axis=-1):
    return np.argmax(encoded_data, axis=axis)

def model_seed_set(model_seed=13):
    os.environ['PYTHONHASHSEED']=str(model_seed)
    np.random.seed(model_seed)
    tf.random.set_seed(model_seed)

def parse_epitope_epitopeReg_from_eiptopeCls(input):
    epitope = '_'.join(input.split('_')[0:-1])
    epitopeReg = epitope+'_Reg'
    return (epitope, epitopeReg)

def predict(input_data_file, 
            transformer_model_folder,
            seq_transformer_info,
            AF2_Feature_Info, 
            selected_epitope = None, 
            output_folder = None,
            task_name = '*'
            ):

    # print('-'*30)
    # print(f'current epitope is {epitope}')

    
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        
    input_df = pd.read_csv(input_data_file) 
    
    # add transformer features
    seq_trans_beta_feature, seq_trans_alpha_feature = generate_transformer_feature_on_the_fly(input_df,
                                                         transformer_model_folder,
                                                         seq_transformer_info = seq_transformer_info
        )    
    # add AF2 structure features
    AF2_feature_beta_feature, AF2_feature_alpha_feature = generate_AF2_feature_on_the_fly(input_df, 
                                                                                        AF2_Feature_Info,
                                                                                        CDR3b = 'TRB_cdr3', 
                                                                                        b_vgene = 'TRB_v_gene',
                                                                                        b_jgene = 'TRB_j_gene',
                                                                                        CDR3a = 'TRA_cdr3', 
                                                                                        a_vgene = 'TRA_v_gene',
                                                                                        a_jgene = 'TRA_j_gene',
                                                                                        task_name = task_name,
                                                                                        ID = 'ID', 
                                                                                    )

    test_input = df_to_input_feature_extraction(input_df)
    test_input['TRB_cdr3_splited'] = seq_trans_beta_feature
    test_input['TRA_cdr3_splited'] = seq_trans_alpha_feature
    test_input['TRB_cdr3_Stru'] = AF2_feature_beta_feature
    test_input['TRA_cdr3_Stru'] = AF2_feature_alpha_feature
            
    output_df = pd.DataFrame()
    output_df['ID'] = test_input['ID']
    output_df['TRB_cdr3'] = test_input['TRB_cdr3']
    output_df['TRA_cdr3'] = test_input['TRA_cdr3']
    output_df['TRB_v_gene'] = test_input['TRB_v_gene']
    output_df['TRB_j_gene'] = test_input['TRB_j_gene']
    output_df['TRA_v_gene'] = test_input['TRA_v_gene']
    output_df['TRA_j_gene'] = test_input['TRA_j_gene']

    for curr_epitope in selected_epitope: 

        test_input[curr_epitope+'_Reg_Stage'] = np.zeros(shape=(len(input_df),),dtype=np.int64)
        
        model_save_path = DeepAIR_BAP_saved_model_dict[curr_epitope]
    
        
        # now load the saved model, from (tempoarary) file, into notebook
        loaded_model = RegressionModelWithProcessor.from_file(model_save_path)
        # print('1.-'*30)
        # print(tf.executing_eagerly()) 

        preds = loaded_model.run(test_input)[0]
        if tf.is_tensor(preds):
            preds = preds.numpy()
    
        output_df[curr_epitope] = preds 
        
        del test_input[curr_epitope+'_Reg_Stage']
        
    output_df.to_csv(os.path.join(output_folder,'prediction_results.csv'))
    
    return output_df

def main():
    
    parser = argparse.ArgumentParser(description='DeepAIR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # data options
    parser.add_argument('--input_data_file', 
                        type=str, 
                        default='../DeepAIR/sampledata/BAP/B0801_RAKFKQLL_BZLF1_EBV_Reg.csv',
                        help='path to the input dataframe'
                    )
    parser.add_argument('--result_folder', 
                        type=str, 
                        default='../DeepAIR/result_BAP/B0801_RAKFKQLL_BZLF1_EBV', 
                        help='folder to save the results'
                    )
    parser.add_argument('--epitope', 
                        type=str,         
                        default='B0801_RAKFKQLL_BZLF1_EBV', 
                        help='Select an interested epitope'
                    )
    parser.add_argument('--AF2_feature_folder', 
                        type=str, 
                        default='../DeepAIR/sampledata/structure_feature/BAP',
                        help='AF2 feature file name'
                    )
    parser.add_argument('--transformer_model_folder',   
                        type=str, 
                        default='../DeepAIR/ProtTrans/prot_bert_bfd', 
                        help='Root of the transformer model'
                    )
    # model selection
    parser.add_argument('--mode', type=str, default='combined', help='seq_only, stru_only, combined')
    parser.add_argument('--predictor_info', type=str, default='GatedFusion', help='CNN, GatedFusion')
    parser.add_argument('--SeqEncoderTrans', action='store_false', help='transformer encoder')
    parser.add_argument('--AF2_Info', type=str, default='Feature', help='3D_structure, Feature')
    parser.add_argument('--dataset_name', type=str, default='10X', help='assign the dataset_name for fast pairing the obtained structure features')
    parser.add_argument('--model_seed', type=int, default=42, help='model seed')

    args = parser.parse_args()      

    input_data_file = args.input_data_file
    result_folder = args.result_folder
    AF2_feature_folder = args.AF2_feature_folder
    transformer_model_folder = args.transformer_model_folder
    model_seed = args.model_seed    
    SeqEncoderTrans = args.SeqEncoderTrans
    AF2_Info = args.AF2_Info

    # setting network seeds
    model_seed_set(model_seed = model_seed)

    # seq_transformer_info
    if SeqEncoderTrans:
        #---------------------------------------------------#
        transformer_tokenizer_name = os.path.abspath(transformer_model_folder)
        transformer_model_name = os.path.abspath(transformer_model_folder)    
        transformer_tokenizer = BertTokenizer.from_pretrained(transformer_tokenizer_name, do_lower_case=False)
        SeqInforModel = TFBertModel.from_pretrained(transformer_model_name, from_pt=False)
        SeqInforModel.trainable = False
        #---------------------------------------------------#    
    else:
        transformer_tokenizer = None
        SeqInforModel = None
    
    seq_transformer_info = dict()
    seq_transformer_info['whether_use_transformer'] = SeqEncoderTrans
    seq_transformer_info['tokenizer'] = transformer_tokenizer
    seq_transformer_info['tokenizer_fea_len'] = 40
    seq_transformer_info['SeqInforModel'] = SeqInforModel
    seq_transformer_info['transformer_feature'] = 'pooler_output' # 'pooler_output', 'last_hidden_state'
    seq_transformer_info['Transformer_fea_len'] = 1024

    if AF2_Info == 'Feature':
        # AF2 Features
        AF2_Feature_Info = dict()
        AF2_Feature_Info['seq_max_len'] = 40
        AF2_Feature_Info['fea_dim'] = 384
        AF2_Feature_Info['feature_file'] = AF2_feature_folder
    else:
        AF2_Feature_Info = None

    if not isinstance (args.epitope, list):
        selected_epitope = [args.epitope]
    else:
        selected_epitope = args.epitope
    print('-'*30)
    print(selected_epitope)    
    
    output_value = predict(input_data_file, 
                            transformer_model_folder, 
                            seq_transformer_info, # encoder sequence
                            AF2_Feature_Info, # encoder structure
                            selected_epitope = selected_epitope, 
                            output_folder= result_folder,
                            task_name=args.dataset_name
                    ) #'TRB_v_gene','TRB_j_gene','TRA_v_gene','TRA_j_gene','TRB_cdr3','TRA_cdr3',
 
#%%
if __name__ == '__main__':  
    main()