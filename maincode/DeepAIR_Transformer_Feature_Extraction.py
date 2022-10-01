import os
import sys
import h5py
# from turtle

import logging
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.abspath('../DeepAIR'))

from transformers import TFBertModel, BertTokenizer

from deepair.modelling.processing import AASeqProcessorTransformer
from deepair.modelling import extractors
from deepair.modelling.feature_extract import FeatureExtractiongModel
from deepair.utility.utility import df_to_input_feature_extraction, seq_modafication

#%%
def get_current_time():
    now = datetime.now()
    # current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    current_time = now.strftime("%Y-%m-%d")
    return current_time

def model_seed_set(model_seed=13):
    os.environ['PYTHONHASHSEED']=str(model_seed)
    np.random.seed(model_seed)
    tf.random.set_seed(model_seed)

def set_logger(OutputFilePath):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(OutputFilePath)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def main_model_prepare(seq_transformer_info, max_len_cdr3 = 40):
    # ------------------------------------------------------------------------
    # sequence
    processor_b = AASeqProcessorTransformer(tokenizer = seq_transformer_info['tokenizer'], 
                                            max_len = max_len_cdr3, 
                                            add_special_tokens=True, 
                                            padding = 'max_length', 
                                            truncation = True, 
                                            return_tensors = 'np',
                                            isIDOnly = True
                                            )
    processor_a = AASeqProcessorTransformer(tokenizer = seq_transformer_info['tokenizer'], 
                                            max_len = max_len_cdr3, 
                                            add_special_tokens=True, 
                                            padding = 'max_length', 
                                            truncation = True, 
                                            return_tensors = 'np',
                                            isIDOnly=True
                                            )
    seq_processors = {'TRB_cdr3_splited': processor_b,'TRA_cdr3_splited': processor_a}
       
    # ------------------------------------------------------------------------
    # sequence encoder
    if seq_transformer_info['whether_use_transformer']:
        # sequence model
        seq_encoder_params = {
            'filters': [256],
            'kernel_widths': [5,],
            'dilations': [1],
            'strides': [1],
            'L2_conv': 0.01,
            'dropout_conv': 0.3,
        }

        seq_encoders = {
            'TRB_cdr3_splited': extractors.conv_seq_transformer_extractor(seq_transformer_info,
                                                                    seq_encoder_params,
                                                                    seq_transformer_info['transformer_feature']),
            'TRA_cdr3_splited': extractors.conv_seq_transformer_extractor(seq_transformer_info,
                                                                    seq_encoder_params,
                                                                    seq_transformer_info['transformer_feature'])
        }

    # ------------------------------------------------------------------------
    # classification head 
    model = FeatureExtractiongModel(seq_processors = seq_processors, seq_encoders = seq_encoders)

    return model

def main_model_run(model, data_frame):
    
    df = data_frame
    # print('-'*30 + 'current dataframe' + '-'*30)
    # print(df)

    test_input = df_to_input_feature_extraction(df)

    # print('-'*30 + 'dataframe after dataloader' + '-'*30)
    # print(test_input)

    preds = model.run(test_input)    

    # print('-'*30 + 'prediction after model' + '-'*30)
    # print(preds)
    return preds

def generate_transformer_feature_on_the_fly(input_data_frame,
                                            transformer_model_folder,
                                            seq_transformer_info = None,
                                            ID = 'ID', 
                                            CDR3b = 'TRB_cdr3', 
                                            CDR3a = 'TRA_cdr3', 
                                            model_seed = 32
        ):

    # setting network seeds
    model_seed_set(model_seed=model_seed)

    #---------------------------------------------------#
    transformer_tokenizer_name = os.path.abspath(transformer_model_folder)
    transformer_model_name = os.path.abspath(transformer_model_folder)    
    transformer_tokenizer = BertTokenizer.from_pretrained(transformer_tokenizer_name, do_lower_case=False)
    SeqInforModel = TFBertModel.from_pretrained(transformer_model_name, from_pt=False)
    SeqInforModel.trainable = False
    #---------------------------------------------------#    
    
    if not seq_transformer_info:                
        seq_transformer_info = dict()
        seq_transformer_info['whether_use_transformer'] = True
        seq_transformer_info['tokenizer'] = transformer_tokenizer
        seq_transformer_info['tokenizer_fea_len'] = 40
        seq_transformer_info['SeqInforModel'] = SeqInforModel
        seq_transformer_info['transformer_feature'] = 'pooler_output' # 'pooler_output', 'last_hidden_state'
        seq_transformer_info['Transformer_fea_len'] = 1024

    model = main_model_prepare(seq_transformer_info, max_len_cdr3 = 40)
    if isinstance(input_data_frame[CDR3b], str):
        input_data_frame[CDR3b+'_splited'] = seq_modafication(input_data_frame[CDR3b])
    else:
        input_data_frame[CDR3b+'_splited'] = input_data_frame[CDR3b].apply(seq_modafication)
    
    if isinstance(input_data_frame[CDR3a], str):
        input_data_frame[CDR3a+'_splited'] = seq_modafication(input_data_frame[CDR3a])
    else:
        input_data_frame[CDR3a+'_splited'] = input_data_frame[CDR3a].apply(seq_modafication)

    data_num = len(input_data_frame)
    transformer_fea_dim = seq_transformer_info['Transformer_fea_len']
    
    beta_feature = np.zeros((data_num, 1, transformer_fea_dim))
    alpha_feature = np.zeros((data_num, 1, transformer_fea_dim))

    for index in range(data_num):
        
        row = input_data_frame[index:(index+1)]
        
        print('-'*30)
        print(f'{index}_{row[CDR3b].values[0]}')
        print('-'*30)
        
        preds = main_model_run(model, row)
        
        # B chain
        CDR3b_Feature = preds[CDR3b+'_splited'].numpy()
        beta_feature[index] = CDR3b_Feature
        
        # A chain
        CDR3a_Feature = preds[CDR3a+'_splited'].numpy()
        alpha_feature[index] = CDR3a_Feature
        
    return beta_feature, alpha_feature

