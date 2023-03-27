import os
import sys
import logging
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

from sklearn import metrics

sys.path.insert(0, os.path.abspath('../DeepAIR'))

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from transformers import TFBertModel, BertTokenizer

from config import DeepAIR_MIL_saved_model_dict
from deepair.modelling.classification import SeqClassificationModelWithProcessor
from deepair.utility.utility import df_to_input_feature_extraction
from deepair.utility.utility import violinplot, MIL_head
from deepair.utility.utility import seq_modafication, generate_AF2_feature_on_the_fly
from DeepAIR_Transformer_Feature_Extraction import generate_transformer_feature_on_the_fly

#%%
def model_seed_set(model_seed=13):
    os.environ['PYTHONHASHSEED']=str(model_seed)
    np.random.seed(model_seed)
    tf.random.set_seed(model_seed)
    
def map_probability_to_prediction(input, threshold):
    output = list()
    N, M = np.shape(input)
    for i in range(N):
        output.append(int(input[i,0]>=threshold))
    return output    

def select_one_donor(DF, DonorTitle, test_donor, label ='labels'):
    Test_DF = DF[DF[DonorTitle]==test_donor]
    Test_Label = Test_DF[label].to_list()
    return Test_DF, Test_Label

def output_mean_max_min(preds):
    preds_flutten = np.array(preds).flatten()
    return np.mean(preds_flutten), np.max(preds_flutten), np.min(preds_flutten)

def predict(input_data_file, 
            transformer_model_folder,
            seq_transformer_info,
            AF2_Feature_Info, 
            selected_disease = None, 
            output_folder = None,
            task_name = None,
            ):
    
    # print('-'*30)
    # print(f'current epitope is {epitope}')
    
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    df = pd.read_csv(input_data_file)
    if 'Heavy_cdr3' in list(df.columns):
        cell_type = 'BCELL'
        input_df=df.rename(columns={'Heavy_cdr3':'TRB_cdr3',
                            'Heavy_v_gene':'TRB_v_gene',
                            'Heavy_j_gene':'TRB_j_gene',
                            'Light_cdr3':'TRA_cdr3',
                            'Light_v_gene':'TRA_v_gene',
                            'Light_j_gene':'TRA_j_gene',
                            })
    else:
        cell_type = 'TCELL'
        input_df=df
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
                                                                                          ID = 'ID'
                                                                                    )

    test_input = df_to_input_feature_extraction(input_df)
    test_input['TRB_cdr3_splited'] = seq_trans_beta_feature
    test_input['TRA_cdr3_splited'] = seq_trans_alpha_feature
    test_input['TRB_cdr3_Stru'] = AF2_feature_beta_feature
    test_input['TRA_cdr3_Stru'] = AF2_feature_alpha_feature    
           
    output_df = pd.DataFrame()
    if cell_type == 'TCELL':
        output_df['ID'] = test_input['ID']
        output_df['TRB_cdr3'] = test_input['TRB_cdr3']
        output_df['TRA_cdr3'] = test_input['TRA_cdr3']
        output_df['TRB_v_gene'] = test_input['TRB_v_gene']
        output_df['TRB_j_gene'] = test_input['TRB_j_gene']
        output_df['TRA_v_gene'] = test_input['TRA_v_gene']
        output_df['TRA_j_gene'] = test_input['TRA_j_gene']
    else:
        output_df['ID'] = test_input['ID']
        output_df['Heavy_cdr3'] = test_input['TRB_cdr3']
        output_df['Light_cdr3'] = test_input['TRA_cdr3']
        output_df['Heavy_v_gene'] = test_input['TRB_v_gene']
        output_df['Heavy_j_gene'] = test_input['TRB_j_gene']
        output_df['Light_v_gene'] = test_input['TRA_v_gene']
        output_df['Light_j_gene'] = test_input['TRA_j_gene']
             
    model_save_path = DeepAIR_MIL_saved_model_dict[selected_disease]
    
    # now load the saved model, from (tempoarary) file, into notebook
    loaded_model = SeqClassificationModelWithProcessor.from_file(model_save_path)
    # print('1.-'*30)
    # print(tf.executing_eagerly()) 
    
    # # check the model ROC is the same as before
    preds = loaded_model.run(test_input)
    if tf.is_tensor(preds):
        preds = preds.numpy()     
       
    output_df[selected_disease+'_prob'] = preds
    
    if output_folder:
        output_df.to_csv(os.path.join(output_folder,'prediction_receptors.csv'))
        violin_plot_path = os.path.join(output_folder,'violin_plot_figure.png')
    input_DF = pd.DataFrame()
    input_DF[selected_disease+'_prob'] = output_df[selected_disease+'_prob']
    
    if output_folder:
        print(f'The violin plot of receptor level predictions is given: {violin_plot_path}')
        violinplot(input_DF, save_path=violin_plot_path)
    
    mil_output_df = pd.DataFrame()
    mil_output_df['Prediction Target'] = [selected_disease]
    
    MIL_prediciton = MIL_head(input=input_DF[selected_disease+'_prob'].to_numpy(), MIL_Method = 'majority_voting')
    print(f'The repertoire-level prediction based on the majority voting strategy is {MIL_prediciton}')
    mil_output_df['Prediction (MIL)'] = [MIL_prediciton]
    
    mil_output_df.to_csv(os.path.join(output_folder,'repertoire-level_prediction.csv'))
    
    return mil_output_df, output_df

def main():

    parser = argparse.ArgumentParser(description='DeepAIR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # data options
    parser.add_argument('--input_data_file', 
                        type=str, 
                        default='../DeepAIR/sampledata/MIL/IBD_BCR/GSM3576433_IBD_BCR.csv',
                        help='path to the input dataframe')
    parser.add_argument('--result_folder', 
                        type=str, 
                        default='../DeepAIR/result_MIL', 
                        help='folder to save the results')
    parser.add_argument('--AF2_feature_folder', 
                        type=str, 
                        default='../DeepAIR/sampledata/structure_feature/MIL',
                        help='AF2 feature file name')
    parser.add_argument('--transformer_model_folder', 
                        type=str, 
                        default='../DeepAIR/ProtTrans/prot_bert_bfd', 
                        help='Root of the transformer model'
                        )
    parser.add_argument('--task', 
                        type=str, 
                        default='IBD_BCR', # should be 'IBD_BCR','IBD_TCR','NPC_BCR','NPC_TCR'
                        help='The evaluated disease type'
                        )
    # model selection
    parser.add_argument('--mode', type=str, default='combined', help='seq_only, stru_only, combined')
    parser.add_argument('--predictor_info', type=str, default='GatedFusion', help='CNN, GatedFusion')
    parser.add_argument('--SeqEncoderTrans', action='store_false', help='transformer encoder')
    parser.add_argument('--AF2_Info', type=str, default='Feature', help='3D_structure, Feature')
    
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
    
    assert args.task in DeepAIR_MIL_saved_model_dict.keys()   
    desease_to_task_dict = dict()
    desease_to_task_dict['IBD_BCR']='IBD-BCR-R'
    desease_to_task_dict['IBD_TCR']='IBD-TCR-R'
    desease_to_task_dict['NPC_BCR']='NPC-BCR'
    desease_to_task_dict['NPC_TCR']='NPC-TCR'
    mil_output_df, output_df = predict(input_data_file, 
                                        transformer_model_folder, 
                                        seq_transformer_info, # encoder sequence
                                        AF2_Feature_Info, # encoder structure
                                        selected_disease = args.task,
                                        output_folder= result_folder,
                                        task_name = desease_to_task_dict[args.task]
                                    )

#%%
if __name__ == '__main__':  
    main()