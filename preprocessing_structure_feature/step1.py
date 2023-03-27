import os
import pandas as pd
import numpy as np
import argparse

#%%
def get_the_index_of_the_seq(source, target):
    tar_len = len(target)
    if (source is None or target is None):
        return -1
    for i in range(len(source) - len(target) + 1):
        if target == source[i:i+tar_len]:
            return i, i+tar_len
        else:
            pass

def identify_begin_and_end_of_each_CDR3(vdj_file_path, 
                                        ID = None,
                                        cdr3b = None, 
                                        cdr3a = None, 
                                        seqb = None, 
                                        seqa = None, 
                                        output_table = None):
    DF = pd.read_csv(vdj_file_path)
    cdr3b_begin_list = list()
    cdr3b_end_list = list()
    cdr3a_begin_list = list()
    cdr3a_end_list = list()
    index_list = list()
    index_DF = pd.DataFrame()
    
    for index, row in DF.iterrows():
        print(index)
        try:   
            cdr3b_begin, cdr3b_end = get_the_index_of_the_seq(source = row[seqb], target = row[cdr3b]) 
            cdr3a_begin, cdr3a_end = get_the_index_of_the_seq(source = row[seqa], target = row[cdr3a]) 
            cdr3b_begin_list.append(cdr3b_begin)
            cdr3b_end_list.append(cdr3b_end)
            cdr3a_begin_list.append(cdr3a_begin)
            cdr3a_end_list.append(cdr3a_end)
            index_list.append(row[ID])
        except:
            print('-'*10+'Error'+'-'*10)
            print(row[ID])
            print(row[seqb])
            print(row[cdr3b])
            print(row[seqa])
            print(row[cdr3a])    
    
    index_DF[ID] = index_list                
    index_DF[cdr3a+'_begin'] = cdr3a_begin_list
    index_DF[cdr3a+'_end'] = cdr3a_end_list
    index_DF[cdr3b+'_begin'] = cdr3b_begin_list
    index_DF[cdr3b+'_end'] = cdr3b_end_list    
    Output_DF = DF.merge(index_DF, on=ID, how='inner')
    
    Output_DF.to_csv(output_table, index=False)

## make the DF light/alpha and heavy/beta to fasta file
def get_fasta_file_from_dataframe(DF,
                                  seqb = None,
                                  seqa = None,
                                  seqa_id = None,
                                  seqb_id = None, 
                                  output_fasta_path = None):
    
    seqb_list = DF[seqb].to_list()
    seqa_list = DF[seqa].to_list()  

    for i in range(len(DF)):
        # write the seqa fasta file
        with open(output_fasta_path+seqa_id+'.fasta', 'a') as f:
            f.write('>'+DF.loc[i , seqa_id]  +'\n'+seqa_list[i]+'\n')
        # write the seqb fasta file
        with open(output_fasta_path+seqb_id+'.fasta', 'a') as f:
            f.write('>'+DF.loc[i , seqb_id]  +'\n'+seqb_list[i]+'\n')


## make the light/alpha and heavy/beta to fasta file for input of the AF2
def get_fasta_file(df_path,
                   ID = None,
                   cdr3b = None, 
                   cdr3a = None, 
                   seqb = None, 
                   seqa = None,
                   output_fasta_path = None):
    
    DF = pd.read_csv(df_path)
    seqb_list = DF[seqb].to_list()
    seqa_list = DF[seqa].to_list()
   
    # make the output fasta file path
    os.makedirs(os.path.join(output_fasta_path),exist_ok = True)
    os.makedirs(os.path.join(output_fasta_path),exist_ok = True)

    for i in range(len(DF)):
        output_CDR3b_ID_note = f'{DF.loc[i , cdr3b]}_{DF.loc[i , ID]}'
        output_CDR3a_ID_note = f'{DF.loc[i , cdr3a]}_{DF.loc[i , ID]}'
        # write the seqa fasta file
        with open(os.path.join(output_fasta_path ,  output_CDR3a_ID_note+".fasta"), 'a') as f:
            f.write('>'+output_CDR3a_ID_note  +'\n'+seqa_list[i]+'\n')
        # write the seqb fasta file
        with open(os.path.join(output_fasta_path ,  output_CDR3b_ID_note+".fasta"), 'a') as f:
            f.write('>'+output_CDR3b_ID_note  +'\n'+seqb_list[i]+'\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepAIR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # data options
    parser.add_argument('--AIR_file_path', 
                        type=str, 
                        default='./sampledata/CoV-AbDab_example.csv',
                        help='path to the input AIR file'
                    )
    parser.add_argument('--output_table', 
                        type=str, 
                        default='./sampledata/CoV-AbDab_example_CDR3Region.csv', 
                        help='folder to save the output table'
                    )
    parser.add_argument('--output_fasta_folder', 
                        type=str, 
                        default='./sampledata/fasta', 
                        help='folder to save the output fasta files'
                    )
    parser.add_argument('--ID', type=str, default='ID', help='ID column name')
    parser.add_argument('--Chain1', type=str, default='Heavy', help='can be BCR Heavy chain/TCR Beta chain')
    parser.add_argument('--Chain2', type=str, default='Light', help='can be BCR Light chain/TCR Alpha chain')
    parser.add_argument('--Chain1_cdr3', type=str, default='Heavy_cdr3', help='The cdr3 region of BCR Heavy chain/TCR Beta chain')
    parser.add_argument('--Chain2_cdr3', type=str, default='Light_cdr3', help='The cdr3 region of BCR Light chain/TCR Alpha chain')
    args = parser.parse_args()   


    AIR_file_path = args.AIR_file_path
    output_table = args.output_table
    output_fasta_folder = args.output_fasta_folder
    ID = args.ID
    Chain1 = args.Chain1
    Chain2 = args.Chain2
    Chain1_cdr3 = args.Chain1_cdr3
    Chain2_cdr3 = args.Chain2_cdr3
    
    identify_begin_and_end_of_each_CDR3(AIR_file_path, 
                                        ID = ID,
                                        cdr3b = Chain1_cdr3, 
                                        cdr3a = Chain2_cdr3, 
                                        seqb = Chain1, 
                                        seqa = Chain2, 
                                        output_table = output_table
                                        )
    
    get_fasta_file(df_path = output_table,
                    ID = ID,
                    cdr3b = Chain1_cdr3, 
                    cdr3a = Chain2_cdr3, 
                    seqa = Chain2,
                    seqb = Chain1,
                    output_fasta_path = args.output_fasta_folder
                    )
    

    