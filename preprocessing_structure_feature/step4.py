import os
import pandas as pd
import numpy as np
import pickle
import argparse

#%%
def parse_pdb_csv_to_numpy(input_file):
    pdb_df = pd.read_csv(input_file)
    pdb_numpy = pdb_df.loc[:,['x','y','z','occupancy','temperature_factor']].to_numpy()
    return pdb_numpy

def get_cdr3_pdb_file(pdb_file_path, 
                     output_full_path, 
                     seq_begin, 
                     seq_end, 
                     output_CDR3_path,
                     output_CDR3_pdb_file
                     ):
    """
    Returns a list of the atom coordinates, and their properties in a pdb file
    :param pdb_file_path:
    :return:
    """
    with open(pdb_file_path, 'r') as f:
        lines = [line for line in f.readlines() if 'ATOM' in line]
    column_names = ['atom_num', 'atom_name', 'alternate_location_indicator',
                    'residue_name', 'chain_id', 'residue_num',
                    'code_for_insertions_of_residues', 'x', 'y', 'z', 'occupancy',
                    'temperature_factor', 'segment_identifier', 'element_symbol']
    # Get the index at which each column starts/ends
    column_ends = np.array([3, 10, 15, 16, 19, 21, 25, 26, 37, 45, 53, 59, 65, 75, 77])
    column_starts = column_ends[:-1] + 1
    column_ends = column_ends[1:]  # Ignore the first column (just says 'ATOM')
    # start: array([ 4, 11, 16, 17, 20, 22, 26, 27, 38, 46, 54, 60, 66, 76])
    # end: array([10, 15, 16, 19, 21, 25, 26, 37, 45, 53, 59, 65, 75, 77])

    rows = [[l[start:end+1].replace(' ', '') for start, end in zip(column_starts, column_ends)]
            for l in lines]
    Threed_structure_output = pd.DataFrame(rows, columns=column_names)

    residue_num_list = Threed_structure_output['residue_num'].to_list()
    atom_begin = residue_num_list.index(str(seq_begin+1))
    atom_end = residue_num_list.index(str(seq_end+1))
    atom_len = atom_end - atom_begin
    feature_cdr3 = Threed_structure_output[atom_begin:atom_end]
    
    if output_full_path:
        Threed_structure_output.to_csv(output_full_path, index=False)
    
    if output_CDR3_path:
        feature_cdr3.to_csv(output_CDR3_path, index=False)

    if output_CDR3_pdb_file:
        os.makedirs(os.path.dirname(output_CDR3_pdb_file), exist_ok=True)
        # generate output pdb file
        output_pdb_lines = lines[atom_begin:atom_end]
        atom_num = Threed_structure_output[atom_end:atom_end+1]['atom_num'].values[0]
        residue_name = Threed_structure_output[atom_end-1:atom_end]['residue_name'].values[0]
        chain_identifier = Threed_structure_output[atom_end-1:atom_end]['chain_id'].values[0]
        residue_sequence_number = Threed_structure_output[atom_end-1:atom_end]['residue_num'].values[0]
        output_pdb_lines.append('TER'+' '*3 + atom_num.rjust(5) +' '*6 + residue_name.rjust(3) +' {}'.format(chain_identifier) + residue_sequence_number.rjust(4) +'\n')
        output_pdb_lines.append('END\n')

        # write context to pdb file
        with open(output_CDR3_pdb_file, 'w+') as file_store:
            for line in output_pdb_lines:
                    file_store.write(line)

    return atom_begin, atom_end, atom_len


def parse_AF2_data_and_obtain_maxlen(table_file_path, 
                                    AF2_feature_folder_list, 
                                    output_folder, 
                                    ID = 'ID', 
                                    cdr3b = 'TRB_cdr3', 
                                    cdr3a = 'TRA_cdr3', 
                                    b_vgene = "TRB_v_gene",
                                    b_jgene = "TRB_j_gene",
                                    a_vgene = "TRA_v_gene",
                                    a_jgene = "TRA_j_gene",
                                    cdr3b_begin = 'TRB_cdr3_begin', 
                                    cdr3b_end = 'TRB_cdr3_end',
                                    cdr3a_begin = 'TRA_cdr3_begin',
                                    cdr3a_end = 'TRA_cdr3_end',
                                    af_pdb_file = 'ranked_0.pdb'
                                    ):
    
    atom_begin_b_list = list() 
    atom_end_b_list = list() 
    atom_len_b_list = list() 
    atom_begin_a_list = list() 
    atom_end_a_list = list() 
    atom_len_a_list = list() 

    AIR_df = pd.read_csv(table_file_path)

    for index, row in AIR_df.iterrows():
        print('-'*30)
        print(f'{index}_{row[cdr3b]}_{row[cdr3a]}')
        print('-'*30)
        current_CDR3b_ID_note = f'{row[cdr3b]}_{row[ID]}'
        current_CDR3a_ID_note = f'{row[cdr3a]}_{row[ID]}'
        
        current_cdr3b_begin = row[cdr3b_begin]
        current_cdr3b_end = row[cdr3b_end]
        current_cdr3a_begin = row[cdr3a_begin]
        current_cdr3a_end = row[cdr3a_end]

        # B chain
        for AF2_feature_folder in AF2_feature_folder_list:
            
            
            pdb_file_path = os.path.join(AF2_feature_folder, current_CDR3b_ID_note, af_pdb_file)
            
            if os.path.exists(pdb_file_path):
                output_file_path_full = None
                output_file_path_cdr3 = None
                output_file_path_cdr3_pdb = os.path.join(output_folder, f'{current_CDR3b_ID_note}_relaxed_cdr3.pdb')
                atom_begin_b, atom_end_b, atom_len_b = get_cdr3_pdb_file(pdb_file_path, 
                                                                    output_file_path_full, 
                                                                    current_cdr3b_begin, 
                                                                    current_cdr3b_end, 
                                                                    output_file_path_cdr3,
                                                                    output_file_path_cdr3_pdb
                                                                    )
                atom_begin_b_list.append(atom_begin_b) 
                atom_end_b_list.append(atom_end_b) 
                atom_len_b_list.append(atom_len_b)
        
            # A chain
            pdb_file_path = os.path.join(AF2_feature_folder, current_CDR3a_ID_note, af_pdb_file)
           
            if os.path.exists(pdb_file_path):
                output_file_path_full = None
                output_file_path_cdr3 = None
                output_file_path_cdr3_pdb = os.path.join(output_folder, f'{current_CDR3a_ID_note}_relaxed_model_cdr3.pdb')
                atom_begin_a, atom_end_a, atom_len_a = get_cdr3_pdb_file(pdb_file_path, 
                                                                    output_file_path_full, 
                                                                    current_cdr3a_begin, 
                                                                    current_cdr3a_end, 
                                                                    output_file_path_cdr3,
                                                                    output_file_path_cdr3_pdb
                                                                    )
                atom_begin_a_list.append(atom_begin_a) 
                atom_end_a_list.append(atom_end_a)
                atom_len_a_list.append(atom_len_a) 

    AIR_df['atom_begin_b'] = atom_begin_b_list
    AIR_df['atom_end_b'] = atom_end_b_list
    AIR_df['atom_len_b'] = atom_len_b_list

    AIR_df['atom_begin_a'] = atom_begin_a_list
    AIR_df['atom_end_a'] = atom_end_a_list
    AIR_df['atom_len_a'] = atom_len_a_list

    max_b_length = np.max(np.array(atom_len_b_list))
    max_a_length = np.max(np.array(atom_len_a_list))
    
    root, ext = os.path.splitext(table_file_path)
    bname = os.path.basename(root)
    output_file = os.path.join(output_folder, f'{bname}_MaxAtomLenB-{max_b_length}_MaxAtomLenA-{max_a_length}.csv')
    AIR_df.to_csv(output_file,index=False)
    return output_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepAIR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # data options
    parser.add_argument('--AF2_feature_folder', 
                        type=str, 
                        default='./output',
                        help='path to the AF2 feature folder list'
                    )
    parser.add_argument('--table_file_path',
                        type=str,
                        default='./AB_Combine_Drop_Duplicates_CDR3Region_example.csv',
                        help='path to the table file'
                    )

    parser.add_argument('--output_folder', 
                        type=str, 
                        default='./data_AB_COVID19_withPDB',
                        help='path to the output folder'
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


    AF2_feature_folder_list = list()
    AF2_feature_folder_list.append(args.AF2_feature_folder)
    table_file_path = args.table_file_path
    output_folder = args.output_folder
    ID = args.ID
    Chain1_cdr3 = args.Chain1_cdr3
    Chain2_cdr3 = args.Chain2_cdr3
    Chain1_v_gene = args.Chain1_v_gene
    Chain1_j_gene = args.Chain1_j_gene
    Chain2_v_gene = args.Chain2_v_gene
    Chain2_j_gene = args.Chain2_j_gene
    vdj_file_path = parse_AF2_data_and_obtain_maxlen(table_file_path, 
                                                    AF2_feature_folder_list, 
                                                    output_folder, 
                                                    ID = ID,
                                                    cdr3b = Chain1_cdr3, 
                                                    cdr3a = Chain2_cdr3,
                                                    b_vgene = Chain1_v_gene,
                                                    b_jgene = Chain1_j_gene,
                                                    a_vgene = Chain2_v_gene,
                                                    a_jgene = Chain2_j_gene,
                                                    cdr3b_begin = f'{Chain1_cdr3}_begin', 
                                                    cdr3b_end = f'{Chain1_cdr3}_end',
                                                    cdr3a_begin = f'{Chain2_cdr3}_begin',
                                                    cdr3a_end = f'{Chain2_cdr3}_end',
                                                )