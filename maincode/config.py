import os

# DeepAIR Model
MODEL_SAVE_PATH = os.path.abspath('../checkpoints')

DeepAIR_BRP_saved_model_dict = dict()
DeepAIR_BRP_saved_model_dict['A1101_AVFDRKSDAK_EBNA-3B_EBV'] = os.path.join(MODEL_SAVE_PATH, 'BRP/A1101_AVFDRKSDAK_EBNA-3B_EBV_binder_model')
DeepAIR_BRP_saved_model_dict['A0201_GILGFVFTL_Flu-MP_Influenza'] = os.path.join(MODEL_SAVE_PATH, 'BRP/A0201_GILGFVFTL_Flu-MP_Influenza_binder_model')
DeepAIR_BRP_saved_model_dict['A1101_IVTDFSVIK_EBNA-3B_EBV'] = os.path.join(MODEL_SAVE_PATH, 'BRP/A1101_IVTDFSVIK_EBNA-3B_EBV_binder_model')
DeepAIR_BRP_saved_model_dict['B0801_RAKFKQLL_BZLF1_EBV'] = os.path.join(MODEL_SAVE_PATH, 'BRP/B0801_RAKFKQLL_BZLF1_EBV_binder_model')
DeepAIR_BRP_saved_model_dict['A0201_GLCTLVAML_BMLF1_EBV'] = os.path.join(MODEL_SAVE_PATH, 'BRP/A0201_GLCTLVAML_BMLF1_EBV_binder_model')
DeepAIR_BRP_saved_model_dict['A0201_ELAGIGILTV_MART-1_Cancer'] = os.path.join(MODEL_SAVE_PATH, 'BRP/A0201_ELAGIGILTV_MART-1_Cancer_binder_model')
DeepAIR_BRP_saved_model_dict['A0301_KLGGALQAK_IE-1_CMV'] = os.path.join(MODEL_SAVE_PATH, 'BRP/A0301_KLGGALQAK_IE-1_CMV_binder_model')
DeepAIR_BRP_saved_model_dict['LTDEMIAQY'] = os.path.join(MODEL_SAVE_PATH, 'BRP/LTDEMIAQY_model')
DeepAIR_BRP_saved_model_dict['TTDPSFLGRY'] = os.path.join(MODEL_SAVE_PATH, 'BRP/TTDPSFLGRY_model')
DeepAIR_BRP_saved_model_dict['YLQPRTFLL'] = os.path.join(MODEL_SAVE_PATH, 'BRP/YLQPRTFLL_model')

DeepAIR_BAP_saved_model_dict = dict()
DeepAIR_BAP_saved_model_dict['A1101_AVFDRKSDAK_EBNA-3B_EBV'] = os.path.join(MODEL_SAVE_PATH, 'BAP/A1101_AVFDRKSDAK_EBNA-3B_EBV_Reg_model')
DeepAIR_BAP_saved_model_dict['A0201_GILGFVFTL_Flu-MP_Influenza'] = os.path.join(MODEL_SAVE_PATH, 'BAP/A0201_GILGFVFTL_Flu-MP_Influenza_Reg_model')
DeepAIR_BAP_saved_model_dict['A1101_IVTDFSVIK_EBNA-3B_EBV'] = os.path.join(MODEL_SAVE_PATH, 'BAP/A1101_IVTDFSVIK_EBNA-3B_EBV_Reg_model')
DeepAIR_BAP_saved_model_dict['B0801_RAKFKQLL_BZLF1_EBV'] = os.path.join(MODEL_SAVE_PATH, 'BAP/B0801_RAKFKQLL_BZLF1_EBV_Reg_model')
DeepAIR_BAP_saved_model_dict['A0201_GLCTLVAML_BMLF1_EBV'] = os.path.join(MODEL_SAVE_PATH, 'BAP/A0201_GLCTLVAML_BMLF1_EBV_Reg_model')
DeepAIR_BAP_saved_model_dict['A0201_ELAGIGILTV_MART-1_Cancer'] = os.path.join(MODEL_SAVE_PATH, 'BAP/A0201_ELAGIGILTV_MART-1_Cancer_Reg_model')
DeepAIR_BAP_saved_model_dict['A0301_KLGGALQAK_IE-1_CMV'] = os.path.join(MODEL_SAVE_PATH, 'BAP/A0301_KLGGALQAK_IE-1_CMV_Reg_model')

# DeepAIR_MIL Model
DeepAIR_MIL_saved_model_dict = dict()
DeepAIR_MIL_saved_model_dict['IBD_BCR'] = os.path.join(MODEL_SAVE_PATH, 'MIL/IBD_BCR/IBD_BCR_model')
DeepAIR_MIL_saved_model_dict['IBD_TCR'] = os.path.join(MODEL_SAVE_PATH, 'MIL/IBD_TCR/IBD_TCR_model')
DeepAIR_MIL_saved_model_dict['NPC_BCR'] = os.path.join(MODEL_SAVE_PATH, 'MIL/NPC_BCR/NPC_BCR_model')
DeepAIR_MIL_saved_model_dict['NPC_TCR'] = os.path.join(MODEL_SAVE_PATH, 'MIL/NPC_TCR/NPC_TCR_model')