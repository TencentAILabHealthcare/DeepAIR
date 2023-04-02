# DeepAIR

[![python >3.8.13](https://img.shields.io/badge/python-3.8.13-brightgreen)](https://www.python.org/) 

### DeepAIR: a deep-learning framework for effective integration of sequence and 3D structure to enable adaptive immune receptor analysis
Structural-docking-based binding between the adaptive immune receptors (AIRs), including T cell receptor (TCR) and B cell receptor (BCR), and the antigens is one of the most fundamental processes in adaptive immunity. However, current methods for predicting AIR-antigen binding largely rely on sequence-derived features of AIR. In this study, we present a deep-learning framework, termed DeepAIR, for the accurate prediction of AIR-antigen binding by integrating both sequence-derived and structure-derived features of AIRs. DeepAIR consists of three feature encoders, including a trainable-embedding-layer-based gene encoder, a transformer-based sequence encoder, and a pre-trained AlphaFold2-based structure encoder. DeepAIR deploys a gating-based attention mechanism to extract important features from the three encoders, and a tensor fusion mechanism to integrate obtained features for multiple tasks, including the prediction of AIR-antigen binding affinity, AIR-antigen binding reactivity, and the classification of the immune repertoire. We systematically evaluated the performance of DeepAIR on multiple datasets. DeepAIR shows outstanding prediction performance in terms of AUC (area under the ROC curve) in predicting the binding reactivity to various antigens, as well as the classification of immune repertoire for nasopharyngeal carcinoma (NPC) and inflammatory bowel disease (IBD). We anticipate that DeepAIR can serve as a useful tool for characterizing and profiling antigen-binding AIRs, thereby informing the design of personalized immunotherapy.

![avatar](./figure/Figure1.png)

<center>Flowchart of DeepAIR. DeepAIR has three major processing stages, including multi-channel feature extraction, multimodal feature fusion, and task-specific prediction. At the multi-channel feature extraction stage, three feature encoders are involved and used to extract informative features from the gene, sequence, and structure inputs. Then the resulting features produced by three different encoders are further integrated via a gating-based attention mechanism as well as the tensor fusion at the multimodal feature fusion stage to generate a comprehensive representation. Finally, at the task-specific prediction stage, specifically designed prediction layers are utilized to map the obtained representations to the output results. </center>

# System requirements
## Hardware requirements
`DeepAIR` package requires only a standard computer with enough RAM and a NVIDIA GPU to support operations.
## Software requirements
### OS requirements
This tool is supported for Linux. The tool has been tested on the following systems: <br>
+ CentOS Linux release 8.2.2.2004
+ Ubuntu 18.04.5 LTS
### Python dependencies
`DeepAIR` mainly depends on the Python scientific stack.   <br>

+ The important packages including:
```
    umap-learn                   0.5.1
    scikit-learn                 0.23.2
    tensorflow-gpu               2.7.0
    biopython                    1.76    
    huggingface-hub              0.2.1
    matplotlib                   3.5.1
    numpy                        1.19.5
    pandas                       1.4.2
    tokenizers                   0.12.1
    transformers                 4.19.4
```
+ Transformers is from HuggingFace Transformer (Transformer Version: 4.19.4):
    [https://huggingface.co/docs/transformers/installation]
+ `./DeepAIR/requirements.txt` describes more details of the requirements.    
### Pretrained model Requirements
Download `ProtBert-BFD` model

`ProtBert-BFD` is based on Bert model which pretrained on a large corpus of protein sequences in a self-supervised fashion, which is avaiable from [https://huggingface.co/Rostlab/prot_bert_bfd/tree/main]
    
    The downloaded model should be stored as:
    ./ProtTrans/prot_bert_bfd  \
        config.json
        special_tokens_map.json
        tf_model.h5
        tokenizer_config.json
        vocab.txt


# Structure information preprocessing 
DeepAIR utilizes AlphaFold2 to extract structure information (Structure information preprocessing). This initial structure features are then recalibrated in the structure encoder of DeepAIR to make them more suitable for binding affinity and reactivity prediction. The detailed steps are given in `./DeepAIR/preprocessing_structure_feature`, and a indenpendt README.md is also provided as:`./DeepAIR/preprocessing_structure_feature/ReadMe.md`

There are three main steps:

## step 1 
Generate the corresponding fasta file for each chain and identify the begining position and end position of the CDR3 region in each alpha/light and beta/heavy chains. 

```python
python step1.py \
    --AIR_file_path ./sampledata/CoV-AbDab_example.csv \ # path to the input AIR file
    --output_table ./sampledata/CoV-AbDab_example_CDR3Region.csv \ # path to save the output table
    --output_fasta_folder ./sampledata/fasta \ # folder to save the output fasta files
```
In step1, a fasta folder and a csv file containing CDR3Region information will be generated. Each chain is denoted with its CDR3 region and an unique ID (Can be assigned by the user).

## step 2 
First, we need to deploy alphafold2:

https://github.com/deepmind/alphafold

In order to get the representations from af2 in "features.pkl", you should add "return_representations=True" in line 79 and line 87 of ./alphafold/alphafold/model/model.py before building the image.

Then, set Data_dir in step2.sh to the path of the dataset you downloaded, and run step2.sh:
```bash
bash step2.sh
```
Then you can get the "features.pkl", "relaxed_model_1_ptm.pdb" and other files generated by af2 in the output folder.

## step 3 
Extract the structure feature of the CDR3 region of BCR heavy/light or TCR alpha/beta chains.
```python
python step3.py \
    --AIR_file_path ./sampledata/CoV-AbDab_example_CDR3Region.csv \ # path to the vdj file
    --AF2_feature_folder ./sampledata/output_af2_structure \ # path to the AF2 feature folder
    --output_folder ./sampledata/output_feature \ # path to the output table
```
At step 3, we can get the CDR3 structure feature files, which can be used and recalibated by DeepAIR.

# Install guide
## For docker users
 
### 1-Pull docker images from docker-hub.（optional）
```
docker pull deepair1/deepair:latest
```
+ If you don't want to show the username, do as follows:
```
docker tag deepair1/deepair:latest deepair:latest
docker rmi deepair1/deepair:latest
```

### 2-Download docker file `deepair.tar` from [GoogleDrive](https://drive.google.com/file/d/1-fahB823OZLwsqKu_8DRPMyOw6yYwpHl/view?usp=sharing) or [OneDrive](https://1drv.ms/u/s!Au2cJRs-_u93gvZeel2I1EmYt3F17A?e=lKYQHj).（optional）
+ Add docker image by loading the docker file:
```
docker load < deepair.tar
```

### 3-Start your docker and run examples:
```
docker run --name deepair --gpus all -it --rm  deepair:latest /bin/bash
```
+ If there are multiple GPUs in your device and you just want to use only one GPU:
```
docker run --name deepair --gpus '"device=0"' -it --rm  deepair:latest /bin/bash
```
+ Test directly because all files are contained in this docker image: 
```
python ./maincode/DeepAIR_BRP.py
python ./maincode/DeepAIR_BAP.py
python ./maincode/DeepAIR_MIL.py
```

## For conda users

### 1-Configure the enviroment.
```
git clone https://github.com/TencentAILabHealthcare/DeepAIR.git 
cd ./DeepAIR
conda create -n deepair python=3.8
conda activate deepair
pip install umap-learn==0.5.1 tensorflow-gpu==2.7.0 scikit-learn==0.23.2 biopython==1.76 huggingface-hub==0.2.1 matplotlib==3.5.1 numpy==1.19.5 pandas==1.4.2 tokenizer==1.0.0 transformers==4.19.4 umap-learn==0.5.1 seaborn==0.10.1 protobuf==3.20.3
conda install cuda -c nvidia
conda deactivate
```

### 2-Download pretrained model and test:
+ Download `config.json, special_tokens_map.json, tf_model.h5, tokenizer_config.json, vocab.txt` from [https://huggingface.co/Rostlab/prot_bert_bfd/tree/main] and save them in the folder `./DeepAIR/ProtTrans/prot_bert_bfd`.
+ Run commands in the folder `./DeepAIR`:
```
conda activate deepair 
CUDA_VISIBLE_DEVICES=0 python ./maincode/DeepAIR_BRP.py
CUDA_VISIBLE_DEVICES=0 python ./maincode/DeepAIR_BAP.py
CUDA_VISIBLE_DEVICES=0 python ./maincode/DeepAIR_MIL.py
```

# Config file

(1) Edite the `./maincode/config.py` file which provides the paths of the obtained DeepAIR models (A well-edited file is given as a default example.)

# Runing

(1) For binding reactivity prediciton (BRP) (Classification)

    python ./maincode/DeepAIR_BRP.py  \
        --input_data_file  \ # path to the input table 
        --result_folder  \ #  folder to save the results
        --epitope  \ # selected epitope for the evaluation, can be a epitope such as "--epitope A1101_AVFDRKSDAK_EBNA-3B_EBV" 
        --AF2_feature_folder  \ # AF2 feature folder
        --transformer_model_folder  \ # folder to save the pretrained BERT model 

(2) For binding affinity prediciton (BAP) (Regression)

    python ./maincode/DeepAIR_BAP.py 
        --input_data_file  \ # path to the input table 
        --result_folder  \ #  folder to save the results
        --epitope  \ # selected epitope for the evaluation, can be a epitope such as "--epitope A1101_AVFDRKSDAK_EBNA-3B_EBV" 
        --AF2_feature_folder  \ # AF2 feature folder
        --transformer_model_folder  \ # folder to save the pretrained BERT model

(3) For immune repertoire classification (Multiple instance learning (MIL))
    
    python ./maincode/DeepAIR_MIL.py 
        --input_data_file  \ # path to the input table (an immnue repertoire of a subject)
        --result_folder  \ #  folder to save the results
        --AF2_feature_folder  \ # AF2 feature folder
        --transformer_model_folder  \ # folder to save the pretrained BERT model
        --task \ # can be one of 'IBD_BCR' (inflammatory bowel disease (BCR)), 'IBD_TCR' (inflammatory bowel disease (TCR)), 'NPC_BCR' (nasopharyngeal carcinoma (BCR)), or 'NPC_TCR'(nasopharyngeal carcinoma (TCR))

# Runing examples

(1) For binding reactivity prediciton (BRP) (Classification)

    python ./maincode/DeepAIR_BRP.py  \
    --input_data_file ../DeepAIR/DataSplit/test/BRP/A0301_KLGGALQAK_IE-1_CMV_binder_test.csv  \
    --result_folder ../DeepAIR/result_BRP/A0301_KLGGALQAK_IE-1_CMV  \
    --epitope A0301_KLGGALQAK_IE-1_CMV  \
    --AF2_feature_folder ../DeepAIR/DataSplit/structure_feature  \
    --transformer_model_folder ../DeepAIR/ProtTrans/prot_bert_bfd  \

(2) For binding affinity prediciton (BAP) (Regression)

    python ./maincode/DeepAIR_BAP.py  \
    --input_data_file ../DeepAIR/DataSplit/test/BRP/A0301_KLGGALQAK_IE-1_CMV_binder_test.csv  \
    --result_folder ../DeepAIR/result_BAP/A0301_KLGGALQAK_IE-1_CMV  \
    --epitope A0301_KLGGALQAK_IE-1_CMV  \
    --AF2_feature_folder ../DeepAIR/DataSplit/structure_feature  \
    --transformer_model_folder ../DeepAIR/ProtTrans/prot_bert_bfd  \

(3) For immune repertoire classification (Multiple instance learning (MIL))

    python ./maincode/DeepAIR_MIL.py  \
    --input_data_file ../DeepAIR/DataSplit/test/MIL/IBD_BCR_test.csv  \
    --result_folder ../DeepAIR/result_MIL/IBD_BCR  \
    --AF2_feature_folder ../DeepAIR/DataSplit/structure_feature  \
    --transformer_model_folder ../DeepAIR/ProtTrans/prot_bert_bfd  \
    --task IBD_BCR \

# Time cost

Typical install time on a "normal" desktop computer is about 30 minutes.

# Dataset:

Example data are given in `./sampledata`. The full data including the obtained structures and training/validation/test splits are given on Google Drive at: https://drive.google.com/drive/folders/16i8mR56aL_hX5H-D8gY45Iyk6K_tuVkc. All code and data are also available at Zenodo (https://doi.org/10.5281/zenodo.7792621).

# Disclaimer
This tool is for research purpose and not approved for clinical use.

This is not an official Tencent product.

# Coypright

This tool is developed in Tencent AI Lab.

The copyright holder for this project is Tencent AI Lab.

All rights reserved.

# Citation

Zhao Y, He B, Li C, Xu Z, Su X, Rossjohn J, Song J, Yao J. DeepAIR: a deep-learning framework for effective integration of sequence and 3D structure to enable adaptive immune receptor analysis. bioRxiv.
