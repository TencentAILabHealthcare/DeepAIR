DATA_DIR=YourPath/alphafold2/  # Please replace YourPath with the exact path on your machine {Your downloaded data directory}. # Path to the directory containing the AlphaFold 2 downloaded data.
FASTA_DIR=YourPath/DeepAIR/preprocessing_structure_feature/sampledata/fasta/ # Please replace YourPath with the exact path on your machine. # Path to the directory containing the input FASTA files.
OUTPUT_DIR=YourPath/DeepAIR/preprocessing_structure_feature/sampledata/output_af2_structure/ # Please replace YourPath with the exact path on your machine # Path to the directory where the results will be saved.

for i in `ls $FASTA_DIR`; do
  echo $FASTA_DIR$i
  python3 alphafold/docker/run_docker.py   --fasta_paths=$FASTA_DIR$i \
  --max_template_date=2022-05-14 \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --model_preset=monomer_ptm \
  --docker_image_name=alphafold_v3 \
  --models_to_relax=best \
  --db_preset=reduced_dbs \
  --num_multimer_predictions_per_model=1
done