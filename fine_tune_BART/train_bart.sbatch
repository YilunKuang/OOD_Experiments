#!/bin/bash
###SBATCH --cpus-per-task=2
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=128G
#SBATCH --job-name=fine_tune_bart_on_gigaword
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=slurm_bart_on_gigaword_%j.out

## fine-tune vanilla bart on gigaword

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '

echo "Running - fine tune BART on gigaword"
source /ext3/env.sh
conda activate

python run_summarization.py \
    --model_name_or_path bart \
    --do_train \
    --do_eval \
    --dataset_name gigaword \
    --source_prefix "summarize: " \
    --output_dir /scratch/yk2516/OOD_Text_Generation/BART-Gigaword \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --cache_dir /scratch/yk2516/cache'
