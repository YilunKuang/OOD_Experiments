#!/bin/bash
###SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=128G
#SBATCH --job-name=fine_tune_bart_on_wikihow
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=slurm_bart_on_wikihow_%j.out

## fine-tune vanilla bart on gigaword

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '

echo "Running - fine tune BART on wikihow"
source /ext3/env.sh
conda activate

python run_summarization.py \
    --model_name_or_path facebook/bart-large \
    --do_train \
    --do_eval \
    --dataset_name wikihow \
    --dataset_config all \
    --output_dir /scratch/yk2516/OOD_Text_Generation/BART-Wikihow \
    --predict_with_generate \
    --save_total_limit 2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --cache_dir /scratch/yk2516/cache'
