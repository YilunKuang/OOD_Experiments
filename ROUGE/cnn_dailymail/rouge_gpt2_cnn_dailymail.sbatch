#!/bin/bash
###SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=128G
#SBATCH --job-name=rouge_calc
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/OOD_Text_Generation/GPT2-CNN/slurm_train_%j.out


# Changes need to be make:
# 1. node, output dir
# 2. echo, python file name

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '
echo "Running - evaluating fine-tuned GPT2 (Gigaword) on CNN_Dailymail (Rouge)"
source /ext3/env.sh
conda activate
python -u rouge_gpt2_gigaword_cnn_dailymail.py'
