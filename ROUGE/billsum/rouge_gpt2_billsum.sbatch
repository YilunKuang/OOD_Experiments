#!/bin/bash
###SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=128G
#SBATCH --job-name=rouge_calc
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/OOD_Text_Generation/BART-Billsum/slurm_train_%j.out

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '
echo "Running - evaluating fine-tuned BART (Gigaword) on Billsum (Rouge)"
source /ext3/env.sh
conda activate
python -u rouge_gpt2_gigaword_billsum.py'
