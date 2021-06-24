#!/bin/bash
###SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=128G
#SBATCH --job-name=generate_summaries
#SBATCH --mail-user=nm3571@nyu.edu
#SBATCH --output=summary_generated_%j.out


singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '

echo "Running"
source /ext3/env.sh
conda activate bart2

python -u generate_summaries.py'
