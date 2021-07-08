#!/bin/bash
###SBATCH --cpus-per-task=2
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=128G
#SBATCH --job-name=fine_tune_gpt2
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/OOD_Text_Generation/GPT2-XSum/slurm_train_%j.out

## Testing finetuned gpt2 on wikihow test set.

# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '

# echo "Running - evaluating checkpoint on wikihow"
# source /ext3/env.sh
# conda activate bart2

# python -u run_clm_batch.py --model_name_or_path /scratch/nm3571/ckpts/checkpoint-36000 --dataset_name wikihow --dataset_config_name all --data_dir /scratch/nm3571 --eval_accumulation_steps 1 --output_dir /scratch/nm3571/ --per_device_eval_batch_size=1 --cache_dir /scratch/nm3571/cache'

# # Testing finetuned gpt2 on gigaword test set

# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '

# echo "Running - evaluating checkpoint on gigaword"
# source /ext3/env.sh
# conda activate bart2

# python -u run_clm_batch.py --model_name_or_path /scratch/nm3571/ckpts/checkpoint-36000 --dataset_name gigaword --eval_accumulation_steps 10 --output_dir /scratch/nm3571/ --per_device_eval_batch_size=1 --cache_dir /scratch/nm3571/cache'

## Testing fine-tuned gpt2 on billsum test set

# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '

# echo "Running - evaluating finetuned gpt2 on billsum"
# source /ext3/env.sh
# conda activate bart2

# python -u run_clm_batch.py --model_name_or_path /scratch/nm3571/ckpts/checkpoint-36000 --dataset_name billsum --eval_accumulation_steps 20 --output_dir /scratch/nm3571/ --per_device_eval_batch_size=1 --cache_dir /scratch/nm3571/cache'

# # Testing fine-tuned gpt2 on big patent test set

# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '

# echo "Running - evaluating finetuned gpt2 on big patent"
# source /ext3/env.sh
# conda activate bart2

# python -u run_clm_batch.py --model_name_or_path /scratch/nm3571/ckpts/checkpoint-36000 --dataset_name big_patent --dataset_config_name g --eval_accumulation_steps 1 --output_dir /scratch/nm3571/ --per_device_eval_batch_size=1 --cache_dir /scratch/nm3571/cache'

# Testing gpt2 on wikihow test set.

# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '

# echo "Running - evaluating gpt2 on wikihow"
# source /ext3/env.sh
# conda activate bart2

# python -u run_clm_batch.py --model_name_or_path gpt2 --dataset_name wikihow --dataset_config_name all --data_dir /scratch/nm3571 --eval_accumulation_steps 1 --output_dir /scratch/nm3571/ --per_device_eval_batch_size=1 --cache_dir /scratch/nm3571/cache'

# # Testing gpt2 on gigaword test set

# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '

# echo "Running - evaluating gpt2 on gigaword"
# source /ext3/env.sh
# conda activate bart2

# python -u run_clm_batch.py --model_name_or_path gpt2 --dataset_name gigaword --eval_accumulation_steps 10 --output_dir /scratch/nm3571/ --per_device_eval_batch_size=1 --cache_dir /scratch/nm3571/cache'

# Testing gpt2 on billsum test set

# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '
# echo "Running - evaluating gpt2 gpt2 on billsum"
# source /ext3/env.sh
# conda activate bart2
# python -u run_clm_batch.py --model_name_or_path gpt2 --dataset_name billsum --eval_accumulation_steps 20 --output_dir /scratch/nm3571/ --per_device_eval_batch_size=1 --cache_dir /scratch/nm3571/cache'

# -------------------------------------------------------------------------#
## Testing fine-tuned gpt2 on XSum test set
# -------------------------------------------------------------------------#

# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '

# echo "Running - evaluating finetuned gpt2 on XSum"
# source /ext3/env.sh
# conda activate

# python -u run_clm_batch.py --model_name_or_path /scratch/yk2516/OOD_Text_Generation/checkpoint-36000 --dataset_name xsum --do_predict --eval_accumulation_steps 20 --output_dir /scratch/yk2516/OOD_Text_Generation/GPT2-XSum --per_device_eval_batch_size=1 --cache_dir /scratch/yk2516/cache'

# -------------------------------------------------------------------------#
# Testing vanilla gpt2 on XSum test set - change to 85 examples
# -------------------------------------------------------------------------#

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '

echo "Running - evaluating vanilla gpt2 on XSum"
source /ext3/env.sh
conda activate

python -u run_clm_batch.py --model_name_or_path gpt2 --dataset_name xsum --do_predict --eval_accumulation_steps 20 --output_dir /scratch/yk2516/OOD_Text_Generation/GPT2-XSum --per_device_eval_batch_size=1 --cache_dir /scratch/yk2516/cache'

# -------------------------------------------------------------------------#
# Testing fine-tuned gpt2 on CNN Dailymail test set - change to 85 examples
# -------------------------------------------------------------------------#

# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '

# echo "Running - evaluating finetuned gpt2 on CNN Dailymail"
# source /ext3/env.sh
# conda activate

# python -u run_clm.py --model_name_or_path /scratch/yk2516/OOD_Text_Generation/checkpoint-36000 --dataset_name cnn_dailymail --dataset_config_name 3.0.0 --do_predict --eval_accumulation_steps 20 --output_dir /scratch/yk2516/OOD_Text_Generation/GPT2-CNN --per_device_eval_batch_size=1 --cache_dir /scratch/yk2516/cache'

# -------------------------------------------------------------------------#
# Testing vanilla gpt2 on CNN Dailymail test set - change to 85 examples
# Warning: This part of the code is not correctly implemented yet.
# -------------------------------------------------------------------------#
# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '

# echo "Running - evaluating vanilla gpt2 on CNN Dailymail"
# source /ext3/env.sh
# conda activate

# python -u run_clm.py --model_name_or_path gpt2 --dataset_name cnn_dailymail --dataset_config_name 3.0.0 --do_predict --eval_accumulation_steps 20 --output_dir /scratch/yk2516/OOD_Text_Generation/GPT2-CNN --per_device_eval_batch_size=1 --cache_dir /scratch/yk2516/cache'
