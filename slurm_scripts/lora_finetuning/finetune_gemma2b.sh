#!/bin/bash
#SBATCH --partition=accelerated-h100
#SBATCH -J lora_gemma2b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -o ./logs/dolly15k/lora_gemma2b/%x.%N.%j.out
#SBATCH -e ./logs/dolly15k/lora_gemma2b/%x.%N.%j.err
#SBATCH --time=2-00:00:00

# Activate conda environment
source ~/.bashrc
conda activate golu

# Standard arguments
export LOGIN_TOKEN="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"
export HF_MODEL_NAME="google/gemma-2b"
export HF_TOKENIZER_NAME="philschmid/gemma-tokenizer-chatml"
export HF_DATASET_NAME="philschmid/dolly-15k-oai-style"
export DATASET_DIR="/hkfs/work/workspace/scratch/fr_id74-finetuning_llms"
export RESULTS_PATH="/hkfs/work/workspace/scratch/fr_id74-finetuning_llms/results"

python -m src.training_scripts.lora_finetuning --login_token $LOGIN_TOKEN --hf_model_name $HF_MODEL_NAME \
--hf_tokenizer_name $HF_TOKENIZER_NAME --hf_dataset_name $HF_DATASET_NAME --dataset_dir $DATASET_DIR \
--results_path $RESULTS_PATH --seed_to_use 1 --model_activation_layer_name "act_fn" --activation_to_use "baseline" \
--noise_to_use "none" --gaussian_mean 0.0 --gaussian_std 1.0 --poisson_lambda 1.0 --use_deterministic \
--test_dataset_size 100 --dataset_split_seed 42
