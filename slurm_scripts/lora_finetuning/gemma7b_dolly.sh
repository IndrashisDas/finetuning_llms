#!/bin/bash
#SBATCH --partition=accelerated-h100
#SBATCH -J dolly_gemma7b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -o ./logs/dolly15k/lora_gemma7b/%x.%N.%j.out
#SBATCH -e ./logs/dolly15k/lora_gemma7b/%x.%N.%j.err
#SBATCH --time=2-00:00:00

# Activate conda environment
source ~/.bashrc
conda activate golu

# Standard arguments
export SEED=3
export LOGIN_TOKEN="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"
export HF_MODEL_NAME="google/gemma-7b"
export HF_TOKENIZER_NAME="philschmid/gemma-tokenizer-chatml"
export HF_DATASET_NAME="philschmid/dolly-15k-oai-style"
export DATASET_DIR="/hkfs/work/workspace/scratch/fr_id74-finetuning_llms"
export RESULTS_PATH="/hkfs/work/workspace/scratch/fr_id74-finetuning_llms/results"

export MEAN_MIN=-0.5
export MEAN_MAX=0.5
export STD_MIN=-0.5
export STD_MAX=0.5

# Gaussian Noise
for mean in $(seq $MEAN_MIN 0.2 $MEAN_MAX); do
  for std in $(seq $STD_MIN 0.2 $STD_MAX); do
    echo "Running with mean=$mean and std=$std"
    python -m src.training_scripts.gemma_dolly_lora --login_token $LOGIN_TOKEN \
    --hf_model_name $HF_MODEL_NAME --hf_tokenizer_name $HF_TOKENIZER_NAME --hf_dataset_name $HF_DATASET_NAME \
    --dataset_dir $DATASET_DIR --results_path $RESULTS_PATH --seed_to_use $SEED \
    --model_activation_layer_name "act_fn" --activation_to_use "GELU_tanh" \
    --noise_to_use "gaussian" --gaussian_mean $mean --gaussian_std $std \
    --use_deterministic --test_dataset_size 100 --dataset_split_seed 42
  done
done

# Baseline
python -m src.training_scripts.gemma_dolly_lora --login_token $LOGIN_TOKEN --hf_model_name $HF_MODEL_NAME \
--hf_tokenizer_name $HF_TOKENIZER_NAME --hf_dataset_name $HF_DATASET_NAME --dataset_dir $DATASET_DIR \
--results_path $RESULTS_PATH --seed_to_use $SEED --model_activation_layer_name "act_fn" --activation_to_use "baseline" \
--use_deterministic --test_dataset_size 100 --dataset_split_seed 42

# Swish
python -m src.training_scripts.gemma_dolly_lora --login_token $LOGIN_TOKEN --hf_model_name $HF_MODEL_NAME \
--hf_tokenizer_name $HF_TOKENIZER_NAME --hf_dataset_name $HF_DATASET_NAME --dataset_dir $DATASET_DIR \
--results_path $RESULTS_PATH --seed_to_use $SEED --model_activation_layer_name "act_fn" --activation_to_use "Swish" \
--use_deterministic --test_dataset_size 100 --dataset_split_seed 42

# Mish
python -m src.training_scripts.gemma_dolly_lora --login_token $LOGIN_TOKEN --hf_model_name $HF_MODEL_NAME \
--hf_tokenizer_name $HF_TOKENIZER_NAME --hf_dataset_name $HF_DATASET_NAME --dataset_dir $DATASET_DIR \
--results_path $RESULTS_PATH --seed_to_use $SEED --model_activation_layer_name "act_fn" --activation_to_use "Mish" \
--use_deterministic --test_dataset_size 100 --dataset_split_seed 42

# GoLU
python -m src.training_scripts.gemma_dolly_lora --login_token $LOGIN_TOKEN --hf_model_name $HF_MODEL_NAME \
--hf_tokenizer_name $HF_TOKENIZER_NAME --hf_dataset_name $HF_DATASET_NAME --dataset_dir $DATASET_DIR \
--results_path $RESULTS_PATH --seed_to_use $SEED --model_activation_layer_name "act_fn" --activation_to_use "GoLU" \
--use_deterministic --test_dataset_size 100 --dataset_split_seed 42

# ReLU
python -m src.training_scripts.gemma_dolly_lora --login_token $LOGIN_TOKEN --hf_model_name $HF_MODEL_NAME \
--hf_tokenizer_name $HF_TOKENIZER_NAME --hf_dataset_name $HF_DATASET_NAME --dataset_dir $DATASET_DIR \
--results_path $RESULTS_PATH --seed_to_use $SEED --model_activation_layer_name "act_fn" --activation_to_use "ReLU" \
--use_deterministic --test_dataset_size 100 --dataset_split_seed 42

# LeakyReLU
python -m src.training_scripts.gemma_dolly_lora --login_token $LOGIN_TOKEN --hf_model_name $HF_MODEL_NAME \
--hf_tokenizer_name $HF_TOKENIZER_NAME --hf_dataset_name $HF_DATASET_NAME --dataset_dir $DATASET_DIR \
--results_path $RESULTS_PATH --seed_to_use $SEED --model_activation_layer_name "act_fn" --activation_to_use "LeakyReLU" \
--use_deterministic --test_dataset_size 100 --dataset_split_seed 42

# ELU
python -m src.training_scripts.gemma_dolly_lora --login_token $LOGIN_TOKEN --hf_model_name $HF_MODEL_NAME \
--hf_tokenizer_name $HF_TOKENIZER_NAME --hf_dataset_name $HF_DATASET_NAME --dataset_dir $DATASET_DIR \
--results_path $RESULTS_PATH --seed_to_use $SEED --model_activation_layer_name "act_fn" --activation_to_use "ELU" \
--use_deterministic --test_dataset_size 100 --dataset_split_seed 42
