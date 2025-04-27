# ✨ Fine-Tuning LLMs with Noisy Activations and LoRA/QLoRA

This project introduces a novel approach to fine-tuning Large Language Models (LLMs) by integrating noisy activation
functions with efficient parameter fine-tuning techniques such as LoRA and QLoRA. We inject controlled noise into
activation functions during training to potentially enhance model robustness, generalization, and optimization, while
maintaining memory-efficient fine-tuning.

---

## 🛠️ Features

 - 🔥 Fine-tune any HuggingFace-supported LLM with custom noisy activations.

 - 📈 Supports Gaussian and Multiplicative noise injections.

 - 🛡️ Switch seamlessly between baseline and noisy activations.
 
 - 🚀 Integrates LoRA and QLoRA for memory-efficient fine-tuning.
 
 - 🎯 Built-in metrics tracking (loss, perplexity, ROUGE scores, accuracy).
 
 - 🖼️ Auto-generates training plots and saves optimization traces.
 
 - 🧹 Clean dataset preparation pipeline (supports Dolly 15k and beyond).

---

## 💡 Motivation

We are aware that training deep neural networks with controlled noise makes models robust. Be it Normalization
or Dropout, noise has always been helpful. Hence, we find a unique way to train neural networks not just by injecting
QLoRA, but also using noisy activation outputs. Further,
 
 - It encourages exploration of activation space.
 - Acts as an implicit regularizer.
 - May improve convergence and generalization in low-data regimes.

---

## 🧩 Project Structure

| File | Purpose |
|:---|:---|
| `activations.py` | Defines activation functions and wraps them with noise injectors. |
| `train.py` | Main script for fine-tuning models with noisy activations + LoRA/QLoRA. |
| `prepare_dolly.py` | Preprocessing helper for tokenizing and saving the Dolly 15k dataset. |

---

## 🎯 How It Works

1. **Replace Model Activations**  
   Swap selected activation layers (e.g., ReLU, GELU) with `NoisyActivation`.

2. **Inject Noise During Training**  
   During model training, apply Gaussian or Multiplicative noise.

3. **Fine-Tune Efficiently**  
   Leverage LoRA/QLoRA strategies to fine-tune just a fraction of model parameters.

4. **Evaluate and Plot**  
   Track train/test loss, perplexity, and performance metrics. Visualize training dynamics automatically.

---

## 🚀 Quick Start

We make finetuning easily configurable from the command line. This allows us to fine-tune models easily by changing the
parameters as provided below.

### Clone the repository

```bash
git clone https://github.com/IndrashisDas/finetuning_llms.git
```

### Install Dependencies

```bash
pip install torch transformers datasets peft trl huggingface_hub matplotlib nltk evaluate
```

### Fine-Tuning Commands

For fine-tune a model, for example "Gemma7B", some of the environment variables are as follows,

```bash
export SEED=<seed>
export LOGIN_TOKEN=<hf_login_token>
export HF_MODEL_NAME="google/gemma-7b"  # One can train several other models from HF
export HF_TOKENIZER_NAME="philschmid/gemma-tokenizer-chatml"
export HF_DATASET_NAME="philschmid/dolly-15k-oai-style"
export DATASET_DIR="./"
export RESULTS_PATH="./results"

# Below variables can be set only in the case if we use noisy activation fine-tuning 
export MEAN=0.0
export STD=0.2
```

#### Baseline

```bash
python -m src.train --login_token $LOGIN_TOKEN --hf_model_name $HF_MODEL_NAME --hf_tokenizer_name $HF_TOKENIZER_NAME --hf_dataset_name $HF_DATASET_NAME --dataset_dir $DATASET_DIR --results_path $RESULTS_PATH --seed_to_use $SEED  --model_activation_layer_name "act_fn" --activation-to-use "baseline" --test_dataset_size 100 --dataset_split_seed 42 --use_deterministic
```

#### Standard replacement of baseline activation

```bash
python -m src.train --login_token $LOGIN_TOKEN --hf_model_name $HF_MODEL_NAME --hf_tokenizer_name $HF_TOKENIZER_NAME --hf_dataset_name $HF_DATASET_NAME --dataset_dir $DATASET_DIR --results_path $RESULTS_PATH --seed_to_use $SEED  --model_activation_layer_name "act_fn" --activation_to_use "ReLU" --test_dataset_size 100 --dataset_split_seed 42 --use_deterministic
```

#### Replacing standard baseline with some other activation + noise output

```bash
python -m src.train --login_token $LOGIN_TOKEN --hf_model_name $HF_MODEL_NAME --hf_tokenizer_name $HF_TOKENIZER_NAME --hf_dataset_name $HF_DATASET_NAME --dataset_dir $DATASET_DIR --results_path $RESULTS_PATH --seed_to_use $SEED  --model_activation_layer_name "act_fn" --activation_to_use "Swish" --noise_to_use "gaussian" --gaussian_mean $MEAN --gaussian_std $STD --test_dataset_size 100 --dataset_split_seed 42 --use_deterministic
```

Note - the baseline activation in Gemma series of models is the ```tanh``` approximator of ```GELU```.

#### Further, you can leverage this repository to fine-tune your own models


## 🛠️ TODOs

 - Add more advanced noise types (e.g., adversarial noise, dropout variants).

 - Benchmark against standard fine-tuning without noise.

 - Integrate other LoRA variants (e.g., AdaLoRA).

Feel free to contibute to this repository!

## Citation

Please cite GoLU in case you use it in your work 🙌

```bibtex
@misc{noisyactivations-qlora-das-2025,
  author = {Indrashis Das},
  title = {Fine-Tuning LLMs with Noisy Activations and LoRA/QLoRA},
  year = {2025},
  url = {https://github.com/IndrashisDas/finetuning_llms},
  note = {GitHub repository},
  howpublished = {\url{https://github.com/IndrashisDas/finetuning_llms}}
}
```
