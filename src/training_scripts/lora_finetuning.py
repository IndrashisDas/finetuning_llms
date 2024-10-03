"""
The following code fine-tunes LLMs with noisy activations and lora
"""

import os
import json
import datetime
import logging
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from peft import LoraConfig, AutoPeftModelForCausalLM
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline
from trl import SFTTrainer
import evaluate
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt_tab")

from src.prepare_datasets.prepare_dolly import prepare_dolly
from src.utils.activations import NoisyActivation


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def num_parameters(model):
    total_parameters = 0
    trainable_parameters = 0
    for param in model.parameters():
        total_parameters += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    return total_parameters, trainable_parameters

def replace_activation(
    module, args
):
    for name, child in module.named_children():
        if name == args.model_activation_layer_name:
            setattr(
                module, name, NoisyActivation(
                    activation=args.activation_to_use,
                    noise=args.noise_to_use,
                    mean=args.gaussian_mean,
                    std=args.gaussian_std,
                    lambda_value=args.poisson_lambda,
                )
            )
        else:
            replace_activation(child, args)
    return module

def save_opt_trace(opt_trace, path):
    with open(path, 'w') as f:
        json.dump(opt_trace, f, indent=4)

def plot_results(opt_trace, plot_path):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot train and test loss
    ax[0].plot(opt_trace['steps_or_epochs'], opt_trace['train_loss'], label='Train Loss')
    ax[0].plot(opt_trace['steps_or_epochs'], opt_trace['test_loss'], label='Test Loss')
    ax[0].set_xlabel('Steps/Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_title('Train vs Test Loss')

    # Plot train and test perplexity
    ax[1].plot(opt_trace['steps_or_epochs'], opt_trace['train_perplexity'], label='Train Perplexity')
    ax[1].plot(opt_trace['steps_or_epochs'], opt_trace['test_perplexity'], label='Test Perplexity')
    ax[1].set_xlabel('Steps/Epochs')
    ax[1].set_ylabel('Perplexity')
    ax[1].legend()
    ax[1].set_title('Train vs Test Perplexity')

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def main(args):
    
    # Prepare logging --------------------------------------------------------------------------------------------------
    if args.noise_to_use in ['none']:
        extra_string = f'Noise - {args.noise_to_use}'
    elif args.noise_to_use in ['gaussian', 'multiplicative']:
        extra_string = f'Noise - {args.noise_to_use} | Mean - {args.gaussian_mean} | Std Dev - {args.gaussian_std}'
    elif args.noise_to_use in ['poisson']:
        extra_string = f'Noise - {args.noise_to_use} | Lambda - {args.poisson_lambda}'
    logging.info(
        f'{datetime.datetime.now()} | Dataset - {args.hf_dataset_name} | Model - {args.hf_model_name} | Activation - {args.activation_to_use} | {extra_string}')
    logging.info(f'Arguments are as follows \n{args}')
    
    # Login to HuggingFace and set seed --------------------------------------------------------------------------------
    login(token=args.login_token)
    set_seed(args.seed_to_use)
    if args.use_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Get all the paths sorted out -------------------------------------------------------------------------------------
    model_name_for_caching = ((args.hf_model_name).replace('/', '_')).replace('-', "_")
    tokenizer_name_for_caching = ((args.hf_tokenizer_name).replace('/', '_')).replace('-', "_")
    dataset_name_for_caching = ((args.hf_dataset_name).replace('/', '_')).replace('-', "_")
    
    dataset_dir = f'{args.dataset_dir}/{dataset_name_for_caching}'
    model_cache_dir = f'{args.dataset_dir}/{model_name_for_caching}'
    tokenizer_cache_dir = f'{args.dataset_dir}/{tokenizer_name_for_caching}'
    out_dir = f'{args.results_path}/{dataset_name_for_caching}/{model_name_for_caching}/{args.activation_to_use}'
    
    if args.noise_to_use in ['gaussian', 'multiplicative']:
        out_dir = f'{out_dir}_{args.noise_to_use}_{args.gaussian_mean}_{args.gaussian_std}'
    elif args.noise_to_use in ['poisson']:
        out_dir = f'{out_dir}_{args.noise_to_use}_{args.poisson_lambda}'
    elif args.noise_to_use in ['none']:
        out_dir = out_dir
    out_dir = f'{out_dir}/{args.seed_to_use}'
    opt_trace_path = f'{out_dir}/opt_trace.json'
    plots_path = f'{out_dir}/plot.png'
    
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)
    os.makedirs(tokenizer_cache_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(opt_trace_path):  # Check if the training is already done
    
        # Get dictionaries to store results --------------------------------------------------------------------------------
        opt_trace = {
            'logging_strategy':'',
            'steps_or_epochs':[],
            'train_loss':[],
            'test_loss':[],
            'train_perplexity':[],
            'test_perplexity':[],
            'post_train_accuracy':0,
            'post_train_rogue_1':0,
            'post_train_rogue_2':0,
            'post_train_rogue_L':0,
            'post_train_rogue_L_sum':0,
            'post_train_gen_len':0,
            'train_time':0,
            'test_time':0,
        }

        # Create the BnB config --------------------------------------------------------------------------------------------
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
        )
        
        # Create the LoRA config -------------------------------------------------------------------------------------------
        peft_config = LoraConfig(
            lora_alpha=8,
            lora_dropout=0.05,
            r=6,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        
        # Load model and tokenizer -----------------------------------------------------------------------------------------
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model_name,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            cache_dir=model_cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer_name, cache_dir=tokenizer_cache_dir)
        tokenizer.padding_side = 'right'
        if args.activation_to_use != 'baseline':
            model = replace_activation(model, args)
        
        logging.info(f'Loaded model successfully, the model looks like - \n{model}')
        total_parameters, trainable_parameters = num_parameters(model)
        logging.info(f'# of Total Parameters - {total_parameters / 1000000:.4f} Million')
        logging.info(f'# of Trainable Parameters - {trainable_parameters / 1000000:.4f} Million')
        
        # Load the dataset -------------------------------------------------------------------------------------------------
        if 'dolly' in args.hf_dataset_name:
            train_dataset, test_dataset = prepare_dolly(dataset_dir, args)
        else:
            raise ValueError('No dataset found!')
        
        # Define the training arguments ------------------------------------------------------------------------------------
        training_args = TrainingArguments(
            output_dir=out_dir,
            bf16=True,
            tf32=True,
            prediction_loss_only=True,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            num_train_epochs=3,
            optim="adamw_torch_fused",
            learning_rate=2e-4,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            logging_strategy="steps",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=10,
            save_total_limit=1,
            push_to_hub=False,
            report_to="tensorboard",
            neftune_noise_alpha=args.neftune_noise_alpha
        )
        
        max_seq_length = 1512 # max sequence length for model and packing of the dataset

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=peft_config,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            packing=True,
            dataset_kwargs={
                "add_special_tokens": False, # We template with special tokens
                "append_concat_token": False, # No need to add additional separator token
            }
        )

        trainer.train()
        trainer.save_model()
        
        counter = 1
        opt_trace['logging_strategy'] = training_args.logging_strategy
        for item in trainer.state.log_history:
            if 'loss' in item:
                steps_or_epochs = counter if training_args.logging_strategy == 'epochs' else counter * training_args.logging_steps
                opt_trace['steps_or_epochs'].append(steps_or_epochs)
                opt_trace['train_loss'].append(item['loss'])
                opt_trace['train_perplexity'].append(math.exp(item['loss']))
                counter += 1
            elif 'eval_loss' in item:
                opt_trace['test_loss'].append(item['eval_loss'])
                opt_trace['test_perplexity'].append(math.exp(item['eval_loss']))
                opt_trace['test_time'] += item['eval_runtime'] / 60
            elif 'train_runtime' in item:
                opt_trace['train_time'] += item['train_runtime'] / 60
        
        del model
        del trainer
        torch.cuda.empty_cache()
            
        # Load PEFT model on CPU
        model = AutoPeftModelForCausalLM.from_pretrained(
            out_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        # Merge LoRA and base model and save
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")

        del model
        del merged_model
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(out_dir)
        model = AutoPeftModelForCausalLM.from_pretrained(out_dir, device_map="auto", torch_dtype=torch.float16)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        # get token id for end of conversation
        eos_token = tokenizer("<|im_end|>",add_special_tokens=False)["input_ids"][0]

        metric = evaluate.load("rouge")
        
        # helper function to postprocess text
        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]
            # rougeLSum expects newline after each sentence
            preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(sent_tokenize(label)) for label in labels]

            return preds, labels
        
        # helper function to compute accuracy
        def compute_accuracy(decoded_preds, decoded_labels):
            correct = 0
            total = 0
            for pred, label in zip(decoded_preds, decoded_labels):
                pred_tokens = pred.split()
                label_tokens = label.split()
                total += len(label_tokens)
                correct += sum(1 for pred_token, label_token in zip(pred_tokens, label_tokens) if pred_token == label_token)
            return correct / total if total > 0 else 0

        # helper function to compute rogue and consolidate results
        def compute_metrics(preds, labels):
            decoded_preds, decoded_labels = postprocess_text(preds, labels)
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}
            # Compute accuracy
            accuracy = compute_accuracy(decoded_preds, decoded_labels)
            result["accuracy"] = round(accuracy * 100, 2)
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            return result

        references = []
        predictions = []

        with torch.no_grad():
            counter = 0
            for item in test_dataset:
                prompt = pipe.tokenizer.apply_chat_template(
                    [{"role": "user", "content": item['messages'][0]['content']}], tokenize=False, add_generation_prompt=True)
                outputs = pipe(
                    prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=eos_token)
                predictions.append(outputs[0]['generated_text'][len(prompt):].strip())
                references.append(item['messages'][1]['content'])
                counter += 1
                if counter % 10 == 0:
                    logging.info(f'Completed {counter} datapoints - {datetime.datetime.now()}')

        results = compute_metrics(predictions, references)
        
        opt_trace['post_train_accuracy'] = results['accuracy']
        opt_trace['post_train_rogue_1'] = results['rouge1']
        opt_trace['post_train_rogue_2'] = results['rouge2']
        opt_trace['post_train_rogue_L'] = results['rougeL']
        opt_trace['post_train_rogue_L_sum'] = results['rougeLsum']
        opt_trace['post_train_gen_len'] = results['gen_len']
        
        save_opt_trace(opt_trace, opt_trace_path)
        plot_results(opt_trace, plots_path)
    
    else:
        logging.info('Optimization Trace already exists')
    
    logging.info('Training completed!')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Fine-Tuning LLMs with Noisy Activations')
    
    # Standard arguments -----------------------------------------------------------------------------------------------
    parser.add_argument('--login_token', default='', help='Login Token from huggingface', type=str)
    parser.add_argument('--hf_model_name', default='', help='Name of model to train', type=str)
    parser.add_argument('--hf_tokenizer_name', default='', help='Name of tokenizer to use', type=str)
    parser.add_argument('--hf_dataset_name', default='', help='Which dataset to use for fine tuning?', type=str)
    parser.add_argument('--dataset_dir', default='', help='Path to download the data', type=str)
    parser.add_argument('--results_path', default='', help='Path to store the training results', type=str)
    
    # Activation & Seed arguments --------------------------------------------------------------------------------------
    parser.add_argument('--seed_to_use', default=1, help='Seed to fix randomization', type=int)
    parser.add_argument('--model_activation_layer_name', default='', help='Name of the layer that holds the \
        activation', type=str)
    parser.add_argument('--activation_to_use', default='baseline', help='Set it to the activation originally used \
        i.e. the baseline activation', type=str)
    parser.add_argument('--noise_to_use', default='none', choices=['gaussian', 'poisson', 'multiplicative', 'none'],
                        help='Set it to none for baseline', type=str)
    parser.add_argument('--gaussian_mean', default=0.0, help='Mean of Gaussian Dist for Gaussian and Multiplicative \
        noise', type=float)
    parser.add_argument('--gaussian_std', default=1.0, help='Std Dev of Gaussian Dist for Gaussian and Multiplicative \
        noise', type=float)
    parser.add_argument('--poisson_lambda', default=1.0, help='Lambda of Poisson Dist for Poisson noise', type=float)
    
    # Training Arguments -----------------------------------------------------------------------------------------------
    parser.add_argument('--use_deterministic', action='store_true', default=False, help='Use deterministic algorithms')
    parser.add_argument('--test_dataset_size', default=100, help='Part of the dataset to reserve for testing eg - \
        500 for 500 datapoints, else 0.2 for 20 percent of the dataset', type=str)
    parser.add_argument('--dataset_split_seed', default=42, help='Seed to use for splitting dataset', type=int)
    parser.add_argument('--neftune_noise_alpha', type=float, default=None, help='Noise scale factor for NEFTune.')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    main(args)
