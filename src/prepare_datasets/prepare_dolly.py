"""
This file helps to tokenize the dolly-15k dataset and further saves the tokenized files.
"""

import os
import torch
import logging
from datasets import load_dataset


def prepare_dolly(dataset_dir, args):
    
    split_train_data_path = os.path.join(dataset_dir, 'train.pt')
    split_test_data_path = os.path.join(dataset_dir, 'test.pt')
    
    # Check if the tokenized data already exists
    if os.path.exists(split_train_data_path) and os.path.exists(split_test_data_path):
        logging.info(f"Loading train data from path - {split_train_data_path}")
        train_dataset = torch.load(split_train_data_path)
        
        logging.info(f"Loading test data from path - {split_test_data_path}")
        test_dataset = torch.load(split_test_data_path)
    else:
        # Load the dataset
        dataset = load_dataset(args.hf_dataset_name, split='train', cache_dir=dataset_dir)
        
        # Split dataset into train and test
        if "." in args.test_dataset_size:
            test_dataset_size = float(args.test_dataset_size)
        else:
            test_dataset_size = int(args.test_dataset_size)
        train_test_split = dataset.train_test_split(test_size=test_dataset_size, seed=args.dataset_split_seed)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        
        # Save the data for future use
        logging.info(f"Saving train data at path - {split_train_data_path}")
        torch.save(train_dataset, split_train_data_path)
        
        logging.info(f"Saving test data at path - {split_test_data_path}")
        torch.save(test_dataset, split_test_data_path)
    
    num_train_samples = len(train_dataset)
    num_test_samples = len(test_dataset)
    
    logging.info(f"Number of train samples: {num_train_samples}")
    logging.info(f"Number of test samples: {num_test_samples}")
    
    return train_dataset, test_dataset
