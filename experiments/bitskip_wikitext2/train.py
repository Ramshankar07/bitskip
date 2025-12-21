
import os
import argparse
import logging
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from safetensors.torch import save_file

import sys
# Add project root to sys.path to resolve bitnet package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import ExperimentConfig
from model_factory import create_model

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="BitSkip WikiText-2 Training")
    
    # Core Experiment Identifiers
    parser.add_argument("--model_id", type=str, required=True, help="Unique ID for the experiment run")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    
    # Model Configuration Overrides
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "int8", "int4"], help="Weight precision")
    parser.add_argument("--use_hadamard", action="store_true", help="Use Hadamard transform")
    
    # Early Exit Configuration Overrides
    parser.add_argument("--no_early_exit", action="store_true", help="Disable early exit completely")
    parser.add_argument("--early_exit_lambda", type=float, default=0.0, help="Early exit loss weight")
    parser.add_argument("--p_max", type=float, default=0.0, help="Maximum layer dropout probability")
    parser.add_argument("--dropout_schedule", type=str, default="quadratic", help="Dropout schedule")
    
    # Training Overrides
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile to speed up training")
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_wikitext2_loader(tokenizer, batch_size, seq_length):
    logger.info("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )

    # Concatenate all texts
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= seq_length:
            total_length = (total_length // seq_length) * seq_length
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
    )
    
    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]
    test_dataset = lm_datasets["test"]
    
    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        # Attention mask is all 1s since we grouped texts perfectly
        attention_mask = [[1] * len(ids) for ids in input_ids]
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Initialize Config
    config = ExperimentConfig()
    
    # Apply Overrides
    if args.precision == "fp16":
        config.weight_bits = 16
    elif args.precision == "int8":
        config.weight_bits = 8
    elif args.precision == "int4":
        config.weight_bits = 4
        
    config.use_hadamard = args.use_hadamard
    
    if args.no_early_exit:
        config.use_early_exit = False
        config.early_exit_loss_weight = 0.0
        config.dropout_probability_max = 0.0
    else:
        config.use_early_exit = True
        config.early_exit_loss_weight = args.early_exit_lambda
        config.dropout_probability_max = args.p_max
        config.dropout_schedule = args.dropout_schedule
        
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.learning_rate = args.learning_rate
    config.seed = args.seed
    
    # Create Output Directory
    run_dir = os.path.join(args.output_dir, args.model_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize Tokenizer (GPT-2)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize Data
    train_loader, val_loader, test_loader = get_wikitext2_loader(tokenizer, config.batch_size, config.max_position_embeddings)
    
    # Initialize Model
    model = create_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if args.compile:
        if hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model)
        else:
            logger.warning("torch.compile is not available in this version of PyTorch. Skipping compilation.")
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1)
    
    # Initialize scaler for AMP
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    num_training_steps = config.num_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=num_training_steps
    )
    
    # Training Loop
    logger.info(f"Starting training for {args.model_id}...")
    model.train()
    global_step = 0
    total_loss = 0.0
    
    while global_step < num_training_steps:
        for batch in train_loader:
            if global_step >= num_training_steps:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            # Note: BitNetModel2 forward supports exit_layer and training_step for curriculum
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    training_step=global_step, # For curriculum if needed
                )
            
            loss = outputs.loss
            
            # Backward pass
            loss = loss / config.gradient_accumulation_steps
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                
                if (global_step + 1) % 100 == 0:
                    logger.info(f"Step {global_step+1}/{num_training_steps} | Loss: {loss.item() * config.gradient_accumulation_steps:.4f}")
            
            global_step += 1
            
    # Save Model
    logger.info(f"Saving model to {run_dir}")
    save_file(model.state_dict(), os.path.join(run_dir, "model.safetensors"))
    
    # Evaluation
    logger.info("Evaluating on Validation Set...")
    model.eval()
    total_eval_loss = 0.0
    eval_steps = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_eval_loss += outputs.loss.item()
            eval_steps += 1
            
    avg_eval_loss = total_eval_loss / eval_steps
    perplexity = math.exp(avg_eval_loss)
    logger.info(f"Validation Perplexity: {perplexity:.2f}")
    
    # Write results to file
    with open(os.path.join(run_dir, "results.txt"), "w") as f:
        f.write(f"Validation Perplexity: {perplexity:.2f}\n")

if __name__ == "__main__":
    main()
