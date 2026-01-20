
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
from tqdm import tqdm

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
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--eval_every_steps", type=int, default=None, help="Evaluation frequency")
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
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if args.eval_every_steps is not None:
        config.eval_every_steps = args.eval_every_steps
    
    # Create Output Directory
    abs_output_dir = os.path.abspath(args.output_dir)
    run_dir = os.path.join(abs_output_dir, args.model_id)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Using output directory: {run_dir}")
    
    # Initialize Tokenizer (GPT-2)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize Data
    train_loader, val_loader, test_loader = get_wikitext2_loader(tokenizer, config.batch_size, config.max_position_embeddings)
    
    # Initialize Model
    model = create_model(config)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
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
    
    # Early stopping state
    best_val_ppl = float('inf')
    patience_counter = 0
    best_model_state = None
    early_stopped = False
    
    # Training Loop
    logger.info(f"Starting training for {args.model_id}...")
    logger.info(f"Early stopping: eval every {config.eval_every_steps} steps, patience={config.patience}, min_delta={config.min_delta}")
    model.train()
    global_step = 0
    total_loss = 0.0
    
    # Initialize progress bar
    pbar = tqdm(total=num_training_steps, desc=f"Training {args.model_id}", unit="step")
    pbar.set_postfix({"loss": "N/A", "val_ppl": "N/A"})
    
    # Progress bar initialization already done above
    
    try:
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
                
                # Update progress bar
                current_loss = loss.item() * config.gradient_accumulation_steps
                pbar.set_postfix({"loss": f"{current_loss:.4f}", "val_ppl": f"{best_val_ppl:.2f}" if best_val_ppl != float('inf') else "N/A"})
            
            global_step += 1
            pbar.update(1)
            
            # Periodic validation and early stopping check
            if global_step % config.eval_every_steps == 0:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input_ids = val_batch["input_ids"].to(device)
                        val_attention_mask = val_batch["attention_mask"].to(device)
                        val_labels = val_batch["labels"].to(device)
                        
                        val_outputs = model(
                            input_ids=val_input_ids,
                            attention_mask=val_attention_mask,
                            labels=val_labels
                        )
                        val_loss += val_outputs.loss.item()
                        val_steps += 1
                
                avg_val_loss = val_loss / val_steps
                val_ppl = math.exp(avg_val_loss)
                
                logger.info(f"Validation at step {global_step}: PPL={val_ppl:.2f}, Best={best_val_ppl:.2f}")
                
                # Periodically save results even if not the best
                results_file = os.path.join(run_dir, "results.txt")
                with open(results_file, "w") as f:
                    f.write(f"Validation Perplexity: {val_ppl:.2f}\n")
                    f.write(f"Last Step: {global_step}\n")
                    if best_val_ppl != float('inf'):
                        f.write(f"Best Validation PPL: {best_val_ppl:.2f}\n")

                # Check for improvement
                if val_ppl < best_val_ppl - config.min_delta:
                    best_val_ppl = val_ppl
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    logger.info(f"New best validation PPL: {best_val_ppl:.2f}")
                    # Save best model
                    save_file(best_model_state, os.path.join(run_dir, "model.safetensors"))
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Patience: {patience_counter}/{config.patience}")
                    
                    if patience_counter >= config.patience:
                        logger.info(f"Early stopping triggered at step {global_step}")
                        early_stopped = True
                        if best_model_state is not None:
                            model.load_state_dict(best_model_state)
                        break
                
                model.train()
            
            if early_stopped:
                break
        
            if early_stopped:
                break
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current progress...")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise e
    finally:
        pbar.close()
        # Initial results save (in case of crash/interrupt)
        final_ppl = best_val_ppl if best_val_ppl != float('inf') else 0.0
        temp_results_file = os.path.join(run_dir, "results.txt")
        try:
            with open(temp_results_file, "w") as f:
                f.write(f"Validation Perplexity: {final_ppl:.2f}\n")
                f.write(f"Final Step: {global_step}\n")
                if early_stopped:
                    f.write("Early Stopped: True\n")
            logger.info(f"Preliminary results written to {os.path.abspath(temp_results_file)}")
        except Exception as e:
            logger.error(f"Failed to write preliminary results: {e}")
    
    # Load best model if early stopping occurred and we have a saved state
    if early_stopped and best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Using best model checkpoint for final evaluation")
            
    # Ensure we're using the best model for final evaluation
    if best_model_state is not None and not early_stopped:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model checkpoint for final evaluation")
    
    # Save Model
    logger.info(f"Saving model to {run_dir}")
    save_file(model.state_dict(), os.path.join(run_dir, "model.safetensors"))
    
    # Final Evaluation
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
    logger.info(f"Final Validation Perplexity: {perplexity:.2f}")
    if early_stopped:
        logger.info(f"Training stopped early at step {global_step} (best PPL: {best_val_ppl:.2f})")
    
    # Final results save
    try:
        final_results_file = os.path.join(run_dir, "results.txt")
        with open(final_results_file, "w") as f:
            f.write(f"Validation Perplexity: {perplexity:.2f}\n")
            if early_stopped:
                f.write("Early Stopped: True\n")
            f.write(f"Final Step: {global_step}\n")
            f.write(f"Best Validation PPL: {best_val_ppl:.2f}\n")
        
        # Verify file exists
        if os.path.exists(final_results_file):
            logger.info(f"Final results successfully written to {os.path.abspath(final_results_file)}")
            logger.info(f"File size: {os.path.getsize(final_results_file)} bytes")
        else:
            logger.error(f"File {final_results_file} DOES NOT EXIST after write operation!")
            
    except Exception as e:
        logger.error(f"Failed to write results file: {e}")
        raise

if __name__ == "__main__":
    main()
