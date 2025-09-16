
#!/usr/bin/env python3
"""
Simple fine-tuning script for already wrapped/modified model.
Handles compatibility issues and works with the exported model.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from torch.utils.data import IterableDataset
import logging
from typing import Optional, Dict
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingDataset(IterableDataset):
    """Streaming dataset for SFT training."""
    
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
    
    def __iter__(self):
        """Iterate over the streaming dataset."""
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True
        )
        
        count = 0
        for example in dataset:
            if self.max_samples and count >= self.max_samples:
                break
            
            # Format the example
            text = self._format_example(example)
            if not text:
                continue
            
            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Add labels
            tokens["labels"] = tokens["input_ids"].clone()
            
            # Remove batch dimension
            yield {k: v.squeeze(0) for k, v in tokens.items()}
            count += 1
    
    def _format_example(self, example: Dict) -> Optional[str]:
        """Format an example into training text."""
        
        # Handle UltraChat format
        if "messages" in example:
            messages = example["messages"]
            if not messages:
                return None
            
            text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "user":
                    text += f"User: {content}\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n"
            
            return text.strip() if text else None
        
        # Handle plain text format
        if "text" in example:
            return example["text"]
        
        return None

def main():
    """Main training function."""
    
    MODEL_DIR = "./modified_bitnet_model_fixed"  # Your exported model with layer skipping
    OUTPUT_DIR = "./finetuned_output"
    
    DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"
    
    training_config = {
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "warmup_ratio": 0.1,
        "max_seq_length": 512,
        "logging_steps": 10,
        "save_steps": 500,
        "max_samples": 1000,  # Set to None for full dataset
    }
    
    logger.info("=" * 60)
    logger.info("Fine-tuning Model with Layer Skipping")
    logger.info("=" * 60)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {MODEL_DIR}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        logger.info(f"✓ Loaded tokenizer from model directory")
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {MODEL_DIR}: {e}")
        config_path = Path(MODEL_DIR) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_config = json.load(f)
                original_model = model_config.get('_name_or_path', '')
                if original_model and original_model != MODEL_DIR:
                    logger.info(f"Trying to load tokenizer from: {original_model}")
                    tokenizer = AutoTokenizer.from_pretrained(original_model)
                else:
                    raise ValueError(f"No valid tokenizer found")
        else:
            raise ValueError(f"No config.json found in {MODEL_DIR}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info(f"Tokenizer model max length: {tokenizer.model_max_length}")
    
    logger.info(f"Loading model from {MODEL_DIR}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    logger.info("✓ Model loaded successfully")
    
    # Check for layer skip configuration
    if hasattr(model, 'config'):
        logger.info(f"Model type: {model.config.model_type if hasattr(model.config, 'model_type') else 'Unknown'}")
        if hasattr(model.config, 'layer_skip_enabled') and model.config.layer_skip_enabled:
            logger.info("✓ Model has layer skipping enabled")
            if hasattr(model.config, 'layer_skip_probs'):
                logger.info(f"  Skip probabilities: {[f'{p:.3f}' for p in model.config.layer_skip_probs[:5]]}...")
    
    import os
    os.environ['HF_DATASETS_OFFLINE'] = '0'
    os.environ['HF_HOME'] = '/tmp/hf_home_nocache'
    os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_datasets_nocache'
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_nocache'
    
    #streaming dataset with no caching
    logger.info("Creating streaming dataset (no caching)...")
    
    # Import datasets here after setting env vars
    from datasets import load_dataset
    
    # Create custom iterator that truly doesn't cache
    class NoCacheIterableDataset(IterableDataset):
        def __init__(self):
            # Load dataset with streaming, no cache
            self.dataset_iter = load_dataset(
                DATASET_NAME,
                split=DATASET_SPLIT,
                streaming=True,
                keep_in_memory=False,
                cache_dir=None
            )
            
        def __iter__(self):
            count = 0
            for example in self.dataset_iter:
                if training_config["max_samples"] and count >= training_config["max_samples"]:
                    break
                
                # Format messages
                text = ""
                if "messages" in example:
                    for msg in example["messages"]:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "user":
                            text += f"User: {content}\n"
                        elif role == "assistant":
                            text += f"Assistant: {content}\n"
                elif "text" in example:
                    text = example["text"]
                
                if not text:
                    continue
                
                # Tokenize
                tokens = tokenizer(
                    text,
                    truncation=True,
                    max_length=training_config["max_seq_length"],
                    padding="max_length",
                    return_tensors="pt"
                )
                
                tokens["labels"] = tokens["input_ids"].clone()
                yield {k: v.squeeze(0) for k, v in tokens.items()}
                count += 1
    
    train_dataset = NoCacheIterableDataset()
    
    # No eval dataset to avoid compatibility issues
    eval_dataset = None
    
    # Training arguments - simplified for compatibility
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        num_train_epochs=training_config["num_train_epochs"],
        warmup_ratio=training_config["warmup_ratio"],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        save_strategy="steps",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
        dataloader_pin_memory=torch.cuda.is_available(),
        do_eval=False,  # Disable evaluation for compatibility
        load_best_model_at_end=False,  # Disable to avoid eval requirement
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        ),
    )
    
    # Log GPU info if available
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Train
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    try:
        # Train the model
        trainer.train()
        
        # Save final model
        logger.info("\nSaving fine-tuned model...")
        
        # Save with safe_serialization if available
        try:
            trainer.save_model(OUTPUT_DIR, safe_serialization=True)
            logger.info("✓ Model saved in SafeTensors format")
        except:
            trainer.save_model(OUTPUT_DIR)
            logger.info("✓ Model saved")
        
        # Save tokenizer
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info("✓ Tokenizer saved")
        
        # Copy layer skip configuration if it exists
        skip_config_src = Path(MODEL_DIR) / "layer_skip_config.json"
        if skip_config_src.exists():
            import shutil
            skip_config_dst = Path(OUTPUT_DIR) / "layer_skip_config.json"
            shutil.copy2(skip_config_src, skip_config_dst)
            logger.info("✓ Layer skip configuration copied")
        
        logger.info("=" * 60)
        logger.info("✓ Training completed successfully!")
        logger.info(f"✓ Fine-tuned model saved to: {OUTPUT_DIR}")
        logger.info("=" * 60)
        
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU OOM! Try:")
        logger.error("  - Reducing per_device_train_batch_size (currently: {})".format(
            training_config["per_device_train_batch_size"]))
        logger.error("  - Reducing max_seq_length (currently: {})".format(
            training_config["max_seq_length"]))
        logger.error("  - Increasing gradient_accumulation_steps (currently: {})".format(
            training_config["gradient_accumulation_steps"]))
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()