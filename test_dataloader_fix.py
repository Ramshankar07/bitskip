#!/usr/bin/env python3
"""
Quick test script to verify dataloader fixes work
"""

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

def test_dataloader():
    print("Testing dataloader creation...")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer loaded: {len(tokenizer)} tokens")
    except Exception as e:
        print(f"❌ Tokenizer failed: {e}")
        return
    
    # Test dataset loading
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
        print("✅ Dataset loaded")
    except Exception as e:
        print(f"❌ Dataset failed: {e}")
        return
    
    # Test tokenization
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=256,
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    try:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=["text"]
        )
        print("✅ Tokenization successful")
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return
    
    # Test batch creation
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        
        input_ids = []
        attention_masks = []
        labels = []
        
        for item in batch:
            seq_len = len(item["input_ids"])
            
            padded_input_ids = item["input_ids"] + [tokenizer.pad_token_id] * (max_len - seq_len)
            padded_attention_mask = [1] * seq_len + [0] * (max_len - seq_len)
            padded_labels = item["labels"] + [-100] * (max_len - seq_len)
            
            input_ids.append(padded_input_ids)
            attention_masks.append(padded_attention_mask)
            labels.append(padded_labels)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    
    try:
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=2,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False
        )
        print("✅ DataLoader created")
    except Exception as e:
        print(f"❌ DataLoader creation failed: {e}")
        return
    
    # Test first batch
    try:
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)
        print(f"✅ First batch retrieved: {batch['input_ids'].shape}")
        print("🎉 All tests passed!")
    except Exception as e:
        print(f"❌ First batch failed: {e}")

if __name__ == "__main__":
    test_dataloader()
