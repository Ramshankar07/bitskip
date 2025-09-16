#!/usr/bin/env python3
"""
Quick test for the custom streaming loader used in StreamingTrainer.
Fetches a few batches from HuggingFaceH4/ultrachat_200k in streaming mode,
formats messages -> text, tokenizes, and prints batch shapes.
"""

import os
import logging
from datasets import load_dataset
from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_messages(example):
    messages = example.get("messages", [])
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
    text = text.strip()
    return text if text else None


def create_streaming_loader(dataset_name: str,
                            split: str,
                            tokenizer: AutoTokenizer,
                            max_length: int,
                            batch_size: int,
                            max_samples: int | None):
    def batch_iterator(dataset, batch_size, max_samples=None):
        batch = []
        yielded = 0
        for example in dataset:
            if max_samples is not None and yielded >= max_samples:
                break
            text = format_messages(example)
            if text is None:
                continue
            batch.append(text)
            if len(batch) == batch_size:
                tokenized = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokenized["labels"] = tokenized["input_ids"].clone()
                yield tokenized
                yielded += len(batch)
                batch = []
        if batch and (max_samples is None or yielded < max_samples):
            tokenized = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            yield tokenized

    ds = load_dataset(dataset_name, split=split, streaming=True)
    return batch_iterator(ds, batch_size, max_samples)


def main():
    dataset_name = "HuggingFaceH4/ultrachat_200k"
    split = "train_sft"
    tokenizer_name = "gpt2"
    batch_size = 2
    max_length = 128
    max_samples = 6  # keep very small for a quick test

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Creating streaming loader...")
    loader = create_streaming_loader(
        dataset_name=dataset_name,
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        max_samples=max_samples,
    )

    logger.info("Iterating a few batches...")
    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        logger.info(f"Batch {i+1}: input_ids={tuple(input_ids.shape)}, labels={tuple(labels.shape)}, mask={tuple(attention_mask.shape)}")
        # show a peek of decoded text
        decoded = tokenizer.batch_decode(input_ids[:, :32], skip_special_tokens=True)
        logger.info(f"Sample[0] snippet: {decoded[0][:120].replace('\n', ' / ')}")
        if i >= 2:
            break

    logger.info("Streaming loader test completed.")


if __name__ == "__main__":
    main()


