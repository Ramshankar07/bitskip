"""
Streaming data loader for Hugging Face datasets with full dataset coverage tracking.
"""

import os
from typing import Dict, List, Optional, Union, Iterator
from dataclasses import dataclass
import logging
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, DataLoader
import datasets
from datasets import load_dataset, Dataset, DatasetDict, Features, Value
from transformers import PreTrainedTokenizer
from tqdm import tqdm
from huggingface_hub import dataset_info


def get_dataset_size_from_hub(dataset_name: str, subset: Optional[str] = None, split: str = "train") -> Optional[int]:
    """Get dataset size from HuggingFace Hub metadata"""
    try:
        # Handle subset in dataset name
        full_dataset_name = f"{dataset_name}/{subset}" if subset else dataset_name
        
        info = dataset_info(full_dataset_name)
        
        # Navigate the metadata structure
        if hasattr(info, 'dataset_info'):
            dataset_metadata = info.dataset_info
            if split in dataset_metadata.splits:
                return dataset_metadata.splits[split].num_examples
                
        # Alternative structure for some datasets
        if hasattr(info, 'splits'):
            if split in info.splits:
                return info.splits[split].num_examples
                
    except Exception as e:
        logging.warning(f"Could not get size from hub for {dataset_name}: {e}")
        return None
    
    return None


@dataclass
class StreamingConfig:
    """Configuration for streaming data loader."""
    dataset_name: str
    subset: Optional[str] = None
    split: str = "train"
    text_column: str = "text"
    max_length: int = 4096
    batch_size: int = 8
    num_workers: int = 4
    buffer_size: int = 1000
    seed: int = 42
    cache_dir: Optional[str] = None
    streaming: bool = True
    shuffle: bool = True
    shuffle_buffer_size: int = 10000
    track_coverage: bool = True  # New: track full dataset coverage
    max_epochs: Optional[int] = None  # New: limit training epochs
    max_samples: Optional[int] = None  # New: limit number of samples (for testing)


class HuggingFaceStreamingDataset(IterableDataset):
    """
    Streaming dataset for Hugging Face datasets with coverage tracking.
    
    Args:
        config: Streaming configuration
        tokenizer: Tokenizer for text processing
    """
    
    def __init__(
        self,
        config: StreamingConfig,
        tokenizer: PreTrainedTokenizer
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        
        # Get dataset size for coverage tracking
        self.dataset_size = None
        if config.track_coverage:
            self.dataset_size = get_dataset_size_from_hub(
                config.dataset_name, 
                config.subset, 
                config.split
            )
            if self.dataset_size:
                self.logger.info(f"Dataset size: {self.dataset_size:,} examples")
            else:
                self.logger.warning("Could not determine dataset size - coverage tracking disabled")
                config.track_coverage = False
        
        # Load dataset
        try:
            self.dataset = load_dataset(
                config.dataset_name,
                config.subset,
                split=config.split,
                streaming=config.streaming,
                cache_dir=config.cache_dir
            )
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise
        
        # Validate dataset
        if config.text_column not in self.dataset.features:
            raise ValueError(f"Dataset must contain '{config.text_column}' column")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset with coverage tracking."""
        try:
            # Shuffle if enabled
            if self.config.shuffle:
                self.dataset = self.dataset.shuffle(
                    buffer_size=self.config.shuffle_buffer_size,
                    seed=self.config.seed
                )
            
            # Setup progress tracking
            examples_processed = 0
            epochs_completed = 0
            progress_bar = None
            
            # Determine total examples to process
            total_examples = self.dataset_size if self.config.track_coverage and self.dataset_size else None
            if self.config.max_samples is not None:
                total_examples = min(total_examples, self.config.max_samples) if total_examples else self.config.max_samples
            
            if self.config.track_coverage and total_examples:
                progress_bar = tqdm(
                    total=total_examples,
                    desc=f"Epoch {epochs_completed + 1}",
                    unit="examples"
                )
            
            # Process and yield examples
            for example in self.dataset:
                try:
                    # Check if we've reached max_samples limit
                    if self.config.max_samples is not None and examples_processed >= self.config.max_samples:
                        self.logger.info(f"Reached max_samples limit ({self.config.max_samples})")
                        break
                    
                    # Tokenize text
                    encodings = self.tokenizer(
                        example[self.config.text_column],
                        max_length=self.config.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    # Add labels for causal language modeling
                    encodings["labels"] = encodings["input_ids"].clone()
                    
                    # Convert to tensors
                    yield {
                        key: value.squeeze(0) for key, value in encodings.items()
                    }
                    
                    # Update progress tracking
                    examples_processed += 1
                    
                    if progress_bar:
                        progress_bar.update(1)
                    
                    # Check if epoch is complete
                    if (self.config.track_coverage and 
                        total_examples and 
                        examples_processed >= total_examples):
                        
                        epochs_completed += 1
                        examples_processed = 0
                        
                        if progress_bar:
                            progress_bar.close()
                        
                        # Check if we've reached max epochs
                        if (self.config.max_epochs and 
                            epochs_completed >= self.config.max_epochs):
                            self.logger.info(f"Completed {epochs_completed} epochs - stopping")
                            break
                        
                        # Reset for next epoch
                        if self.config.shuffle:
                            self.dataset = self.dataset.shuffle(
                                buffer_size=self.config.shuffle_buffer_size,
                                seed=self.config.seed + epochs_completed
                            )
                        
                        # Recalculate total examples for next epoch
                        total_examples = self.dataset_size if self.config.track_coverage and self.dataset_size else None
                        if self.config.max_samples is not None:
                            total_examples = min(total_examples, self.config.max_samples) if total_examples else self.config.max_samples
                        
                        if progress_bar and total_examples:
                            progress_bar = tqdm(
                                total=total_examples,
                                desc=f"Epoch {epochs_completed + 1}",
                                unit="examples"
                            )
                        
                        self.logger.info(f"Completed epoch {epochs_completed}")
                    
                except Exception as e:
                    self.logger.warning(f"Error processing example: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in dataset iteration: {str(e)}")
            raise
        finally:
            if progress_bar:
                progress_bar.close()


class StreamingDataLoader:
    """
    Data loader for streaming Hugging Face datasets with coverage tracking.
    
    Args:
        config: Streaming configuration
        tokenizer: Tokenizer for text processing
    """
    
    def __init__(
        self,
        config: StreamingConfig,
        tokenizer: PreTrainedTokenizer
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        
        # Create dataset
        self.dataset = HuggingFaceStreamingDataset(config, tokenizer)
        
        # Create data loader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(
        self,
        examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching examples.
        
        Args:
            examples: List of examples to collate
            
        Returns:
            Collated batch
        """
        try:
            batch = {
                key: torch.stack([example[key] for example in examples])
                for key in examples[0].keys()
            }
            return batch
        except Exception as e:
            self.logger.error(f"Error collating batch: {str(e)}")
            raise
    
    def get_dataloader(self) -> DataLoader:
        """Get the data loader."""
        return self.dataloader
    
    def get_dataset_size(self) -> Optional[int]:
        """Get the dataset size if available."""
        return self.dataset.dataset_size if hasattr(self.dataset, 'dataset_size') else None


def create_streaming_dataloader(
    dataset_name,
    tokenizer,
    subset=None,
    split="train",
    text_column="text",
    max_length=128,
    batch_size=8,
    streaming=True,
    # track_coverage=False,
    # max_epochs=None,
    # max_samples=None,
    # seed=None,
    max_samples=None,  # Add max_samples argument
):
    # Hardcode columns to remove (keep only 'text')
    # If you know the dataset may have other columns, list them here
    # For maximum generality, remove all except 'text' after loading
    def extract_text(example):
        return {text_column: example[text_column]}

    dataset = load_dataset(
        dataset_name,
        name=subset,
        split=split,
        streaming=streaming,
        cache_dir=None,  # Disable caching
    )
    # Remove all columns except 'text' (if present)
    # This works even if there are extra columns in the dataset
    dataset = dataset.map(extract_text, remove_columns=[col for col in dataset._head().keys() if col != text_column])

    def batch_iterator(ds, batch_size, max_samples=None):
        batch = []
        yielded = 0
        for example in ds:
            if max_samples is not None and yielded >= max_samples:
                break
            batch.append(example)
            if len(batch) == batch_size:
                tokenized = tokenizer(
                    [ex[text_column] for ex in batch],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                tokenized["labels"] = tokenized["input_ids"].clone()
                yield tokenized
                yielded += len(batch)
                batch = []
        if batch and (max_samples is None or yielded < max_samples):
            tokenized = tokenizer(
                [ex[text_column] for ex in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            yield tokenized

    return batch_iterator(dataset, batch_size, max_samples=max_samples)


def main():
    """Example usage of streaming data loader with coverage tracking."""
    from transformers import AutoTokenizer
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create streaming loader with coverage tracking
    dataloader = create_streaming_dataloader(
        dataset_name="wikitext",
        subset="wikitext-2-raw-v1",
        tokenizer=tokenizer,
        batch_size=8,
        max_length=512,
        track_coverage=True,
        max_epochs=2  # Train for 2 epochs
    )
    
    # Iterate over batches
    batch_count = 0
    for batch in dataloader:
        # Process batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        batch_count += 1
        
        # Print batch info
        print(f"Batch {batch_count} shape: {input_ids.shape}")
        
        # Stop after a few batches for demo
        if batch_count >= 5:
            break


if __name__ == "__main__":
    main() 