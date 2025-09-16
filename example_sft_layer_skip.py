#!/usr/bin/env python3
"""
Example script demonstrating SFT with Layer Skipping, Stochastic Dropout, and Early Exit

This is a simplified example that shows how to:
1. Load a pretrained BitNet model
2. Add layer skipping with stochastic dropout
3. Add early exit mechanisms
4. Perform supervised fine-tuning

Usage:
    python example_sft_layer_skip.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import numpy as np

# Example 1: Basic Layer Skipping with Stochastic Dropout
class BitNetWithLayerSkip(nn.Module):
    """BitNet model with layer skipping and stochastic dropout."""
    
    def __init__(self, base_model, skip_probability=0.1, skip_every_n=4):
        super().__init__()
        self.base_model = base_model
        self.skip_probability = skip_probability
        self.skip_every_n = skip_every_n
        self.num_layers = base_model.config.num_hidden_layers
        
        # Stochastic dropout layers for each layer
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(self._compute_dropout_prob(i))
            for i in range(self.num_layers)
        ])
    
    def _compute_dropout_prob(self, layer_idx):
        """Compute dropout probability based on layer depth (quadratic schedule)."""
        return 0.1 * ((layer_idx / (self.num_layers - 1)) ** 2)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get embeddings
        hidden_states = self.base_model.get_input_embeddings()(input_ids)
        
        # Process through layers with skipping
        for i, layer in enumerate(self.base_model.layers):
            # Apply stochastic dropout
            if self.training:
                hidden_states = self.dropout_layers[i](hidden_states)
            
            # Skip every nth layer with probability
            if (i + 1) % self.skip_every_n == 0 and self.training:
                if torch.rand(1).item() < self.skip_probability:
                    continue  # Skip this layer
            
            # Process through layer
            layer_outputs = layer(hidden_states, attention_mask=attention_mask, **kwargs)
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
        
        # Final layer norm and head
        hidden_states = self.base_model.norm(hidden_states)
        logits = self.base_model.lm_head(hidden_states)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.base_model.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states
        }


# Example 2: Early Exit BitNet
class EarlyExitBitNet(nn.Module):
    """BitNet model with early exit capabilities."""
    
    def __init__(self, base_model, exit_every_n=4, exit_loss_weight=0.3):
        super().__init__()
        self.base_model = base_model
        self.exit_every_n = exit_every_n
        self.exit_loss_weight = exit_loss_weight
        self.num_layers = base_model.config.num_hidden_layers
        self.hidden_size = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size
        
        # Exit heads every n layers
        self.exit_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, self.vocab_size)
            for _ in range(self.num_layers // exit_every_n)
        ])
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get embeddings
        hidden_states = self.base_model.get_input_embeddings()(input_ids)
        
        exit_losses = []
        all_hidden_states = []
        
        # Process through layers
        for i, layer in enumerate(self.base_model.layers):
            # Process through layer
            layer_outputs = layer(hidden_states, attention_mask=attention_mask, **kwargs)
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
            
            all_hidden_states.append(hidden_states)
            
            # Check for early exit
            if (i + 1) % self.exit_every_n == 0:
                exit_head_idx = (i + 1) // self.exit_every_n - 1
                
                if exit_head_idx < len(self.exit_heads):
                    # Compute exit logits
                    exit_logits = self.exit_heads[exit_head_idx](hidden_states)
                    
                    # Compute exit loss if labels provided
                    if labels is not None:
                        exit_loss = F.cross_entropy(
                            exit_logits.view(-1, self.vocab_size),
                            labels.view(-1),
                            ignore_index=-100
                        )
                        exit_losses.append(exit_loss)
                    
                    # Simple early exit decision (can be made more sophisticated)
                    if self.training and len(exit_losses) > 0:
                        # During training, exit with some probability
                        if torch.rand(1).item() < 0.3:  # 30% chance to exit
                            logits = exit_logits
                            break
        else:
            # No early exit, use final layer
            hidden_states = self.base_model.norm(hidden_states)
            logits = self.base_model.lm_head(hidden_states)
        
        # Compute total loss
        total_loss = None
        if labels is not None:
            # Main loss
            main_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            
            # Add exit losses
            if exit_losses:
                exit_loss = sum(exit_losses) / len(exit_losses)
                total_loss = main_loss + self.exit_loss_weight * exit_loss
            else:
                total_loss = main_loss
        
        return {
            "loss": total_loss,
            "logits": logits,
            "hidden_states": all_hidden_states
        }


# Example 3: Combined Model with Both Features
class CombinedBitNet(nn.Module):
    """BitNet with both layer skipping and early exit."""
    
    def __init__(self, base_model, skip_prob=0.1, skip_every_n=4, 
                 exit_every_n=4, exit_loss_weight=0.3):
        super().__init__()
        self.base_model = base_model
        self.skip_prob = skip_prob
        self.skip_every_n = skip_every_n
        self.exit_every_n = exit_every_n
        self.exit_loss_weight = exit_loss_weight
        self.num_layers = base_model.config.num_hidden_layers
        self.hidden_size = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size
        
        # Stochastic dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(0.1 * ((i / (self.num_layers - 1)) ** 2))
            for i in range(self.num_layers)
        ])
        
        # Exit heads
        self.exit_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, self.vocab_size)
            for _ in range(self.num_layers // exit_every_n)
        ])
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden_states = self.base_model.get_input_embeddings()(input_ids)
        exit_losses = []
        
        for i, layer in enumerate(self.base_model.layers):
            # Apply stochastic dropout
            if self.training:
                hidden_states = self.dropout_layers[i](hidden_states)
            
            # Skip every nth layer with probability
            if (i + 1) % self.skip_every_n == 0 and self.training:
                if torch.rand(1).item() < self.skip_prob:
                    continue
            
            # Process through layer
            layer_outputs = layer(hidden_states, attention_mask=attention_mask, **kwargs)
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
            
            # Early exit check
            if (i + 1) % self.exit_every_n == 0:
                exit_head_idx = (i + 1) // self.exit_every_n - 1
                
                if exit_head_idx < len(self.exit_heads):
                    exit_logits = self.exit_heads[exit_head_idx](hidden_states)
                    
                    if labels is not None:
                        exit_loss = F.cross_entropy(
                            exit_logits.view(-1, self.vocab_size),
                            labels.view(-1),
                            ignore_index=-100
                        )
                        exit_losses.append(exit_loss)
                    
                    # Early exit decision
                    if self.training and torch.rand(1).item() < 0.2:
                        logits = exit_logits
                        break
        else:
            hidden_states = self.base_model.norm(hidden_states)
            logits = self.base_model.lm_head(hidden_states)
        
        # Compute loss
        total_loss = None
        if labels is not None:
            main_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            
            if exit_losses:
                exit_loss = sum(exit_losses) / len(exit_losses)
                total_loss = main_loss + self.exit_loss_weight * exit_loss
            else:
                total_loss = main_loss
        
        return {
            "loss": total_loss,
            "logits": logits
        }


def main():
    """Example usage of the models."""
    print("Loading pretrained BitNet model...")
    
    # Load model and tokenizer
    model_name = "microsoft/bitnet-b1.58-2B-4T-bf16"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Example 1: Layer Skipping
    print("\n1. Creating model with layer skipping...")
    layer_skip_model = BitNetWithLayerSkip(base_model, skip_probability=0.1, skip_every_n=4)
    
    # Example 2: Early Exit
    print("2. Creating model with early exit...")
    early_exit_model = EarlyExitBitNet(base_model, exit_every_n=4, exit_loss_weight=0.3)
    
    # Example 3: Combined
    print("3. Creating combined model...")
    combined_model = CombinedBitNet(
        base_model, 
        skip_prob=0.1, 
        skip_every_n=4, 
        exit_every_n=4, 
        exit_loss_weight=0.3
    )
    
    # Example SFT training setup
    print("\n4. Setting up SFT training...")
    
    # Create dummy dataset
    dummy_texts = [
        "This is a sample text for training.",
        "Another example text for the model.",
        "The model will learn from these examples."
    ]
    
    dataset = Dataset.from_dict({"text": dummy_texts})
    
    # SFT training configuration
    training_args = SFTConfig(
        output_dir="./bitnet-sft-example",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=1,
        bf16=True,
        max_seq_length=512,
        dataset_text_field="text",
        packing=True,
        save_steps=100,
    )
    
    # Create trainer (using combined model as example)
    trainer = SFTTrainer(
        model=combined_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    print("SFT training setup complete!")
    print("To start training, run: trainer.train()")
    
    # Example inference
    print("\n5. Example inference...")
    test_text = "The future of AI is"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = combined_model(**inputs)
        logits = outputs["logits"]
        predicted_token = tokenizer.decode(torch.argmax(logits[0, -1, :]))
        print(f"Input: {test_text}")
        print(f"Predicted next token: {predicted_token}")


if __name__ == "__main__":
    main()
