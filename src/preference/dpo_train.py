"""
DPO Training Script for Re-ranker

Trains Mistral-7B with Direct Preference Optimization for ranking.
Supports distributed training via torchrun.

Usage:
    # Single GPU
    python dpo_train.py --config configs/dpo.yaml
    
    # Multi-GPU
    torchrun --nproc_per_node=4 dpo_train.py --config configs/dpo.yaml --distributed
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import DPOConfig, DPOTrainer

import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_preference_data(
    train_path: str,
    eval_path: Optional[str] = None,
    max_samples: Optional[int] = None
) -> tuple:
    """
    Load preference pairs from JSONL files.
    
    Returns:
        train_dataset, eval_dataset
    """
    
    def load_jsonl(path: str) -> List[Dict]:
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    train_data = load_jsonl(train_path)
    if max_samples:
        train_data = train_data[:max_samples]
    
    # Convert to HF Dataset format for DPO
    train_dataset = Dataset.from_list([
        {
            "prompt": d["prompt"],
            "chosen": d["chosen"],
            "rejected": d["rejected"]
        }
        for d in train_data
    ])
    
    eval_dataset = None
    if eval_path and Path(eval_path).exists():
        eval_data = load_jsonl(eval_path)
        if max_samples:
            eval_data = eval_data[:max_samples // 10]
        eval_dataset = Dataset.from_list([
            {
                "prompt": d["prompt"],
                "chosen": d["chosen"],
                "rejected": d["rejected"]
            }
            for d in eval_data
        ])
    
    return train_dataset, eval_dataset


def setup_model_and_tokenizer(config: Dict):
    """
    Setup model and tokenizer with quantization and LoRA.
    """
    model_config = config["model"]
    lora_config = config["lora"]
    
    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(torch, model_config.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=True
    )
    
    # Load model
    logger.info(f"Loading model: {model_config['name']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2" if model_config.get("use_flash_attention_2") else "eager",
        trust_remote_code=True
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"]
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "left"  # For batch generation
    
    return model, tokenizer, peft_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank == 0:
        logger.info(f"Config: {json.dumps(config, indent=2)}")
        logger.info(f"World size: {world_size}")
        
        # Initialize wandb
        if WANDB_AVAILABLE and config.get("logging", {}).get("report_to") == "wandb":
            wandb.init(
                project="gen-retrieval-pref",
                name=config.get("logging", {}).get("run_name", "dpo-reranker"),
                config=config
            )
    
    # Set seed
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    data_config = config["data"]
    train_dataset, eval_dataset = load_preference_data(
        train_path=data_config["train_path"],
        eval_path=data_config.get("eval_path"),
        max_samples=args.max_samples
    )
    
    if local_rank == 0:
        logger.info(f"Train samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Setup model
    model, tokenizer, peft_config = setup_model_and_tokenizer(config)
    
    # Load reference model for DPO
    ref_model = None  # DPOTrainer will create implicit reference
    
    # Training config
    training_config = config["training"]
    dpo_config = config["dpo"]
    
    # DPO Training Arguments
    training_args = DPOConfig(
        output_dir=training_config["output_dir"],
        per_device_train_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
        max_steps=training_config.get("max_steps", -1),
        num_train_epochs=training_config.get("epochs", 1),
        warmup_steps=training_config.get("warmup_steps", 100),
        
        # DPO specific
        beta=dpo_config["beta"],
        loss_type=dpo_config.get("loss_type", "sigmoid"),
        
        # Optimization
        optim=training_config.get("optim", "paged_adamw_32bit"),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        max_grad_norm=training_config.get("max_grad_norm", 0.3),
        
        # Precision
        bf16=training_config.get("bf16", True),
        tf32=training_config.get("tf32", True),
        
        # Distributed
        gradient_checkpointing=config.get("distributed", {}).get("gradient_checkpointing", True),
        ddp_find_unused_parameters=config.get("distributed", {}).get("ddp_find_unused_parameters", False),
        
        # Logging
        logging_steps=config.get("logging", {}).get("log_steps", 10),
        save_steps=config.get("logging", {}).get("save_steps", 200),
        save_total_limit=config.get("logging", {}).get("save_total_limit", 3),
        report_to="wandb" if WANDB_AVAILABLE and config.get("logging", {}).get("report_to") == "wandb" else "none",
        
        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.get("evaluation", {}).get("eval_steps", 100) if eval_dataset else None,
        
        # Data
        max_length=data_config.get("max_length", 2048),
        max_prompt_length=data_config.get("max_prompt_length", 1024),
        
        # Misc
        remove_unused_columns=False,
        seed=seed
    )
    
    # Initialize DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config
    )
    
    # Train
    if local_rank == 0:
        logger.info("Starting DPO training...")
    
    trainer.train()
    
    # Save final model
    if local_rank == 0:
        logger.info(f"Saving model to {training_config['output_dir']}")
        trainer.save_model(training_config["output_dir"])
        tokenizer.save_pretrained(training_config["output_dir"])
        
        # Save adapter separately
        model.save_pretrained(f"{training_config['output_dir']}/adapter")
        
        logger.info("Training complete!")
        
        if WANDB_AVAILABLE:
            wandb.finish()


if __name__ == "__main__":
    main()
