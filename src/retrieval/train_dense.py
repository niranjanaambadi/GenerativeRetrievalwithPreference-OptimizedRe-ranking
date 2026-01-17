"""
Dense Encoder Training Script

Trains a bi-encoder model for dense retrieval using contrastive learning.
Supports distributed training via torchrun.

Usage:
    # Single GPU
    python train_dense.py --config configs/dense_encoder.yaml
    
    # Multi-GPU
    torchrun --nproc_per_node=4 train_dense.py --config configs/dense_encoder.yaml --distributed
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class RetrievalDataset(Dataset):
    """Dataset for contrastive retrieval training."""
    
    def __init__(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        negatives: Optional[Dict[str, List[str]]] = None,
        num_negatives: int = 7
    ):
        """
        Args:
            queries: Dict mapping query_id -> query text
            corpus: Dict mapping doc_id -> document text
            qrels: Dict mapping query_id -> {doc_id -> relevance}
            negatives: Optional hard negatives per query
            num_negatives: Number of negatives to sample
        """
        self.queries = queries
        self.corpus = corpus
        self.qrels = qrels
        self.negatives = negatives
        self.num_negatives = num_negatives
        
        # Build training examples
        self.examples = []
        self.all_doc_ids = list(corpus.keys())
        
        for qid, query in queries.items():
            if qid not in qrels:
                continue
            
            # Get positive documents
            positives = [
                did for did, rel in qrels[qid].items()
                if rel > 0 and did in corpus
            ]
            
            if not positives:
                continue
            
            for pos_id in positives:
                self.examples.append({
                    "qid": qid,
                    "query": query,
                    "pos_id": pos_id,
                    "pos_text": corpus[pos_id]
                })
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        
        # Sample negatives
        if self.negatives and example["qid"] in self.negatives:
            neg_ids = random.sample(
                self.negatives[example["qid"]],
                min(self.num_negatives, len(self.negatives[example["qid"]]))
            )
        else:
            # Random negatives
            positives = set(self.qrels.get(example["qid"], {}).keys())
            neg_ids = []
            while len(neg_ids) < self.num_negatives:
                neg_id = random.choice(self.all_doc_ids)
                if neg_id not in positives and neg_id not in neg_ids:
                    neg_ids.append(neg_id)
        
        neg_texts = [self.corpus[nid] for nid in neg_ids if nid in self.corpus]
        
        return {
            "query": example["query"],
            "positive": example["pos_text"],
            "negatives": neg_texts
        }


def collate_fn(batch: List[Dict]) -> Dict[str, List[str]]:
    """Collate function for DataLoader."""
    queries = [b["query"] for b in batch]
    positives = [b["positive"] for b in batch]
    
    # Flatten negatives
    all_negatives = []
    neg_counts = []
    for b in batch:
        all_negatives.extend(b["negatives"])
        neg_counts.append(len(b["negatives"]))
    
    return {
        "queries": queries,
        "positives": positives,
        "negatives": all_negatives,
        "neg_counts": neg_counts
    }


class BiEncoderTrainer:
    """Trainer for bi-encoder models."""
    
    def __init__(
        self,
        model: SentenceTransformer,
        config: Dict,
        distributed: bool = False
    ):
        self.model = model
        self.config = config
        self.distributed = distributed
        
        # Setup distributed
        if distributed:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.device = torch.device(f"cuda:{self.local_rank}")
            
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            
            self.model = self.model.to(self.device)
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=config.get("distributed", {}).get("find_unused_parameters", False)
            )
        else:
            self.local_rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
        
        self.is_main_process = self.local_rank == 0
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        evaluator: Optional[InformationRetrievalEvaluator] = None
    ):
        """Train the model."""
        config = self.config["training"]
        
        # DataLoader
        if self.distributed:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True
            )
        else:
            sampler = None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        # Scheduler
        total_steps = len(train_loader) * config["epochs"]
        warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss
        temperature = config.get("temperature", 0.05)
        
        # Training loop
        global_step = 0
        best_metric = 0.0
        
        for epoch in range(config["epochs"]):
            if self.distributed:
                sampler.set_epoch(epoch)
            
            self.model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}",
                disable=not self.is_main_process
            )
            
            for batch in progress_bar:
                # Get model (handle DDP wrapper)
                model = self.model.module if hasattr(self.model, "module") else self.model
                
                # Encode
                query_emb = model.encode(
                    batch["queries"],
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                pos_emb = model.encode(
                    batch["positives"],
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                
                # Normalize
                query_emb = F.normalize(query_emb, p=2, dim=-1)
                pos_emb = F.normalize(pos_emb, p=2, dim=-1)
                
                # In-batch negatives loss
                scores = torch.mm(query_emb, pos_emb.t()) / temperature
                labels = torch.arange(scores.size(0), device=self.device)
                
                loss = F.cross_entropy(scores, labels)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Logging
                if self.is_main_process and global_step % config.get("log_steps", 100) == 0:
                    if WANDB_AVAILABLE:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/step": global_step
                        })
            
            avg_loss = epoch_loss / len(train_loader)
            
            if self.is_main_process:
                logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
                
                # Evaluation
                if evaluator is not None:
                    model = self.model.module if hasattr(self.model, "module") else self.model
                    metrics = evaluator(model)
                    
                    metric_name = self.config.get("evaluation", {}).get("metric_for_best_model", "mrr@10")
                    current_metric = metrics.get(metric_name, 0.0)
                    
                    logger.info(f"Evaluation - {metric_name}: {current_metric:.4f}")
                    
                    if WANDB_AVAILABLE:
                        wandb.log({f"eval/{k}": v for k, v in metrics.items()})
                    
                    # Save best model
                    if current_metric > best_metric:
                        best_metric = current_metric
                        self.save_model(config["output_dir"])
                else:
                    # Save each epoch
                    self.save_model(config["output_dir"])
        
        if self.distributed:
            dist.barrier()
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        if not self.is_main_process:
            return
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.save(str(path))
        
        logger.info(f"Model saved to {path}")


def load_msmarco_data(data_path: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load MS MARCO data in BEIR format."""
    data_path = Path(data_path)
    
    # Load corpus
    corpus = {}
    corpus_file = data_path / "corpus.jsonl"
    if corpus_file.exists():
        with open(corpus_file) as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc["_id"]
                text = doc.get("title", "") + " " + doc.get("text", "")
                corpus[doc_id] = text.strip()
    
    # Load queries
    queries = {}
    queries_file = data_path / "queries.jsonl"
    if queries_file.exists():
        with open(queries_file) as f:
            for line in f:
                q = json.loads(line)
                queries[q["_id"]] = q["text"]
    
    # Load qrels
    qrels = {}
    qrels_file = data_path / "qrels" / "train.tsv"
    if qrels_file.exists():
        with open(qrels_file) as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, did, rel = parts[0], parts[1], int(parts[2])
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][did] = rel
    
    return corpus, queries, qrels, {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        logger.info(f"Config: {json.dumps(config, indent=2)}")
        
        # Initialize wandb
        if WANDB_AVAILABLE and config.get("logging", {}).get("report_to") == "wandb":
            wandb.init(
                project="gen-retrieval-pref",
                name=config.get("logging", {}).get("run_name", "dense-encoder"),
                config=config
            )
    
    # Set seed
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load model
    model_config = config["model"]
    model = SentenceTransformer(model_config["name"])
    model.max_seq_length = model_config.get("max_seq_length", 512)
    
    # Load data
    data_config = config["data"]
    corpus, queries, qrels, negatives = load_msmarco_data(data_config.get("train_path", "data/msmarco"))
    
    if local_rank == 0:
        logger.info(f"Loaded {len(corpus)} documents, {len(queries)} queries")
    
    # Create dataset
    train_dataset = RetrievalDataset(
        queries=queries,
        corpus=corpus,
        qrels=qrels,
        negatives=negatives if negatives else None,
        num_negatives=data_config.get("num_negatives", 7)
    )
    
    # Create trainer
    trainer = BiEncoderTrainer(
        model=model,
        config=config,
        distributed=args.distributed
    )
    
    # Train
    trainer.train(train_dataset)
    
    if local_rank == 0:
        logger.info("Training complete!")
        
        if WANDB_AVAILABLE:
            wandb.finish()


if __name__ == "__main__":
    main()
