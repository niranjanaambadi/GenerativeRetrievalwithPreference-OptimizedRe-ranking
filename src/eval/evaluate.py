"""
Evaluation Script

Evaluates the full retrieval pipeline on BEIR datasets.
Compares BM25 → Dense → LLM Re-ranker → DPO Re-ranker.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger
from tqdm import tqdm

import yaml

from ..retrieval.bm25 import BM25Retriever, load_corpus, load_queries
from ..retrieval.dense_encoder import DenseEncoder
from ..rerank.llm_reranker import LLMReranker, DPOReranker
from .metrics import IRMetrics, load_qrels


def load_beir_dataset(dataset_name: str, data_dir: str = "data/beir"):
    """
    Load a BEIR dataset.
    
    Returns:
        corpus, queries, qrels
    """
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
        
        data_path = Path(data_dir) / dataset_name
        
        if not data_path.exists():
            logger.info(f"Downloading {dataset_name}...")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            util.download_and_unzip(url, data_dir)
        
        corpus, queries, qrels = GenericDataLoader(str(data_path)).load(split="test")
        
        # Convert corpus format
        corpus_dict = {
            doc_id: doc.get("title", "") + " " + doc.get("text", "")
            for doc_id, doc in corpus.items()
        }
        
        return corpus_dict, queries, qrels
        
    except ImportError:
        logger.warning("BEIR not installed, loading from local files")
        
        data_path = Path(data_dir) / dataset_name
        
        corpus = load_corpus(str(data_path / "corpus.jsonl"))
        queries = load_queries(str(data_path / "queries.jsonl"))
        qrels = load_qrels(str(data_path / "qrels" / "test.tsv"))
        
        return corpus, queries, qrels


class Pipeline:
    """Full retrieval and re-ranking pipeline."""
    
    def __init__(
        self,
        bm25_top_k: int = 100,
        dense_top_k: int = 20,
        rerank_top_k: int = 10,
        dense_model_path: Optional[str] = None,
        reranker_model: str = "mistralai/Mistral-7B-v0.3",
        dpo_model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.rerank_top_k = rerank_top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.bm25 = BM25Retriever()
        
        self.dense = DenseEncoder(
            model_name=dense_model_path or "sentence-transformers/all-MiniLM-L6-v2",
            device=self.device
        )
        
        self.reranker = None
        self.reranker_model = reranker_model
        
        self.dpo_reranker = None
        self.dpo_model_path = dpo_model_path
    
    def index(self, corpus: Dict[str, str]):
        """Build indices for corpus."""
        logger.info("Building BM25 index...")
        self.bm25.index(corpus)
        
        logger.info("Building dense index...")
        self.dense.build_index(corpus)
    
    def load_reranker(self, use_dpo: bool = False):
        """Lazy load reranker models."""
        if use_dpo and self.dpo_reranker is None and self.dpo_model_path:
            logger.info(f"Loading DPO reranker from {self.dpo_model_path}")
            self.dpo_reranker = DPOReranker(
                model_name=self.dpo_model_path,
                load_in_4bit=True
            )
        elif not use_dpo and self.reranker is None:
            logger.info(f"Loading base reranker: {self.reranker_model}")
            self.reranker = LLMReranker(
                model_name=self.reranker_model,
                load_in_4bit=True
            )
    
    def search(
        self,
        query: str,
        corpus: Dict[str, str],
        stages: List[str] = ["bm25", "dense", "rerank_base"]
    ) -> List[Tuple[str, float]]:
        """
        Run search pipeline.
        
        Args:
            query: Query string
            corpus: Document corpus
            stages: Pipeline stages to run
            
        Returns:
            List of (doc_id, score) tuples
        """
        # Stage 1: BM25
        results = self.bm25.search(query, self.bm25_top_k)
        
        if "dense" not in stages and "rerank" not in "".join(stages):
            return results[:self.rerank_top_k]
        
        # Stage 2: Dense re-ranking
        if "dense" in stages:
            candidate_ids = [doc_id for doc_id, _ in results]
            results = self.dense.search(
                query,
                self.dense_top_k,
                candidates=candidate_ids
            )
        
        if "rerank" not in "".join(stages):
            return results[:self.rerank_top_k]
        
        # Stage 3: LLM re-ranking
        use_dpo = "rerank_dpo" in stages
        self.load_reranker(use_dpo)
        
        reranker = self.dpo_reranker if use_dpo else self.reranker
        
        candidate_ids = [doc_id for doc_id, _ in results]
        candidate_docs = {did: corpus[did] for did in candidate_ids if did in corpus}
        
        results = reranker.rerank(
            query,
            candidate_docs,
            top_k=self.rerank_top_k,
            method="pointwise"
        )
        
        return results
    
    def batch_search(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        stages: List[str] = ["bm25", "dense", "rerank_base"],
        show_progress: bool = True
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Batch search for multiple queries."""
        results = {}
        
        iterator = tqdm(queries.items(), desc="Searching") if show_progress else queries.items()
        for qid, query in iterator:
            results[qid] = self.search(query, corpus, stages)
        
        return results


def evaluate_pipeline(
    corpus: Dict[str, str],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    stages: List[str],
    config: Dict
) -> Dict[str, float]:
    """
    Evaluate a pipeline configuration.
    
    Returns:
        Dict of metric -> value
    """
    pipeline = Pipeline(
        bm25_top_k=config.get("bm25_top_k", 100),
        dense_top_k=config.get("dense_top_k", 20),
        rerank_top_k=config.get("rerank_top_k", 10),
        dense_model_path=config.get("dense_model"),
        reranker_model=config.get("reranker_base", "mistralai/Mistral-7B-v0.3"),
        dpo_model_path=config.get("reranker_dpo")
    )
    
    # Build indices
    pipeline.index(corpus)
    
    # Run search
    rankings = pipeline.batch_search(queries, corpus, stages)
    
    # Compute metrics
    metrics = IRMetrics(k_values=[1, 5, 10, 100])
    results = metrics.compute_batch_metrics(rankings, qrels)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--dataset", type=str, default="trec-covid")
    parser.add_argument("--stages", type=str, nargs="+", default=["bm25", "dense", "rerank_dpo"])
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    corpus, queries, qrels = load_beir_dataset(args.dataset)
    
    logger.info(f"Corpus: {len(corpus)} docs, Queries: {len(queries)}, Qrels: {len(qrels)}")
    
    # Evaluate
    logger.info(f"Running evaluation with stages: {args.stages}")
    
    retrieval_config = config.get("retrieval", {})
    models_config = config.get("models", {})
    
    eval_config = {
        **retrieval_config,
        **models_config
    }
    
    results = evaluate_pipeline(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        stages=args.stages,
        config=eval_config
    )
    
    # Print results
    metrics = IRMetrics()
    metrics.print_results(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "stages": args.stages,
            "metrics": results
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
