"""
Information Retrieval Metrics

Implements standard IR evaluation metrics:
- nDCG@k (Normalized Discounted Cumulative Gain)
- MRR@k (Mean Reciprocal Rank)
- Recall@k
- Precision@k
- MAP (Mean Average Precision)
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger


def dcg_at_k(relevances: List[int], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at k.
    
    DCG@k = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)
    """
    relevances = relevances[:k]
    dcg = 0.0
    
    for i, rel in enumerate(relevances, 1):
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    
    return dcg


def ndcg_at_k(
    relevances: List[int],
    k: int,
    ideal_relevances: Optional[List[int]] = None
) -> float:
    """
    Compute Normalized DCG at k.
    
    nDCG@k = DCG@k / IDCG@k
    """
    dcg = dcg_at_k(relevances, k)
    
    if ideal_relevances is None:
        ideal_relevances = sorted(relevances, reverse=True)
    
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mrr_at_k(relevances: List[int], k: int) -> float:
    """
    Compute Reciprocal Rank at k.
    
    RR@k = 1 / rank of first relevant document (0 if none in top k)
    """
    for i, rel in enumerate(relevances[:k], 1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def recall_at_k(
    relevances: List[int],
    k: int,
    total_relevant: int
) -> float:
    """
    Compute Recall at k.
    
    Recall@k = |relevant docs in top k| / |total relevant docs|
    """
    if total_relevant == 0:
        return 0.0
    
    relevant_in_k = sum(1 for rel in relevances[:k] if rel > 0)
    return relevant_in_k / total_relevant


def precision_at_k(relevances: List[int], k: int) -> float:
    """
    Compute Precision at k.
    
    Precision@k = |relevant docs in top k| / k
    """
    relevant_in_k = sum(1 for rel in relevances[:k] if rel > 0)
    return relevant_in_k / k


def average_precision(
    relevances: List[int],
    total_relevant: int
) -> float:
    """
    Compute Average Precision.
    
    AP = (1/R) * sum_{k=1}^{n} P(k) * rel(k)
    where R is total relevant docs
    """
    if total_relevant == 0:
        return 0.0
    
    ap = 0.0
    relevant_count = 0
    
    for i, rel in enumerate(relevances, 1):
        if rel > 0:
            relevant_count += 1
            ap += relevant_count / i
    
    return ap / total_relevant


class IRMetrics:
    """Class to compute and aggregate IR metrics."""
    
    def __init__(self, k_values: List[int] = [1, 5, 10, 100]):
        """
        Args:
            k_values: List of k values for metrics
        """
        self.k_values = k_values
        self.results = {}
    
    def compute_query_metrics(
        self,
        ranking: List[Tuple[str, float]],
        qrels: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Compute metrics for a single query.
        
        Args:
            ranking: List of (doc_id, score) tuples, sorted by score desc
            qrels: Dict mapping doc_id -> relevance label
            
        Returns:
            Dict of metric_name -> value
        """
        # Get relevances in ranking order
        relevances = [qrels.get(doc_id, 0) for doc_id, _ in ranking]
        
        # Total relevant documents
        total_relevant = sum(1 for rel in qrels.values() if rel > 0)
        
        metrics = {}
        
        for k in self.k_values:
            metrics[f"ndcg@{k}"] = ndcg_at_k(relevances, k)
            metrics[f"mrr@{k}"] = mrr_at_k(relevances, k)
            metrics[f"recall@{k}"] = recall_at_k(relevances, k, total_relevant)
            metrics[f"precision@{k}"] = precision_at_k(relevances, k)
        
        metrics["map"] = average_precision(relevances, total_relevant)
        
        return metrics
    
    def compute_batch_metrics(
        self,
        rankings: Dict[str, List[Tuple[str, float]]],
        all_qrels: Dict[str, Dict[str, int]]
    ) -> Dict[str, float]:
        """
        Compute aggregated metrics over multiple queries.
        
        Args:
            rankings: Dict mapping query_id -> ranking list
            all_qrels: Dict mapping query_id -> qrels dict
            
        Returns:
            Dict of metric_name -> mean value
        """
        all_metrics = []
        
        for qid, ranking in rankings.items():
            if qid not in all_qrels:
                continue
            
            qrels = all_qrels[qid]
            metrics = self.compute_query_metrics(ranking, qrels)
            all_metrics.append(metrics)
        
        if not all_metrics:
            return {}
        
        # Aggregate
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[key] = np.mean(values)
        
        self.results = aggregated
        return aggregated
    
    def print_results(self, results: Optional[Dict[str, float]] = None):
        """Pretty print results."""
        if results is None:
            results = self.results
        
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        
        # Group by metric type
        ndcg_metrics = {k: v for k, v in results.items() if k.startswith("ndcg")}
        mrr_metrics = {k: v for k, v in results.items() if k.startswith("mrr")}
        recall_metrics = {k: v for k, v in results.items() if k.startswith("recall")}
        
        print("\nnDCG:")
        for k, v in sorted(ndcg_metrics.items()):
            print(f"  {k}: {v:.4f}")
        
        print("\nMRR:")
        for k, v in sorted(mrr_metrics.items()):
            print(f"  {k}: {v:.4f}")
        
        print("\nRecall:")
        for k, v in sorted(recall_metrics.items()):
            print(f"  {k}: {v:.4f}")
        
        if "map" in results:
            print(f"\nMAP: {results['map']:.4f}")
        
        print("=" * 50)


def convert_to_trec_format(
    rankings: Dict[str, List[Tuple[str, float]]],
    run_name: str = "run"
) -> str:
    """
    Convert rankings to TREC format.
    
    Format: qid Q0 docid rank score run_name
    """
    lines = []
    
    for qid, ranking in rankings.items():
        for rank, (doc_id, score) in enumerate(ranking, 1):
            lines.append(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}")
    
    return "\n".join(lines)


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    """
    Load qrels from TREC format file.
    
    Format: qid 0 docid relevance
    """
    qrels = {}
    
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = rel
    
    return qrels


if __name__ == "__main__":
    # Example usage
    metrics = IRMetrics(k_values=[1, 5, 10])
    
    # Example ranking and qrels
    ranking = [
        ("doc1", 0.9),
        ("doc2", 0.8),
        ("doc3", 0.7),
        ("doc4", 0.6),
        ("doc5", 0.5)
    ]
    
    qrels = {
        "doc1": 2,  # Highly relevant
        "doc3": 1,  # Relevant
        "doc5": 1   # Relevant
    }
    
    results = metrics.compute_query_metrics(ranking, qrels)
    metrics.print_results(results)
