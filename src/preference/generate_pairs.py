"""
Synthetic Preference Pair Generation

Generates preference pairs for DPO training from retrieval data.
Creates (query, chosen_doc, rejected_doc) triplets based on relevance labels.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm


def load_msmarco_data(data_path: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load MS MARCO data.
    
    Returns:
        corpus: Dict[doc_id -> text]
        queries: Dict[query_id -> text]  
        qrels: Dict[query_id -> {doc_id -> relevance}]
    """
    data_path = Path(data_path)
    
    # Load corpus
    corpus = {}
    corpus_file = data_path / "corpus.jsonl"
    if corpus_file.exists():
        logger.info(f"Loading corpus from {corpus_file}")
        with open(corpus_file) as f:
            for line in tqdm(f, desc="Loading corpus"):
                doc = json.loads(line)
                doc_id = str(doc["_id"])
                text = doc.get("title", "") + " " + doc.get("text", "")
                corpus[doc_id] = text.strip()
    
    # Load queries
    queries = {}
    queries_file = data_path / "queries.jsonl"
    if queries_file.exists():
        logger.info(f"Loading queries from {queries_file}")
        with open(queries_file) as f:
            for line in f:
                q = json.loads(line)
                queries[str(q["_id"])] = q["text"]
    
    # Load qrels
    qrels = {}
    for split in ["train", "dev", "test"]:
        qrels_file = data_path / "qrels" / f"{split}.tsv"
        if qrels_file.exists():
            logger.info(f"Loading qrels from {qrels_file}")
            with open(qrels_file) as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        qid, did, rel = str(parts[0]), str(parts[1]), int(parts[2])
                        if qid not in qrels:
                            qrels[qid] = {}
                        qrels[qid][did] = rel
    
    return corpus, queries, qrels


def generate_preference_pairs(
    corpus: Dict[str, str],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    num_pairs_per_query: int = 5,
    max_doc_length: int = 500,
    negative_strategy: str = "random"
) -> List[Dict]:
    """
    Generate preference pairs for DPO training.
    
    Args:
        corpus: Document corpus
        queries: Query texts
        qrels: Relevance judgments
        num_pairs_per_query: Number of pairs to generate per query
        max_doc_length: Maximum document length
        negative_strategy: How to sample negatives ("random", "hard")
        
    Returns:
        List of preference pair dicts
    """
    pairs = []
    all_doc_ids = list(corpus.keys())
    
    for qid, query in tqdm(queries.items(), desc="Generating pairs"):
        if qid not in qrels:
            continue
        
        # Get positive documents
        positives = [
            did for did, rel in qrels[qid].items()
            if rel > 0 and did in corpus
        ]
        
        if not positives:
            continue
        
        # Get negative documents
        positive_set = set(positives)
        
        for _ in range(num_pairs_per_query):
            # Sample positive
            pos_id = random.choice(positives)
            pos_text = corpus[pos_id]
            
            # Sample negative
            if negative_strategy == "random":
                neg_id = random.choice(all_doc_ids)
                while neg_id in positive_set:
                    neg_id = random.choice(all_doc_ids)
            else:
                # For "hard" negatives, would need BM25 retrieval
                neg_id = random.choice(all_doc_ids)
                while neg_id in positive_set:
                    neg_id = random.choice(all_doc_ids)
            
            neg_text = corpus[neg_id]
            
            # Truncate if needed
            if len(pos_text) > max_doc_length:
                pos_text = pos_text[:max_doc_length] + "..."
            if len(neg_text) > max_doc_length:
                neg_text = neg_text[:max_doc_length] + "..."
            
            # Create DPO format
            # Randomly assign A/B to avoid position bias
            if random.random() < 0.5:
                prompt = f"""You are a search relevance expert. Given a query and two documents, determine which document is more relevant.

Query: {query}

Document A: {pos_text}

Document B: {neg_text}

Which document is more relevant? Answer with only "A" or "B"."""
                chosen = "A"
                rejected = "B"
            else:
                prompt = f"""You are a search relevance expert. Given a query and two documents, determine which document is more relevant.

Query: {query}

Document A: {neg_text}

Document B: {pos_text}

Which document is more relevant? Answer with only "A" or "B"."""
                chosen = "B"
                rejected = "A"
            
            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "query_id": qid,
                "pos_doc_id": pos_id,
                "neg_doc_id": neg_id
            })
    
    return pairs


def generate_hard_negative_pairs(
    corpus: Dict[str, str],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    bm25_results: Dict[str, List[Tuple[str, float]]],
    num_pairs_per_query: int = 5,
    max_doc_length: int = 500
) -> List[Dict]:
    """
    Generate preference pairs using BM25 hard negatives.
    
    Hard negatives are documents that BM25 ranked highly but are not relevant.
    """
    pairs = []
    
    for qid, query in tqdm(queries.items(), desc="Generating hard pairs"):
        if qid not in qrels or qid not in bm25_results:
            continue
        
        # Get positive documents
        positives = [
            did for did, rel in qrels[qid].items()
            if rel > 0 and did in corpus
        ]
        
        if not positives:
            continue
        
        # Get hard negatives from BM25 results
        positive_set = set(positives)
        hard_negatives = [
            did for did, score in bm25_results[qid]
            if did not in positive_set and did in corpus
        ][:20]  # Top 20 hard negatives
        
        if not hard_negatives:
            continue
        
        for _ in range(min(num_pairs_per_query, len(hard_negatives))):
            pos_id = random.choice(positives)
            neg_id = random.choice(hard_negatives)
            
            pos_text = corpus[pos_id]
            neg_text = corpus[neg_id]
            
            # Truncate
            if len(pos_text) > max_doc_length:
                pos_text = pos_text[:max_doc_length] + "..."
            if len(neg_text) > max_doc_length:
                neg_text = neg_text[:max_doc_length] + "..."
            
            # Randomize position
            if random.random() < 0.5:
                prompt = f"""You are a search relevance expert. Given a query and two documents, determine which document is more relevant.

Query: {query}

Document A: {pos_text}

Document B: {neg_text}

Which document is more relevant? Answer with only "A" or "B"."""
                chosen = "A"
                rejected = "B"
            else:
                prompt = f"""You are a search relevance expert. Given a query and two documents, determine which document is more relevant.

Query: {query}

Document A: {neg_text}

Document B: {pos_text}

Which document is more relevant? Answer with only "A" or "B"."""
                chosen = "B"
                rejected = "A"
            
            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "query_id": qid,
                "pos_doc_id": pos_id,
                "neg_doc_id": neg_id,
                "is_hard_negative": True
            })
    
    return pairs


def split_data(
    pairs: List[Dict],
    eval_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split pairs into train and eval sets."""
    random.seed(seed)
    random.shuffle(pairs)
    
    split_idx = int(len(pairs) * (1 - eval_ratio))
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]
    
    return train_pairs, eval_pairs


def save_pairs(pairs: List[Dict], output_path: str):
    """Save pairs to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    
    logger.info(f"Saved {len(pairs)} pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="msmarco")
    parser.add_argument("--data_path", type=str, default="data/msmarco")
    parser.add_argument("--output", type=str, default="data/preferences/train.jsonl")
    parser.add_argument("--num_pairs_per_query", type=int, default=5)
    parser.add_argument("--max_doc_length", type=int, default=500)
    parser.add_argument("--negative_strategy", type=str, default="random", 
                       choices=["random", "hard"])
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load data
    logger.info(f"Loading {args.dataset} data from {args.data_path}")
    corpus, queries, qrels = load_msmarco_data(args.data_path)
    
    logger.info(f"Loaded {len(corpus)} documents, {len(queries)} queries, {len(qrels)} qrels")
    
    # Generate pairs
    pairs = generate_preference_pairs(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        num_pairs_per_query=args.num_pairs_per_query,
        max_doc_length=args.max_doc_length,
        negative_strategy=args.negative_strategy
    )
    
    logger.info(f"Generated {len(pairs)} preference pairs")
    
    # Split
    train_pairs, eval_pairs = split_data(pairs, args.eval_ratio, args.seed)
    
    # Save
    output_path = Path(args.output)
    save_pairs(train_pairs, output_path)
    save_pairs(eval_pairs, output_path.parent / "eval.jsonl")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
