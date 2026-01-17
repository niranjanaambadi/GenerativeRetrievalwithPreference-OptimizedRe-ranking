"""
BM25 Sparse Retrieval Module

Implements BM25 retrieval for first-stage candidate generation.
Uses rank-bm25 library with preprocessing optimizations.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class BM25Retriever:
    """BM25 retriever for sparse first-stage retrieval."""
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter  
            epsilon: Floor for IDF values
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.bm25 = None
        self.doc_ids = []
        self.documents = []
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()
    
    def index(
        self,
        documents: Dict[str, str],
        show_progress: bool = True
    ) -> None:
        """
        Build BM25 index from documents.
        
        Args:
            documents: Dict mapping doc_id -> document text
            show_progress: Show progress bar
        """
        logger.info(f"Indexing {len(documents)} documents...")
        
        self.doc_ids = list(documents.keys())
        self.documents = list(documents.values())
        
        # Tokenize all documents
        tokenized_docs = []
        iterator = tqdm(self.documents, desc="Tokenizing") if show_progress else self.documents
        for doc in iterator:
            tokenized_docs.append(self._tokenize(doc))
        
        # Build BM25 index
        self.bm25 = BM25Okapi(
            tokenized_docs,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon
        )
        
        logger.info("BM25 index built successfully")
    
    def search(
        self,
        query: str,
        top_k: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call index() first.")
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            (self.doc_ids[idx], float(scores[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def batch_search(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
        show_progress: bool = True
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: Dict mapping query_id -> query text
            top_k: Number of results per query
            show_progress: Show progress bar
            
        Returns:
            Dict mapping query_id -> list of (doc_id, score)
        """
        results = {}
        
        iterator = tqdm(queries.items(), desc="BM25 Search") if show_progress else queries.items()
        for qid, query in iterator:
            results[qid] = self.search(query, top_k)
        
        return results
    
    def save(self, path: str) -> None:
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "doc_ids": self.doc_ids,
                "documents": self.documents,
                "k1": self.k1,
                "b": self.b,
                "epsilon": self.epsilon
            }, f)
        
        logger.info(f"BM25 index saved to {path}")
    
    def load(self, path: str) -> None:
        """Load index from disk."""
        path = Path(path)
        
        with open(path / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        
        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]
        self.documents = data["documents"]
        self.k1 = data["k1"]
        self.b = data["b"]
        self.epsilon = data["epsilon"]
        
        logger.info(f"BM25 index loaded from {path}")


def load_corpus(path: str) -> Dict[str, str]:
    """
    Load corpus from JSONL file.
    
    Expected format: {"_id": "doc_id", "text": "document text", ...}
    """
    corpus = {}
    
    with open(path, "r") as f:
        for line in tqdm(f, desc="Loading corpus"):
            doc = json.loads(line)
            doc_id = doc.get("_id", doc.get("id", doc.get("docid")))
            text = doc.get("text", doc.get("contents", doc.get("body", "")))
            
            # Combine title if available
            title = doc.get("title", "")
            if title:
                text = f"{title} {text}"
            
            corpus[str(doc_id)] = text
    
    return corpus


def load_queries(path: str) -> Dict[str, str]:
    """
    Load queries from JSONL file.
    
    Expected format: {"_id": "query_id", "text": "query text", ...}
    """
    queries = {}
    
    with open(path, "r") as f:
        for line in f:
            query = json.loads(line)
            qid = query.get("_id", query.get("id", query.get("qid")))
            text = query.get("text", query.get("query", ""))
            queries[str(qid)] = text
    
    return queries


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()
    
    # Load data
    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    
    # Build index and search
    retriever = BM25Retriever()
    retriever.index(corpus)
    
    results = retriever.batch_search(queries, top_k=args.top_k)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")
