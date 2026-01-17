"""
Dense Bi-Encoder Module

Implements dense retrieval using sentence transformers.
Supports both training and inference with FAISS indexing.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class DenseEncoder:
    """Dense bi-encoder for semantic similarity retrieval."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        max_seq_length: int = 512
    ):
        """
        Initialize dense encoder.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use (cuda/cpu)
            max_seq_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length
        
        logger.info(f"Loading model {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length
        
        self.index = None
        self.doc_ids = []
        self.doc_embeddings = None
        
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to dense vectors.
        
        Args:
            texts: Single text or list of texts
            batch_size: Encoding batch size
            show_progress: Show progress bar
            normalize: L2 normalize embeddings
            
        Returns:
            Numpy array of embeddings [N, dim]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def build_index(
        self,
        documents: Dict[str, str],
        batch_size: int = 32,
        index_type: str = "flat",
        nlist: int = 100
    ) -> None:
        """
        Build FAISS index for documents.
        
        Args:
            documents: Dict mapping doc_id -> document text
            batch_size: Encoding batch size
            index_type: "flat" for exact search, "ivf" for approximate
            nlist: Number of clusters for IVF index
        """
        logger.info(f"Building index for {len(documents)} documents...")
        
        self.doc_ids = list(documents.keys())
        doc_texts = list(documents.values())
        
        # Encode documents
        self.doc_embeddings = self.encode(
            doc_texts,
            batch_size=batch_size,
            show_progress=True
        )
        
        dim = self.doc_embeddings.shape[1]
        
        # Create FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine sim
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.doc_embeddings)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.index.add(self.doc_embeddings)
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Query string
            top_k: Number of results to return
            candidates: Optional list of candidate doc_ids to search within
            
        Returns:
            List of (doc_id, score) tuples
        """
        query_embedding = self.encode(query, show_progress=False)
        
        if candidates is not None:
            # Re-rank within candidates
            candidate_indices = [
                self.doc_ids.index(did) for did in candidates 
                if did in self.doc_ids
            ]
            candidate_embeddings = self.doc_embeddings[candidate_indices]
            
            scores = np.dot(candidate_embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = [
                (candidates[idx], float(scores[idx]))
                for idx in top_indices
            ]
        else:
            # Full index search
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = [
                (self.doc_ids[idx], float(score))
                for score, idx in zip(scores[0], indices[0])
                if idx != -1
            ]
        
        return results
    
    def batch_search(
        self,
        queries: Dict[str, str],
        top_k: int = 20,
        batch_size: int = 32,
        candidates_per_query: Optional[Dict[str, List[str]]] = None,
        show_progress: bool = True
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: Dict mapping query_id -> query text
            top_k: Number of results per query
            batch_size: Query encoding batch size
            candidates_per_query: Optional dict of candidate doc_ids per query
            show_progress: Show progress bar
            
        Returns:
            Dict mapping query_id -> list of (doc_id, score)
        """
        results = {}
        
        if candidates_per_query is None:
            # Batch encode all queries
            qids = list(queries.keys())
            query_texts = list(queries.values())
            
            query_embeddings = self.encode(
                query_texts,
                batch_size=batch_size,
                show_progress=show_progress
            )
            
            # Batch search
            scores, indices = self.index.search(query_embeddings, top_k)
            
            for i, qid in enumerate(qids):
                results[qid] = [
                    (self.doc_ids[idx], float(score))
                    for score, idx in zip(scores[i], indices[i])
                    if idx != -1
                ]
        else:
            # Per-query search with candidates
            iterator = tqdm(queries.items(), desc="Dense Search") if show_progress else queries.items()
            for qid, query in iterator:
                candidates = candidates_per_query.get(qid)
                results[qid] = self.search(query, top_k, candidates)
        
        return results
    
    def save(self, path: str) -> None:
        """Save encoder and index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(str(path / "model"))
        
        # Save index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save doc_ids
        with open(path / "doc_ids.json", "w") as f:
            json.dump(self.doc_ids, f)
        
        # Save embeddings
        np.save(path / "embeddings.npy", self.doc_embeddings)
        
        logger.info(f"Dense encoder saved to {path}")
    
    def load(self, path: str) -> None:
        """Load encoder and index from disk."""
        path = Path(path)
        
        # Load model
        self.model = SentenceTransformer(str(path / "model"), device=self.device)
        
        # Load index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load doc_ids
        with open(path / "doc_ids.json", "r") as f:
            self.doc_ids = json.load(f)
        
        # Load embeddings
        self.doc_embeddings = np.load(path / "embeddings.npy")
        
        logger.info(f"Dense encoder loaded from {path}")


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training bi-encoders.
    Uses in-batch negatives with temperature scaling.
    """
    
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        pos_embeddings: torch.Tensor,
        neg_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            query_embeddings: [batch_size, dim]
            pos_embeddings: [batch_size, dim]
            neg_embeddings: Optional [batch_size, num_neg, dim]
            
        Returns:
            Scalar loss
        """
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        pos_embeddings = F.normalize(pos_embeddings, p=2, dim=-1)
        
        # Positive scores
        pos_scores = torch.sum(query_embeddings * pos_embeddings, dim=-1) / self.temperature
        
        if neg_embeddings is not None:
            # Explicit negatives
            neg_embeddings = F.normalize(neg_embeddings, p=2, dim=-1)
            neg_scores = torch.bmm(
                neg_embeddings,
                query_embeddings.unsqueeze(-1)
            ).squeeze(-1) / self.temperature
            
            # Concatenate positive and negative scores
            all_scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
            labels = torch.zeros(all_scores.size(0), dtype=torch.long, device=all_scores.device)
            
            loss = F.cross_entropy(all_scores, labels)
        else:
            # In-batch negatives
            scores = torch.mm(query_embeddings, pos_embeddings.t()) / self.temperature
            labels = torch.arange(scores.size(0), dtype=torch.long, device=scores.device)
            
            loss = F.cross_entropy(scores, labels)
        
        return loss


if __name__ == "__main__":
    # Example usage
    encoder = DenseEncoder()
    
    # Example documents
    documents = {
        "doc1": "Machine learning is a subset of artificial intelligence.",
        "doc2": "Deep learning uses neural networks with many layers.",
        "doc3": "Natural language processing deals with text data.",
    }
    
    # Build index
    encoder.build_index(documents)
    
    # Search
    results = encoder.search("What is deep learning?", top_k=2)
    for doc_id, score in results:
        print(f"{doc_id}: {score:.4f}")
