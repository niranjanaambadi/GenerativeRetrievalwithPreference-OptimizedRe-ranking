"""
LLM Re-ranker Module

Implements LLM-based re-ranking using Mistral-7B.
Supports both pointwise and pairwise ranking approaches.
"""

import json
from typing import Dict, List, Optional, Tuple, Union

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from tqdm import tqdm

from .prompts import (
    POINTWISE_PROMPT,
    PAIRWISE_PROMPT,
    LISTWISE_PROMPT,
    format_pointwise_prompt,
    format_pairwise_prompt
)


class LLMReranker:
    """LLM-based re-ranker using generative models."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.3",
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        use_flash_attention: bool = True,
        max_length: int = 2048
    ):
        """
        Initialize LLM re-ranker.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use
            load_in_4bit: Use 4-bit quantization
            use_flash_attention: Use Flash Attention 2
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        logger.info(f"Loading {model_name}...")
        
        # Quantization config
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
            device_map="auto",
            attn_implementation="flash_attention_2" if use_flash_attention else "eager",
            trust_remote_code=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def _get_relevance_score(
        self,
        query: str,
        document: str,
        prompt_template: str = POINTWISE_PROMPT
    ) -> float:
        """
        Get relevance score for a single query-document pair.
        
        Uses pointwise scoring based on model's probability of "Yes" vs "No".
        """
        prompt = format_pointwise_prompt(query, document, prompt_template)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            
            # Get token IDs for "Yes" and "No"
            yes_tokens = self.tokenizer.encode("Yes", add_special_tokens=False)
            no_tokens = self.tokenizer.encode("No", add_special_tokens=False)
            
            yes_logit = logits[0, yes_tokens[0]].item()
            no_logit = logits[0, no_tokens[0]].item()
            
            # Convert to probability
            prob = torch.softmax(
                torch.tensor([yes_logit, no_logit]),
                dim=0
            )[0].item()
        
        return prob
    
    def _compare_documents(
        self,
        query: str,
        doc_a: str,
        doc_b: str,
        prompt_template: str = PAIRWISE_PROMPT
    ) -> str:
        """
        Compare two documents for a query.
        
        Returns "A" or "B" indicating which is more relevant.
        """
        prompt = format_pairwise_prompt(query, doc_a, doc_b, prompt_template)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Parse response
        if "A" in response.upper()[:3]:
            return "A"
        elif "B" in response.upper()[:3]:
            return "B"
        else:
            return "A"  # Default to first
    
    def rerank_pointwise(
        self,
        query: str,
        documents: Dict[str, str],
        top_k: int = 10,
        show_progress: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents using pointwise scoring.
        
        Args:
            query: Query string
            documents: Dict mapping doc_id -> document text
            top_k: Number of results to return
            show_progress: Show progress bar
            
        Returns:
            List of (doc_id, score) tuples sorted by relevance
        """
        scores = {}
        
        iterator = tqdm(documents.items(), desc="Scoring") if show_progress else documents.items()
        for doc_id, doc_text in iterator:
            scores[doc_id] = self._get_relevance_score(query, doc_text)
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]
    
    def rerank_pairwise(
        self,
        query: str,
        documents: Dict[str, str],
        top_k: int = 10,
        show_progress: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents using pairwise comparisons (bubble sort style).
        
        More expensive but often more accurate than pointwise.
        """
        doc_ids = list(documents.keys())
        n = len(doc_ids)
        
        # Simple bubble sort with pairwise comparisons
        scores = {did: 0 for did in doc_ids}
        
        comparisons = []
        for i in range(n):
            for j in range(i + 1, min(i + 5, n)):  # Limited comparisons
                comparisons.append((doc_ids[i], doc_ids[j]))
        
        iterator = tqdm(comparisons, desc="Comparing") if show_progress else comparisons
        for doc_a, doc_b in iterator:
            winner = self._compare_documents(
                query,
                documents[doc_a],
                documents[doc_b]
            )
            
            if winner == "A":
                scores[doc_a] += 1
            else:
                scores[doc_b] += 1
        
        # Sort by wins
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Normalize scores
        max_score = max(s for _, s in ranked) if ranked else 1
        ranked = [(did, score / max_score) for did, score in ranked]
        
        return ranked[:top_k]
    
    def rerank(
        self,
        query: str,
        documents: Dict[str, str],
        top_k: int = 10,
        method: str = "pointwise",
        show_progress: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents.
        
        Args:
            query: Query string
            documents: Dict mapping doc_id -> document text
            top_k: Number of results
            method: "pointwise" or "pairwise"
            show_progress: Show progress bar
        """
        if method == "pointwise":
            return self.rerank_pointwise(query, documents, top_k, show_progress)
        elif method == "pairwise":
            return self.rerank_pairwise(query, documents, top_k, show_progress)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def batch_rerank(
        self,
        queries: Dict[str, str],
        candidates_per_query: Dict[str, Dict[str, str]],
        top_k: int = 10,
        method: str = "pointwise",
        show_progress: bool = True
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch re-rank for multiple queries.
        
        Args:
            queries: Dict mapping query_id -> query text
            candidates_per_query: Dict mapping query_id -> {doc_id -> doc_text}
            top_k: Number of results per query
            method: Ranking method
            show_progress: Show progress bar
            
        Returns:
            Dict mapping query_id -> list of (doc_id, score)
        """
        results = {}
        
        iterator = tqdm(queries.items(), desc="Re-ranking") if show_progress else queries.items()
        for qid, query in iterator:
            if qid in candidates_per_query:
                results[qid] = self.rerank(
                    query,
                    candidates_per_query[qid],
                    top_k,
                    method,
                    show_progress=False
                )
        
        return results


class DPOReranker(LLMReranker):
    """Re-ranker with DPO fine-tuned model."""
    
    def __init__(
        self,
        model_name: str = "checkpoints/dpo-mistral-7b",
        **kwargs
    ):
        """
        Initialize DPO re-ranker.
        
        Args:
            model_name: Path to DPO fine-tuned model
            **kwargs: Additional arguments for LLMReranker
        """
        super().__init__(model_name=model_name, **kwargs)
        
        # Try to load LoRA adapter if present
        try:
            from peft import PeftModel
            
            adapter_path = f"{model_name}/adapter"
            if Path(adapter_path).exists():
                logger.info(f"Loading LoRA adapter from {adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapter_path
                )
        except Exception as e:
            logger.warning(f"Could not load adapter: {e}")


if __name__ == "__main__":
    # Example usage
    reranker = LLMReranker(
        model_name="mistralai/Mistral-7B-v0.3",
        load_in_4bit=True
    )
    
    query = "What is machine learning?"
    documents = {
        "doc1": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "doc2": "The weather today is sunny with a high of 75 degrees.",
        "doc3": "Deep learning is a type of machine learning using neural networks with many layers."
    }
    
    results = reranker.rerank(query, documents, top_k=3, show_progress=True)
    
    print("\nRanking results:")
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}")
