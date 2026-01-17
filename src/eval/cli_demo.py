"""
CLI Demo for Search Pipeline

Interactive command-line demo to test the retrieval pipeline.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ..retrieval.bm25 import BM25Retriever
from ..retrieval.dense_encoder import DenseEncoder
from ..rerank.llm_reranker import LLMReranker


class CLIDemo:
    """Interactive CLI demo for search pipeline."""
    
    def __init__(
        self,
        corpus_path: Optional[str] = None,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "mistralai/Mistral-7B-v0.3",
        use_reranker: bool = True
    ):
        self.corpus = {}
        self.bm25 = BM25Retriever()
        self.dense = DenseEncoder(model_name=dense_model)
        
        self.reranker = None
        self.reranker_model = reranker_model
        self.use_reranker = use_reranker
        
        if corpus_path:
            self.load_corpus(corpus_path)
    
    def load_corpus(self, path: str):
        """Load corpus from JSONL file."""
        logger.info(f"Loading corpus from {path}")
        
        with open(path) as f:
            for line in f:
                doc = json.loads(line)
                doc_id = str(doc.get("_id", doc.get("id")))
                text = doc.get("title", "") + " " + doc.get("text", "")
                self.corpus[doc_id] = text.strip()
        
        logger.info(f"Loaded {len(self.corpus)} documents")
        
        # Build indices
        logger.info("Building indices...")
        self.bm25.index(self.corpus)
        self.dense.build_index(self.corpus)
        logger.info("Indices ready!")
    
    def add_document(self, doc_id: str, text: str):
        """Add a single document to the corpus."""
        self.corpus[doc_id] = text
        
        # Rebuild indices (inefficient, but simple for demo)
        self.bm25.index(self.corpus, show_progress=False)
        self.dense.build_index(self.corpus, batch_size=32)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_dense: bool = True,
        use_rerank: bool = False
    ) -> List[Dict]:
        """
        Search for documents.
        
        Returns list of dicts with doc_id, score, text snippet
        """
        # BM25 retrieval
        bm25_results = self.bm25.search(query, top_k=100)
        
        if not use_dense:
            results = bm25_results[:top_k]
        else:
            # Dense re-ranking
            candidate_ids = [doc_id for doc_id, _ in bm25_results]
            results = self.dense.search(query, top_k=20, candidates=candidate_ids)
        
        if use_rerank and self.use_reranker:
            # Load reranker if needed
            if self.reranker is None:
                logger.info(f"Loading reranker: {self.reranker_model}")
                self.reranker = LLMReranker(
                    model_name=self.reranker_model,
                    load_in_4bit=True
                )
            
            # Re-rank
            candidate_ids = [doc_id for doc_id, _ in results]
            candidate_docs = {did: self.corpus[did] for did in candidate_ids}
            results = self.reranker.rerank(query, candidate_docs, top_k=top_k)
        else:
            results = results[:top_k]
        
        # Format results
        formatted = []
        for doc_id, score in results:
            text = self.corpus.get(doc_id, "")
            snippet = text[:200] + "..." if len(text) > 200 else text
            formatted.append({
                "doc_id": doc_id,
                "score": score,
                "snippet": snippet
            })
        
        return formatted
    
    def interactive(self):
        """Run interactive demo."""
        print("\n" + "=" * 60)
        print("  Search Pipeline Demo")
        print("=" * 60)
        print("\nCommands:")
        print("  <query>     - Search for documents")
        print("  :dense      - Toggle dense retrieval")
        print("  :rerank     - Toggle LLM re-ranking")
        print("  :top <k>    - Set number of results")
        print("  :quit       - Exit")
        print("=" * 60)
        
        use_dense = True
        use_rerank = False
        top_k = 5
        
        while True:
            try:
                user_input = input("\n🔍 Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            if user_input == ":quit":
                print("Goodbye!")
                break
            
            if user_input == ":dense":
                use_dense = not use_dense
                print(f"Dense retrieval: {'ON' if use_dense else 'OFF'}")
                continue
            
            if user_input == ":rerank":
                use_rerank = not use_rerank
                print(f"LLM re-ranking: {'ON' if use_rerank else 'OFF'}")
                continue
            
            if user_input.startswith(":top"):
                try:
                    top_k = int(user_input.split()[1])
                    print(f"Top-k set to: {top_k}")
                except:
                    print("Usage: :top <number>")
                continue
            
            # Search
            print(f"\n[Settings: dense={use_dense}, rerank={use_rerank}, top_k={top_k}]")
            
            results = self.search(
                user_input,
                top_k=top_k,
                use_dense=use_dense,
                use_rerank=use_rerank
            )
            
            if not results:
                print("No results found.")
                continue
            
            print(f"\n📚 Results ({len(results)} documents):")
            print("-" * 50)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. [{result['doc_id']}] (score: {result['score']:.4f})")
                print(f"   {result['snippet']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, help="Path to corpus JSONL file")
    parser.add_argument("--query", type=str, help="Single query (non-interactive mode)")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--dense_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reranker_model", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--use_rerank", action="store_true")
    args = parser.parse_args()
    
    # Initialize demo
    demo = CLIDemo(
        corpus_path=args.corpus,
        dense_model=args.dense_model,
        reranker_model=args.reranker_model,
        use_reranker=args.use_rerank
    )
    
    if args.query:
        # Single query mode
        results = demo.search(
            args.query,
            top_k=args.top_k,
            use_dense=True,
            use_rerank=args.use_rerank
        )
        
        print(f"\nQuery: {args.query}")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result['doc_id']}] (score: {result['score']:.4f})")
            print(f"   {result['snippet']}")
    else:
        # Interactive mode
        if not demo.corpus:
            # Load sample data
            print("\nNo corpus specified. Creating sample documents...")
            demo.add_document("doc1", "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve over time.")
            demo.add_document("doc2", "Deep learning uses neural networks with many layers to learn representations of data.")
            demo.add_document("doc3", "Natural language processing deals with the interaction between computers and human language.")
            demo.add_document("doc4", "Computer vision is a field of AI that trains computers to interpret and understand visual information.")
            demo.add_document("doc5", "Reinforcement learning is a type of machine learning where agents learn by interacting with their environment.")
        
        demo.interactive()


if __name__ == "__main__":
    main()
