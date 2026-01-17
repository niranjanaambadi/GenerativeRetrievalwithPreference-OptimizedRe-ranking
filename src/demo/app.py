"""
Gradio Search Demo

Interactive web interface for the search pipeline.
Shows comparison between different pipeline stages.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from loguru import logger

# Import pipeline components
try:
    from ..retrieval.bm25 import BM25Retriever
    from ..retrieval.dense_encoder import DenseEncoder
    from ..rerank.llm_reranker import LLMReranker, DPOReranker
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from retrieval.bm25 import BM25Retriever
    from retrieval.dense_encoder import DenseEncoder
    from rerank.llm_reranker import LLMReranker, DPOReranker


class SearchDemo:
    """Search demo with Gradio interface."""
    
    def __init__(
        self,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "mistralai/Mistral-7B-v0.3",
        dpo_model_path: Optional[str] = None
    ):
        self.corpus = {}
        self.bm25 = BM25Retriever()
        self.dense = DenseEncoder(model_name=dense_model)
        
        self.reranker = None
        self.dpo_reranker = None
        self.reranker_model = reranker_model
        self.dpo_model_path = dpo_model_path
        
        self._indexed = False
    
    def load_sample_data(self):
        """Load sample documents for demo."""
        sample_docs = {
            "doc1": {
                "title": "Introduction to Machine Learning",
                "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve their performance over time without being explicitly programmed. It includes supervised learning, unsupervised learning, and reinforcement learning approaches."
            },
            "doc2": {
                "title": "Deep Learning Fundamentals",
                "text": "Deep learning uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input. This approach has revolutionized computer vision, natural language processing, and speech recognition."
            },
            "doc3": {
                "title": "Natural Language Processing",
                "text": "NLP is a field of AI that focuses on the interaction between computers and human language. It enables machines to read, understand, and derive meaning from human languages. Applications include machine translation, sentiment analysis, and chatbots."
            },
            "doc4": {
                "title": "Computer Vision Applications",
                "text": "Computer vision trains computers to interpret and understand visual information from the world. Applications include image classification, object detection, facial recognition, and autonomous vehicles."
            },
            "doc5": {
                "title": "Reinforcement Learning",
                "text": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward. It has been successful in game playing, robotics, and recommendation systems."
            },
            "doc6": {
                "title": "Transformer Architecture",
                "text": "Transformers are a type of neural network architecture that uses self-attention mechanisms. They have become the foundation for modern NLP models like BERT, GPT, and T5, achieving state-of-the-art results on many benchmarks."
            },
            "doc7": {
                "title": "COVID-19 Vaccine Development",
                "text": "The rapid development of COVID-19 vaccines was a remarkable scientific achievement. mRNA vaccines from Pfizer and Moderna showed over 90% efficacy in clinical trials, while viral vector vaccines from AstraZeneca and Johnson & Johnson provided alternative options."
            },
            "doc8": {
                "title": "Climate Change Effects",
                "text": "Climate change is causing rising global temperatures, sea level rise, and more frequent extreme weather events. Scientists warn that immediate action is needed to reduce greenhouse gas emissions and mitigate the worst effects."
            },
            "doc9": {
                "title": "Search Engine Optimization",
                "text": "SEO is the practice of optimizing websites to rank higher in search engine results. Key factors include content quality, keyword optimization, backlinks, page speed, and mobile-friendliness."
            },
            "doc10": {
                "title": "Information Retrieval Systems",
                "text": "Information retrieval systems help users find relevant information from large collections. Modern systems combine keyword matching with semantic understanding using techniques like BM25, dense retrieval, and neural re-ranking."
            }
        }
        
        self.corpus = {
            doc_id: doc["title"] + " " + doc["text"]
            for doc_id, doc in sample_docs.items()
        }
        
        self.doc_metadata = sample_docs
        self._build_indices()
    
    def load_corpus(self, file_path: str):
        """Load corpus from uploaded file."""
        try:
            with open(file_path) as f:
                for line in f:
                    doc = json.loads(line)
                    doc_id = str(doc.get("_id", doc.get("id")))
                    title = doc.get("title", "")
                    text = doc.get("text", "")
                    self.corpus[doc_id] = f"{title} {text}".strip()
            
            self._build_indices()
            return f"Loaded {len(self.corpus)} documents"
        except Exception as e:
            return f"Error loading corpus: {e}"
    
    def _build_indices(self):
        """Build search indices."""
        if self.corpus:
            self.bm25.index(self.corpus, show_progress=False)
            self.dense.build_index(self.corpus, batch_size=32)
            self._indexed = True
    
    def _load_reranker(self, use_dpo: bool = False):
        """Lazy load reranker."""
        if use_dpo and self.dpo_reranker is None and self.dpo_model_path:
            self.dpo_reranker = DPOReranker(
                model_name=self.dpo_model_path,
                load_in_4bit=True
            )
        elif not use_dpo and self.reranker is None:
            self.reranker = LLMReranker(
                model_name=self.reranker_model,
                load_in_4bit=True
            )
    
    def search(
        self,
        query: str,
        use_dense: bool = True,
        use_rerank: bool = False,
        use_dpo: bool = False,
        top_k: int = 5
    ) -> List[Dict]:
        """Run search and return results."""
        if not self._indexed or not query.strip():
            return []
        
        # BM25
        results = self.bm25.search(query, top_k=100)
        
        # Dense
        if use_dense:
            candidate_ids = [doc_id for doc_id, _ in results]
            results = self.dense.search(query, top_k=20, candidates=candidate_ids)
        
        # Re-rank
        if use_rerank:
            self._load_reranker(use_dpo)
            reranker = self.dpo_reranker if use_dpo else self.reranker
            
            if reranker:
                candidate_ids = [doc_id for doc_id, _ in results]
                candidate_docs = {did: self.corpus[did] for did in candidate_ids}
                results = reranker.rerank(query, candidate_docs, top_k=top_k)
        
        results = results[:top_k]
        
        # Format
        formatted = []
        for rank, (doc_id, score) in enumerate(results, 1):
            text = self.corpus.get(doc_id, "")
            formatted.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(score, 4),
                "text": text[:300] + "..." if len(text) > 300 else text
            })
        
        return formatted
    
    def format_results(self, results: List[Dict]) -> str:
        """Format results as HTML."""
        if not results:
            return "<p>No results found.</p>"
        
        html = ""
        for r in results:
            html += f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px; background: #f9f9f9;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold; color: #2563eb;">#{r['rank']} {r['doc_id']}</span>
                    <span style="background: #e0e7ff; padding: 4px 8px; border-radius: 4px; font-size: 0.9em;">
                        Score: {r['score']}
                    </span>
                </div>
                <p style="margin-top: 10px; color: #374151;">{r['text']}</p>
            </div>
            """
        
        return html
    
    def compare_search(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[str, str, str]:
        """Compare different pipeline configurations."""
        if not self._indexed or not query.strip():
            return "Please enter a query", "", ""
        
        # BM25 only
        bm25_results = self.search(query, use_dense=False, use_rerank=False, top_k=top_k)
        bm25_html = self.format_results(bm25_results)
        
        # BM25 + Dense
        dense_results = self.search(query, use_dense=True, use_rerank=False, top_k=top_k)
        dense_html = self.format_results(dense_results)
        
        # Note for reranker
        rerank_html = """
        <div style="padding: 20px; background: #fef3c7; border-radius: 8px;">
            <p><strong>LLM Re-ranking</strong></p>
            <p>LLM re-ranking is computationally expensive. In a production demo, 
            this would show results from the Mistral-7B re-ranker (with or without DPO).</p>
            <p>Enable this by setting <code>use_rerank=True</code> and ensuring the model is loaded.</p>
        </div>
        """
        
        return bm25_html, dense_html, rerank_html


def create_demo(
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    reranker_model: str = "mistralai/Mistral-7B-v0.3",
    dpo_model_path: Optional[str] = None
):
    """Create Gradio demo interface."""
    
    # Initialize search
    search_demo = SearchDemo(
        dense_model=dense_model,
        reranker_model=reranker_model,
        dpo_model_path=dpo_model_path
    )
    search_demo.load_sample_data()
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .search-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="Generative Retrieval Demo") as demo:
        gr.HTML("""
        <div class="search-header">
            <h1>🔍 Generative Retrieval with Preference-Optimized Re-ranking</h1>
            <p>Compare BM25, Dense Retrieval, and LLM Re-ranking</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your search query...",
                    lines=1
                )
            with gr.Column(scale=1):
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Results"
                )
        
        search_btn = gr.Button("🔍 Search", variant="primary")
        
        with gr.Tabs():
            with gr.Tab("BM25 (Sparse)"):
                bm25_output = gr.HTML(label="BM25 Results")
            
            with gr.Tab("+ Dense Retrieval"):
                dense_output = gr.HTML(label="Dense Results")
            
            with gr.Tab("+ LLM Re-ranking"):
                rerank_output = gr.HTML(label="Re-ranked Results")
        
        # Example queries
        gr.Examples(
            examples=[
                ["What is machine learning?"],
                ["How do transformers work in NLP?"],
                ["COVID-19 vaccine efficacy"],
                ["climate change effects on weather"],
                ["information retrieval neural ranking"]
            ],
            inputs=query_input
        )
        
        # About section
        with gr.Accordion("About this Demo", open=False):
            gr.Markdown("""
            ## Pipeline Architecture
            
            This demo showcases a three-stage retrieval pipeline:
            
            1. **BM25 (Sparse Retrieval)**: Fast keyword-based retrieval using term frequency statistics
            2. **Dense Retrieval**: Neural bi-encoder (MiniLM) for semantic similarity
            3. **LLM Re-ranking**: Mistral-7B with DPO for preference-aligned ranking
            
            ## Technical Details
            
            - **Dense Encoder**: `sentence-transformers/all-MiniLM-L6-v2` (22M params)
            - **Re-ranker**: `mistralai/Mistral-7B-v0.3` (7B params)
            - **DPO Training**: Direct Preference Optimization on synthetic preference pairs
            
            ## Metrics
            
            The pipeline is evaluated on BEIR benchmarks using:
            - nDCG@10 (primary metric)
            - MRR@10
            - Recall@100
            """)
        
        # Event handlers
        search_btn.click(
            fn=search_demo.compare_search,
            inputs=[query_input, top_k_slider],
            outputs=[bm25_output, dense_output, rerank_output]
        )
        
        query_input.submit(
            fn=search_demo.compare_search,
            inputs=[query_input, top_k_slider],
            outputs=[bm25_output, dense_output, rerank_output]
        )
    
    return demo


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--dense_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reranker_model", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--dpo_model", type=str, default=None)
    args = parser.parse_args()
    
    demo = create_demo(
        dense_model=args.dense_model,
        reranker_model=args.reranker_model,
        dpo_model_path=args.dpo_model
    )
    
    demo.launch(
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
