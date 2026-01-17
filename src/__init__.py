"""
Generative Retrieval with Preference-Optimized Re-ranking

A hybrid search pipeline combining:
- BM25 sparse retrieval
- Dense bi-encoder retrieval  
- LLM-based re-ranking with DPO
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from . import retrieval
from . import rerank
from . import preference
from . import eval
from . import demo
