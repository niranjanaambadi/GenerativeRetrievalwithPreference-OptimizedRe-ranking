"""
Prompts for LLM Re-ranking

Contains prompt templates for different ranking approaches:
- Pointwise: Score individual query-document pairs
- Pairwise: Compare two documents for a query
- Listwise: Rank a list of documents
"""

# Pointwise relevance prompt
POINTWISE_PROMPT = """You are a search relevance expert. Determine if the following document is relevant to the query.

Query: {query}

Document: {document}

Is this document relevant to the query? Answer with only "Yes" or "No"."""

# Pairwise comparison prompt
PAIRWISE_PROMPT = """You are a search relevance expert. Given a query and two documents, determine which document is more relevant to the query.

Query: {query}

Document A: {doc_a}

Document B: {doc_b}

Which document is more relevant to the query? Answer with only "A" or "B"."""

# Listwise ranking prompt
LISTWISE_PROMPT = """You are a search relevance expert. Rank the following documents by their relevance to the query.

Query: {query}

Documents:
{documents}

Provide the ranking as a comma-separated list of document numbers (e.g., "3, 1, 2, 4").

Ranking:"""

# Detailed pointwise prompt with scoring rubric
POINTWISE_DETAILED_PROMPT = """You are a search relevance expert evaluating document relevance.

Query: {query}

Document: {document}

Evaluate the document's relevance to the query using this scale:
- Highly Relevant (3): The document directly answers the query
- Relevant (2): The document contains useful information related to the query
- Marginally Relevant (1): The document is tangentially related
- Not Relevant (0): The document is not related to the query

Score (0-3):"""

# Chain-of-thought pairwise prompt
PAIRWISE_COT_PROMPT = """You are a search relevance expert. Compare these two documents for the given query.

Query: {query}

Document A: {doc_a}

Document B: {doc_b}

First, briefly analyze the relevance of each document to the query.
Then, state which document is more relevant.

Analysis:"""

# Instruction-following prompt for chat models
INSTRUCT_POINTWISE_PROMPT = """<s>[INST] You are a search relevance expert.

Given the query: "{query}"

And the document: "{document}"

Is this document relevant to the query? Reply with only "Yes" or "No". [/INST]"""

INSTRUCT_PAIRWISE_PROMPT = """<s>[INST] You are a search relevance expert.

Query: {query}

Document A: {doc_a}

Document B: {doc_b}

Which document is more relevant to the query? Reply with only "A" or "B". [/INST]"""


def format_pointwise_prompt(
    query: str,
    document: str,
    template: str = POINTWISE_PROMPT,
    max_doc_length: int = 500
) -> str:
    """
    Format a pointwise prompt.
    
    Args:
        query: Query string
        document: Document text
        template: Prompt template
        max_doc_length: Maximum document length in characters
        
    Returns:
        Formatted prompt string
    """
    # Truncate document if needed
    if len(document) > max_doc_length:
        document = document[:max_doc_length] + "..."
    
    return template.format(query=query, document=document)


def format_pairwise_prompt(
    query: str,
    doc_a: str,
    doc_b: str,
    template: str = PAIRWISE_PROMPT,
    max_doc_length: int = 400
) -> str:
    """
    Format a pairwise prompt.
    
    Args:
        query: Query string
        doc_a: First document text
        doc_b: Second document text
        template: Prompt template
        max_doc_length: Maximum document length per document
        
    Returns:
        Formatted prompt string
    """
    # Truncate documents if needed
    if len(doc_a) > max_doc_length:
        doc_a = doc_a[:max_doc_length] + "..."
    if len(doc_b) > max_doc_length:
        doc_b = doc_b[:max_doc_length] + "..."
    
    return template.format(query=query, doc_a=doc_a, doc_b=doc_b)


def format_listwise_prompt(
    query: str,
    documents: dict,
    template: str = LISTWISE_PROMPT,
    max_doc_length: int = 200
) -> str:
    """
    Format a listwise prompt.
    
    Args:
        query: Query string
        documents: Dict mapping doc_id -> document text
        template: Prompt template
        max_doc_length: Maximum length per document
        
    Returns:
        Formatted prompt string
    """
    doc_strings = []
    for i, (doc_id, doc_text) in enumerate(documents.items(), 1):
        if len(doc_text) > max_doc_length:
            doc_text = doc_text[:max_doc_length] + "..."
        doc_strings.append(f"[{i}] {doc_text}")
    
    documents_text = "\n\n".join(doc_strings)
    
    return template.format(query=query, documents=documents_text)


# DPO training specific prompts
DPO_CHOSEN_PROMPT = """Query: {query}

The following document is highly relevant to the query:
{document}

This document is relevant because it directly addresses the query."""

DPO_REJECTED_PROMPT = """Query: {query}

The following document is not relevant to the query:
{document}

This document is not relevant because it does not address the query."""


def create_dpo_pair(
    query: str,
    chosen_doc: str,
    rejected_doc: str
) -> dict:
    """
    Create a DPO training pair.
    
    Args:
        query: Query string
        chosen_doc: Preferred (more relevant) document
        rejected_doc: Less preferred (less relevant) document
        
    Returns:
        Dict with prompt, chosen, and rejected responses
    """
    prompt = f"""You are a search relevance expert. Given a query and two documents, determine which document is more relevant.

Query: {query}

Document A: {chosen_doc}

Document B: {rejected_doc}

Which document is more relevant? Answer with only "A" or "B"."""

    return {
        "prompt": prompt,
        "chosen": "A",
        "rejected": "B"
    }
