import json 
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
import tiktoken

import numpy as np
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from rich.console import Console
from rich.table import Table

# from common import Chunk
from supabase import Client, create_client

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
console = Console()

client: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b")

@dataclass
class Chunk:
    """
    Represents a single chunk of text from a document.
    
    Attributes:
        content (str): The actual text content of the chunk
        metadata (dict): Header hierarchy and optional context
                        Example: {"title": "Revenue", "section": "Q3 Results"}
        vector_text (str|None): Text used for embeddings (content + context)
                               If None, use content directly
    
    Visual representation:
    ┌─────────────────────────────────────────┐
    │ Chunk                                   │
    ├─────────────────────────────────────────┤
    │ content: "Revenue increased by 217%..." │
    │ metadata: {                             │
    │   "title": "NVIDIA Q3 Report",          │
    │   "section": "Data Center",             │
    │   "context": "NVIDIA's Data Center..."  │
    │ }                                       │
    │ vector_text: "[context]\n\n[content]"   │
    └─────────────────────────────────────────┘
    """
    content: str
    metadata: dict[str, Any] = field(default_factory=dict) # field(default_factory=dict): Creates a new empty dict for each instance, Avoids the mutable default argument bug, Each Chunk gets its own metadata dict
    vector_text: str | None = None

    # ------------------------------------------------------------------------
    # PROPERTY: breadcrumb
    # ------------------------------------------------------------------------
    # Creates a hierarchical path showing where this chunk sits in the doc
    # Example: "NVIDIA Report > Financial Results > Revenue Breakdown"
    # ------------------------------------------------------------------------
    @property
    def breadcrumb(self) -> str:
        """
        Generate a breadcrumb trail from the document hierarchy.
        
        Process:
        1. Extract title, section, subsection from metadata
        2. Filter out None values
        3. Join with " > " separator
        
        Example:
        metadata = {"title": "Report", "section": "Revenue", "subsection": None}
        → "Report > Revenue"
        """
        # Get values in order: title (H1) > section (H2) > subsection (H3)
        parts = [self.metadata.get(k) for k in ("title", "section", "subsection")]
        
        # filter(None, parts) removes any None values
        # " > ".join() combines them with arrows
        return " > ".join(filter(None, parts))
    
    # ------------------------------------------------------------------------
    # PROPERTY: token_count
    # ------------------------------------------------------------------------
    # Counts tokens using tiktoken (same tokenizer as GPT models)
    # Important for:
    # - Ensuring chunks fit within model context windows
    # - Calculating embedding costs (charged per token)
    # - Monitoring chunk size distribution
    # ------------------------------------------------------------------------
    @property
    def token_count(self) -> int:
        """
        Count tokens in the chunk content.
        
        Why cl100k_base?
        - This is the encoding used by GPT-4, GPT-3.5, and text-embedding models
        - Ensures accurate token counts for cost estimation
        - Different from character count (e.g., "hello" = 1 token, not 5 chars)
        
        Example:
        "NVIDIA reported revenue" → ~4 tokens
        "NVIDIA reported revenue of $60.9B" → ~8 tokens
        """
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(self.content))


def parse_pgvector(vector_str: str | list[float]) -> list[float]:
    if isinstance(vector_str, list):
        return vector_str
    return [float(x) for x in vector_str.strip("[]").split(",")]


def cosine_similarity(vec1: str | list[float], vec2: str | list[float]) -> float:
    v1 = parse_pgvector(vec1)
    v2 = parse_pgvector(vec2)

# code needed here

console.print("[bold cyan]Embedding Similarity[/bold cyan]\n")

similar_texts = [
    "The company's revenue increased significantly this quarter",
    "The firm's earnings grew substantially this quarter",
    "The new AI chip architecture uses advanced transistors",
]

emb1 = embedding_model.embed_query(similar_texts[0])
emb2 = embedding_model.embed_query(similar_texts[1])
emb3 = embedding_model.embed_query(similar_texts[2])

sim_12 = cosine_similarity(emb1, emb2)
sim_13 = cosine_similarity(emb1, emb3)

console.print(f"Text 1: [dim]{similar_texts[0]}[/dim]")
console.print(f"Text 2: [dim]{similar_texts[1]}[/dim]")
console.print(f"[green]Similarity:[/green] {sim_12:.4f}\n")

console.print(f"Text 1: [dim]{similar_texts[0]}[/dim]")
console.print(f"Text 3: [dim]{similar_texts[2]}[/dim]")
console.print(f"[yellow]Similarity:[/yellow] {sim_13:.4f}\n")

console.print(f"[dim]Embedding dimensions: {len(emb1)}[/dim]\n")
console.print("--" * 80 + "\n")

console.print("[bold cyan]Loading and Storing Chunks with Embeddings[/bold cyan]\n")


chunks_file = DATA_DIR / "mvidia-q3-2026-press-release-chunks.json"
chunks = load_chunks_from_json(chunks_file)

console.print("\n[bold]Chunk Statistics:[/bold]")
console.print(f"  . Total chunks: [cyan]{len(chunks)}[/cyan]")
console.print(
    f"  . Total tokens: [cyan]{sum(chunk.token_count for chunk in chunks)}[/cyan]\n"
)

# Clear existing data and store new chunks
clear_chunks_table()
store_chunks(chunks)

console.print("--" * 80 + "\n")

console.print("[bold cyan]Fetching Chunks from Database[/bold cyan]\n")

all_chunks = fetch_all_chunks()
console.print(f"[green]\/[/green] Retrieved {len(all_chunks)} chunks\n")

display_chunks(all_chunks[:5])

console.print("--" * 80 + "\n")

console.print("[bold cyan]Full-Text Search[/bold cyan]\n")

# Search for chunks about data center - both words appear in the first 3 chunks
query = "data center"
console.print(f"[yellow]Searching for: '{query}' (using AND operator)[/yellow]\n")

# Use AND operator to find chunks containing BOTH words
search_results = search_chunks(query, operator="&")

console.print(f"[green]\/[/green] Found {len(search_results)} results\n")
display_chunks(search_results[:3])

console.print("--" * 80 + "\n")


console.print("[bold cyan]Vector Search (Semantic Search)[/bold cyan]\n")

# Use natural language that won't match exact keywords but finds semantic
# "record-breaking financial results" semantically matches the content ab
query = "record-breaking quarterly financial results"
console.print(f"[yellow]Searching for: '{query}'[/yellow]\n")

vector_results = vector_search_chunks(query, limit=3)

console.print(f"[green]\/[/green] Found {len(vector_results)} results\n")

table = Table(show_header=True, header_style="bold magenta")
table.add_column("#", style="dim", width=4)
table.add_column("Similarity", style="green", width=12)
table.add_column("Preview", style="white", width=70)

for i, (chunk, similarity) in enumerate[tuple[Chunk, float]](vector_results):
    preview =chunks,content[:120].replace("\n", " ")
    preview += "..." if len(chunk.content) > 120 else ""
    table.add_row(str(i), f"{similarity:.4f}", preview)

console.print(table)