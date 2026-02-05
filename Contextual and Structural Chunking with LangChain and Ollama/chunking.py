# ============================================================================
# IMPORTS AND SETUP
# ============================================================================
# This script implements a sophisticated document chunking system for RAG
# (Retrieval Augmented Generation) applications. It splits markdown documents
# into semantically meaningful chunks and optionally enriches them with LLM-
# generated context for improved semantic search.
# ============================================================================

from pathlib import Path  # Modern path handling (better than os.path)
from dataclasses import dataclass, field, asdict  # Clean data structure definitions
from typing import Any  # Type hints for better code clarity
import re  # Regular expressions for pattern matching
import json  # JSON serialization for saving chunks
import tiktoken  # OpenAI's tokenizer - counts tokens like GPT models do

# LangChain imports for document processing
from langchain.chat_models import init_chat_model  # Initialize LLM connections
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,  # Splits markdown by headers (#, ##, ###)
    RecursiveCharacterTextSplitter  # Smart text splitter that respects boundaries
)

# Rich library for beautiful terminal output
from rich.panel import Panel  # Display content in bordered panels
from rich.table import Table  # Create formatted tables
from rich.console import Console  # Enhanced console printing

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define data directory relative to this script's location
# Path(__file__) = this script's path
# .parent = the directory containing this script
# / "data" = subdirectory named "data"
DATA_DIR = Path(__file__).parent / "data"

# Initialize Rich console for pretty terminal output
console = Console()

# ============================================================================
# LLM PROMPT FOR CONTEXT ENRICHMENT
# ============================================================================
# This prompt is used to generate contextual summaries for each chunk.
# 
# WHY WE NEED THIS:
# When you split a document into chunks, each chunk loses context about where
# it fits in the larger document. For example, a chunk might say "Revenue 
# increased by 217%" but not mention it's talking about NVIDIA's Data Center
# segment. This makes semantic search less effective.
#
# SOLUTION:
# We ask an LLM to read each chunk in the context of the full document and
# write a 2-3 sentence summary that provides the missing context. This summary
# is then prepended to the chunk when creating embeddings.
#
# EXAMPLE TRANSFORMATION:
# Before: "Revenue increased by 217% year-over-year"
# After:  "NVIDIA's Data Center segment experienced unprecedented growth in
#          fiscal 2024, driven by AI infrastructure demand. Revenue increased
#          by 217% year-over-year, reaching $47.5 billion."
# ============================================================================

CONTEXT_ENRICHMENT_PROMPT = """You are a financial analyst preparing document chunks for semantic search retrieval.

Document excerpt (first 5000 characters)
<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Provide a concise context (2-3 sentences) for this chunk, considering the following guidelines:
1. Identify the main topic or concept discussed in the chunk.
2. Mention any relevant information or comparisons from the broader document context.
3. If applicable, note how this information relates to the overall theme or purpose of the document.
4. Include any key figures, dates, or percentages that provide important context.
5. Do not use phrases like "This chunk discusses" or "This section provides". Instead, directly state the context.

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""

# ============================================================================
# CHUNK DATA STRUCTURE
# ============================================================================
# This dataclass represents a single chunk of text with its metadata.
# 
# DATACLASS BENEFITS:
# - Automatic __init__, __repr__, __eq__ methods
# - Type hints built-in
# - Clean, readable code
#
# CHUNK LIFECYCLE:
# 1. Created with content + metadata (headers from markdown)
# 2. Optionally enriched with LLM-generated context
# 3. Saved to JSON or embedded in vector database
# ============================================================================

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
    
# ============================================================================
# LLM MODEL CONFIGURATION
# ============================================================================
# This initializes the language model used for context enrichment.
# 
# MODEL CHOICE: Qwen3 8B (running locally via Ollama)
# - Open-source alternative to GPT-4
# - Runs on your machine (no API costs, privacy-friendly)
# - 8 billion parameters (good balance of speed vs quality)
#
# ARCHITECTURE FLOW:
# ┌──────────────┐      ┌─────────────┐      ┌──────────────┐
# │ Your Script  │─────>│   Ollama    │─────>│  Qwen3 8B    │
# │              │      │  (Server)   │      │   Model      │
# └──────────────┘      └─────────────┘      └──────────────┘
#     ↑                                              │
#     └──────────────────────────────────────────────┘
#                    Generated context
# ============================================================================

model = init_chat_model(
    # Model identifier - tells Ollama which model to use
    # Format: "model_name:size"
    "qwen3:8b",
    
    # Provider: Ollama is a local LLM runtime (like Docker for AI models)
    # Alternatives: "openai", "anthropic", "cohere", etc.
    model_provider="ollama",
    
    # reasoning=False: Disable chain-of-thought reasoning
    # - Faster responses
    # - We just need direct context generation, not step-by-step thinking
    reasoning=False,
    
    # n_ctx: Context window size (max tokens the model can "see" at once)
    # 16384 tokens ≈ 12,000 words ≈ 40-50 pages of text
    # Large enough to see full document + chunk + prompt
    n_ctx=16384,
    
    # seed: Random seed for reproducibility
    # Same seed = same output for same input
    # Useful for debugging and testing
    seed=42
)

# ============================================================================
# WHY LOCAL vs CLOUD?
# ============================================================================
# LOCAL (Ollama):              CLOUD (OpenAI/Anthropic):
# ✓ No per-token costs         ✓ Better quality
# ✓ Privacy (data stays local) ✓ Faster (no local GPU needed)
# ✓ No rate limits             ✓ More reliable
# ✗ Requires GPU/powerful CPU  ✗ Costs money per call
# ✗ Slower inference           ✗ Data sent to external servers
# ============================================================================

# ============================================================================
# COST COMPARISON EXAMPLE:
# ============================================================================
# If processing 100 chunks with OpenAI GPT-4:
# - ~500 tokens per chunk (context + prompt)
# - 100 chunks × 500 tokens = 50,000 tokens
# - Input: $0.30 (50K × $0.01/1K)
# - Output: $0.60 (50K × $0.03/1K assuming 500 tokens output)
# - Total: ~$0.90 per document
#
# With Qwen3 via Ollama:
# - $0.00 (just electricity + initial setup time)
# ============================================================================

# ============================================================================
# MAIN CHUNKING FUNCTION
# ============================================================================
# This is the core function that orchestrates the entire chunking process.
#
# CHUNKING STRATEGY VISUALIZATION:
# 
# Original Document:
# ┌─────────────────────────────────────────────────────────────┐
# │ # Title (H1)                                                │
# │ Introduction text...                                        │
# │ ## Section A (H2)                                           │
# │ More text about section A...                                │
# │ ### Subsection A.1 (H3)                                     │
# │ Detailed content...                                         │
# │ ### Subsection A.2 (H3)                                     │
# │ More detailed content...                                    │
# │ ## Section B (H2)                                           │
# │ Different topic...                                          │
# └─────────────────────────────────────────────────────────────┘
#
# After Header-Based Splitting:
# ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
# │ Chunk 1         │ │ Chunk 2         │ │ Chunk 3         │
# │ Title + Intro   │ │ Section A +     │ │ Subsection A.2  │
# │ metadata: {     │ │ Subsection A.1  │ │ metadata: {     │
# │  title: "Title" │ │ metadata: {     │ │  title: "..."   │
# │ }               │ │  title: "..."   │ │  section: "A"   │
# │                 │ │  section: "A"   │ │  subsection:    │
# │                 │ │  subsection:    │ │   "A.2"         │
# │                 │ │   "A.1"         │ │ }               │
# │                 │ │ }               │ │                 │
# └─────────────────┘ └─────────────────┘ └─────────────────┘
#
# If a chunk is too large, RecursiveCharacterTextSplitter further divides it
# while trying to preserve semantic boundaries (paragraphs, sentences, etc.)
# ============================================================================

def chunk_markdown(
        markdown: str,              # The full markdown document as a string
        max_tokens: int = 1024,     # Maximum chunk size (default: ~750 words)
        min_tokens: int = 256,      # Minimum chunk size (default: ~190 words)
        chunk_overlap: int = 100,   # Token overlap between chunks
        enrich_with_llm: bool = False  # Whether to add LLM-generated context
) -> list[Chunk]:
    """
    Split a markdown document into semantically meaningful chunks.
    
    PROCESSING PIPELINE:
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │ 1. Cleanup   │──>│ 2. Header    │──>│ 3. Size      │──>│ 4. Merge     │
    │              │   │    Split     │   │    Split     │   │    Small     │
    └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                                                      │
    ┌──────────────┐                                                  │
    │ 5. Optional  │<─────────────────────────────────────────────────┘
    │    Enrich    │
    └──────────────┘
    
    Parameters Explained:
    
    max_tokens (1024):
        - Chunks larger than this get split further
        - Why 1024? Good balance for embeddings:
          * Most embedding models work well with 512-1024 tokens
          * Small enough to be specific
          * Large enough to contain complete thoughts
        - Too small: Loses context, more chunks = more storage/processing
        - Too large: Less precise retrieval, may exceed model limits
    
    min_tokens (256):
        - Chunks smaller than this get merged with adjacent chunks
        - Why 256? Prevents tiny, low-information chunks
        - Example: A 50-token chunk might just be a header with one sentence
    
    chunk_overlap (100):
        - When splitting large chunks, this many tokens repeat between chunks
        - Why overlap? Preserves context at boundaries
        
        Visual example:
        Chunk 1: [........................................overlap]
        Chunk 2:                           [overlap........................................]
        
        - Without overlap: "...in Q3. Revenue increased..." might split to:
          Chunk 1: "...in Q3."
          Chunk 2: "Revenue increased..."
        - With overlap: Both chunks contain "in Q3. Revenue increased..."
    
    enrich_with_llm (False):
        - If True: Call LLM to generate context for each chunk
        - If False: Skip enrichment (faster, cheaper)
    
    Returns:
        list[Chunk]: List of chunk objects with content and metadata
    """
    
    # ------------------------------------------------------------------------
    # STEP 1: INITIALIZE TOKENIZER & CLEAN TEXT
    # ------------------------------------------------------------------------
    # Get the same tokenizer used by OpenAI models
    # This ensures token counts match what embedding APIs expect
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Clean up page break markers (common in PDF→markdown conversions)
    # Replace HTML comments with newlines to avoid joining paragraphs
    text = markdown.replace("<!-- page_break -->", "\n")

    # ------------------------------------------------------------------------
    # STEP 2: CREATE HEADER-AWARE SPLITTER
    # ------------------------------------------------------------------------
    # This splitter understands markdown structure and preserves hierarchy
    # 
    # headers_to_split_on format: (markdown_syntax, metadata_key)
    # - "#"   → H1 → stored as "title"
    # - "##"  → H2 → stored as "section"  
    # - "###" → H3 → stored as "subsection"
    #
    # Example transformation:
    # Input:
    #   # Q3 Report
    #   ## Revenue
    #   Text about revenue...
    #
    # Output:
    #   Chunk(
    #     content="Text about revenue...",
    #     metadata={"title": "Q3 Report", "section": "Revenue"}
    #   )
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),      # H1 headers
            ("##", "section"),   # H2 headers  
            ("###", "subsection")  # H3 headers
        ],
        strip_headers=False,  # Keep "# Title" in content
    )

    # ------------------------------------------------------------------------
    # STEP 3: CREATE TOKEN-AWARE SPLITTER (for oversized chunks)
    # ------------------------------------------------------------------------
    # This splitter uses tiktoken to count tokens (not characters!)
    # 
    # RecursiveCharacterTextSplitter tries splitting at these separators in order:
    # 1. "\n\n" (double newline - paragraph breaks) ← tries first
    # 2. "\n" (single newline - line breaks)
    # 3. " " (spaces - word breaks)
    # 4. "" (characters - last resort)
    #
    # It's "recursive" because if a paragraph is still too large after splitting
    # on "\n\n", it tries splitting on "\n", then " ", etc.
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=max_tokens,      # Target size
        chunk_overlap=chunk_overlap,  # Overlap at boundaries
        encoding_name="cl100k_base"   # Use OpenAI's tokenizer
    )

    # ------------------------------------------------------------------------
    # STEP 4: PROCESS EACH HEADER-BASED SECTION
    # ------------------------------------------------------------------------
    chunks = []
    
    # Split by headers first
    for section in header_splitter.split_text(text):
        # Get the text content and metadata
        content = section.page_content.strip()
        
        # Skip empty sections (e.g., headers with no content)
        if not content:
            continue

        # Count tokens in this section
        token_count = len(tokenizer.encode(content))
        
        # Check if this section contains a markdown table
        # Tables look like:
        # | Header 1 | Header 2 |
        # |----------|----------|
        # | Data 1   | Data 2   |
        #
        # Regex breakdown:
        # ^\|.+\$ - Line starting with | and ending with something
        # re.MULTILINE - Check each line separately
        # "|---" in content - Table separator row
        has_table = (
            bool(re.search(r"^\|.+\$", content, re.MULTILINE)) and "|---" in content
        )

        # DECISION TREE:
        # ┌─────────────────────────────────────┐
        # │ Is this section small enough?       │
        # │ (token_count ≤ max_tokens)          │
        # └────────────┬────────────────────────┘
        #              │
        #      ┌───────┴───────┐
        #     YES              NO
        #      │               │
        #      ▼               ▼
        # ┌─────────┐    ┌──────────────┐
        # │ Has     │    │ Further      │
        # │ table?  │    │ split with   │
        # └────┬────┘    │ Recursive    │
        #      │         │ Splitter     │
        #  ┌───┴───┐     └──────────────┘
        # YES     NO
        #  │       │
        #  ▼       ▼
        # Keep  Keep
        # whole  whole
        # (don't (fits
        # break  in
        # table) limit)
        
        # If it has a table OR fits within max_tokens, keep it whole
        if has_table or token_count <= max_tokens:
            chunks.append(Chunk(content, section.metadata))
        else:
            # Section is too large - split it further while preserving metadata
            # create_documents returns LangChain Document objects
            for sub in text_splitter.create_documents(
                [content],                    # Text to split
                metadatas=[section.metadata]  # Metadata to copy to each chunk
            ):
                chunks.append(
                    Chunk(sub.page_content, sub.metadata)
                )
                
    # ------------------------------------------------------------------------
    # STEP 5: MERGE SMALL CHUNKS
    # ------------------------------------------------------------------------
    # Small chunks are inefficient and lack context
    # This step merges consecutive chunks that are too small
    chunks = _merge_small_chunks(chunks, min_tokens, tokenizer)

    # ------------------------------------------------------------------------
    # STEP 6: OPTIONAL LLM ENRICHMENT
    # ------------------------------------------------------------------------
    # If requested, add contextual summaries to each chunk
    if enrich_with_llm:
        chunks = _enrich_chunks(markdown, chunks)

    return chunks

# ============================================================================
# HELPER FUNCTION: MERGE SMALL CHUNKS
# ============================================================================
# Combines consecutive chunks that are smaller than the minimum token threshold.
#
# WHY THIS MATTERS:
# After header-based splitting, you might get tiny chunks like:
# - Chunk 1: "## Introduction" (5 tokens)
# - Chunk 2: "This section covers..." (50 tokens)
# - Chunk 3: "### Key Points" (10 tokens)
#
# These small chunks are problematic:
# ✗ Lack sufficient context for semantic search
# ✗ Increase storage/processing costs (more embeddings to generate)
# ✗ May contain mostly headers with little substance
#
# MERGING STRATEGY VISUALIZATION:
# Before:
# ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────┐  ┌──────┐
# │ 500  │  │ 100  │  │ 150  │  │  800     │  │ 200  │  (tokens)
# │ ok   │  │ SMALL│  │ SMALL│  │  ok      │  │ SMALL│
# └──────┘  └──────┘  └──────┘  └──────────┘  └──────┘
#
# After (min_tokens=256):
# ┌──────┐  ┌───────────────┐  ┌──────────┐  ┌──────────┐
# │ 500  │  │ 100+150=250   │  │  800     │  │ 800+200  │
# │ ok   │  │ MERGED        │  │  ok      │  │ =1000    │
# └──────┘  └───────────────┘  └──────────┘  └──────────┘
#              ↑ Still small      ↑ ok         ↑ Merged with
#              but no more                        next
#              small chunks
#              after it
# ============================================================================

def _merge_small_chunks(
    chunks: list[Chunk],     # List of chunks to process
    min_tokens: int,         # Minimum acceptable size
    tokenizer                # Tokenizer for counting
) -> list[Chunk]:
    """
    Merge consecutive small chunks to meet minimum token threshold.
    
    Algorithm:
    1. Start with first chunk as baseline
    2. For each subsequent chunk:
       - If it's too small → merge with previous chunk
       - If it's big enough → keep as separate chunk
    3. Return merged list
    
    Important: This is a GREEDY algorithm (single pass, left-to-right)
    - It doesn't optimize globally
    - It merges small chunks with the previous chunk
    - Last chunk might still be small if there's nothing after it
    
    Example walkthrough:
    Input: [500, 100, 150, 800, 200] tokens (min=256)
    
    Step 1: merged = [500]
    Step 2: 100 < 256 → merge → merged = [600]
    Step 3: 150 < 256 → merge → merged = [750]  
    Step 4: 800 ≥ 256 → keep → merged = [750, 800]
    Step 5: 200 < 256 → merge → merged = [750, 1000]
    """
    
    # Edge case: empty list
    if not chunks:
        return []
    
    # Start with the first chunk
    # This will be our "accumulator" - we'll merge small chunks into it
    merged = [chunks[0]]
    
    # Iterate through remaining chunks (starting from index 1)
    for chunk in chunks[1:]:
        # Count tokens in current chunk
        if len(tokenizer.encode(chunk.content)) < min_tokens:
            # This chunk is too small - merge it with the previous chunk
            
            # Get the last chunk from our merged list (the "accumulator")
            prev = merged[-1]
            
            # Create a new merged chunk:
            # - Content: previous content + "\n\n" + current content
            # - Metadata: keep previous chunk's metadata (hierarchical context)
            #   (We could merge metadata too, but keeping prev is simpler)
            #
            # Why "\n\n"? 
            # - Preserves paragraph separation
            # - Prevents sentences from running together
            # - Example: "...in Q3." + "Revenue increased..." → 
            #           "...in Q3.\n\nRevenue increased..."
            merged[-1] = Chunk(
                prev.content + "\n\n" + chunk.content,  # Concatenate with spacing
                prev.metadata                            # Preserve hierarchy
            )
        else:
            # This chunk is large enough - keep it separate
            merged.append(chunk)
    
    return merged


# ============================================================================
# HELPER FUNCTION: ENRICH CHUNKS WITH LLM CONTEXT
# ============================================================================
# Uses an LLM to generate contextual summaries for each chunk.
#
# THE PROBLEM THIS SOLVES:
# When you search for "data center revenue growth", a chunk containing just
# "Revenue increased 217%" won't match well because it lacks context about:
# - What company? (NVIDIA)
# - Which segment? (Data Center)
# - What time period? (Q3 FY2026)
#
# THE SOLUTION:
# Ask an LLM to read each chunk in context and generate a 2-3 sentence
# summary that fills in these gaps. Then, when creating embeddings, we
# use: context + original_content (not just original_content).
#
# ENRICHMENT FLOW:
# ┌─────────────────┐
# │  Full Document  │ (First 5000 chars for context)
# └────────┬────────┘
#          │
#          ▼
# ┌─────────────────────────────────────────────────────────┐
# │  Chunk 1: "Revenue increased by 217% year-over-year"    │
# └────────┬────────────────────────────────────────────────┘
#          │
#          ▼
# ┌─────────────────────────────────────────────────────────┐
# │  LLM (Qwen3)                                            │
# │  Prompt: "Given this document and this chunk,           │
# │           what context is missing?"                     │
# └────────┬────────────────────────────────────────────────┘
#          │
#          ▼
# ┌─────────────────────────────────────────────────────────┐
# │  Generated Context:                                     │
# │  "NVIDIA's Data Center segment experienced              │
# │   unprecedented growth in Q3 FY2026, driven by          │
# │   demand for AI infrastructure. This represents         │
# │   the segment's strongest year-over-year performance."  │
# └────────┬────────────────────────────────────────────────┘
#          │
#          ▼
# ┌─────────────────────────────────────────────────────────┐
# │  Enriched Chunk:                                        │
# │  vector_text = context + "\n\n" + original_content      │
# │                                                         │
# │  This is what gets embedded for semantic search!        │
# └─────────────────────────────────────────────────────────┘
# ============================================================================

def _enrich_chunks(
    full_doc: str,        # The complete document (for context)
    chunks: list[Chunk]   # Chunks to enrich
) -> list[Chunk]:
    """
    Add LLM-generated contextual summaries to each chunk.
    
    Process:
    1. For each chunk, send (document_excerpt + chunk) to LLM
    2. LLM generates a 2-3 sentence contextual summary
    3. Create new Chunk with:
       - Original content preserved
       - Context added to metadata
       - vector_text = context + "\n\n" + content (for embeddings)
    
    Performance considerations:
    - Makes N LLM calls (where N = number of chunks)
    - With 50 chunks: ~2-5 minutes with local model
    - With cloud APIs: faster but costs money
    
    Example transformation:
    
    Before enrichment:
    ───────────────────────────────────────────────────────
    Chunk(
        content="Revenue increased by 217% to $47.5B",
        metadata={"section": "Data Center"},
        vector_text=None
    )
    
    After enrichment:
    ───────────────────────────────────────────────────────
    Chunk(
        content="Revenue increased by 217% to $47.5B",
        metadata={
            "section": "Data Center",
            "context": "NVIDIA's Data Center segment experienced..."
        },
        vector_text="NVIDIA's Data Center segment experienced 
                     unprecedented growth...\n\nRevenue increased 
                     by 217% to $47.5B"
    )
    
    The vector_text is what gets embedded in the vector database!
    """
    
    enriched = []  # List to store enriched chunks
    
    # Process each chunk with a progress counter
    # enumerate(chunks, 1) gives us: (1, chunk1), (2, chunk2), etc.
    for i, chunk in enumerate(chunks, 1):
        # Print progress (dim style = subtle gray text)
        console.print(f"[dim]Enriching chunk {i}/{len(chunks)}...[/dim]")

        # ────────────────────────────────────────────────────────────
        # STEP 1: CREATE THE PROMPT
        # ────────────────────────────────────────────────────────────
        # Format the prompt template with actual values
        # - {document} → first 5000 characters of full document
        #   (5000 chars ≈ 3500 tokens, gives LLM enough context)
        # - {chunk} → the current chunk's content
        prompt = CONTEXT_ENRICHMENT_PROMPT.format(
            document=full_doc[:5000],  # Limit to avoid exceeding context window
            chunk=chunk.content
        )
        
        # ────────────────────────────────────────────────────────────
        # STEP 2: CALL THE LLM
        # ────────────────────────────────────────────────────────────
        # model.invoke() sends the prompt and waits for response
        # Returns a message object with .content attribute
        # .strip() removes leading/trailing whitespace
        context = model.invoke(prompt).content.strip()
        
        # ────────────────────────────────────────────────────────────
        # STEP 3: CLEAN UP THE CONTENT
        # ────────────────────────────────────────────────────────────
        # Remove excessive newlines (3+ in a row → 2)
        # Why? Sometimes markdown has lots of blank lines
        # Example: "text\n\n\n\n\nmore text" → "text\n\nmore text"
        # 
        # Regex breakdown:
        # \n{3,} - Match 3 or more consecutive newlines
        # \n\n   - Replace with exactly 2 newlines
        content = re.sub(r"\n{3,}", "\n\n", chunk.content).strip()

        # ────────────────────────────────────────────────────────────
        # STEP 4: CREATE ENRICHED CHUNK
        # ────────────────────────────────────────────────────────────
        enriched.append(
            Chunk(
                content=content,  # Cleaned original content
                
                # Merge old metadata with new context
                # {**dict1, **dict2} is Python's dictionary merge syntax
                # Example: {**{"a": 1}, **{"b": 2}} → {"a": 1, "b": 2}
                metadata={**chunk.metadata, "context": context},
                
                # This is the key part! vector_text is what gets embedded
                # Format: [CONTEXT]\n\n[ORIGINAL CONTENT]
                # The embedding model will see both when creating vectors
                vector_text=f"{context}\n\n{content}",
            )
        )
    
    return enriched


# ============================================================================
# ENRICHMENT IMPACT EXAMPLE
# ============================================================================
# Original chunk:
# "Revenue increased by 217% year-over-year, reaching $47.5 billion."
#
# Generated context:
# "NVIDIA's Data Center segment experienced unprecedented growth in Q3 
#  FY2026, driven by strong demand for AI infrastructure and accelerated 
#  computing solutions."
#
# Vector text (what gets embedded):
# "NVIDIA's Data Center segment experienced unprecedented growth in Q3 
#  FY2026, driven by strong demand for AI infrastructure and accelerated 
#  computing solutions.
#
#  Revenue increased by 217% year-over-year, reaching $47.5 billion."
#
# Now when someone searches "NVIDIA AI revenue Q3", this chunk will match
# much better because the context contains those keywords!
# ============================================================================

# ============================================================================
# DISPLAY FUNCTION: CHUNK TABLE OVERVIEW
# ============================================================================
# Creates a formatted table showing all chunks at a glance.
#
# EXAMPLE OUTPUT:
# ┌────────────────────────────────────────────────────────────────────────┐
# │                           Chunk Overview                               │
# ├────┬─────────────────────┬────────┬────────────────────────────────────┤
# │  # │ Breadcrumb          │ Tokens │ Preview                            │
# ├────┼─────────────────────┼────────┼────────────────────────────────────┤
# │  1 │ Q3 Report > Revenue │    487 │ NVIDIA reported record revenue...  │
# │  2 │ Q3 Report > Costs   │    312 │ Operating expenses increased...    │
# │  3 │ Q3 Report > Outlook │    256 │ Management expects continued...    │
# └────┴─────────────────────┴────────┴────────────────────────────────────┘
#
# WHY THIS IS USEFUL:
# - Quick sanity check: Are chunks reasonably sized?
# - Identify outliers: Any chunks too large or too small?
# - Verify hierarchy: Do breadcrumbs make sense?
# - Spot issues: Empty chunks, weird token counts, etc.
# ============================================================================

def display_chunk_table(chunks: list[Chunk]) -> None:
    """
    Display chunks in a formatted table using Rich library.
    
    The table shows:
    - Index number (1, 2, 3...)
    - Breadcrumb trail (document hierarchy)
    - Token count (for size estimation)
    - Content preview (first 100 chars)
    
    This is for human review, not machine processing.
    """
    
    # Create a Rich Table object
    # - title: Header text displayed above table
    # - show_header: Display column names
    # - header_style: Color/formatting for column headers
    table = Table(
        title="Chunk Overview",
        show_header=True,
        header_style="bold magenta"  # Makes headers stand out
    )
    
    # ────────────────────────────────────────────────────────────────────
    # DEFINE COLUMNS
    # ────────────────────────────────────────────────────────────────────
    # Each column has:
    # - Header text
    # - Optional formatting (style, width, alignment)
    
    # Column 1: Chunk number
    table.add_column(
        "#",                    # Column header
        style="dim",           # Gray text (not the main focus)
        width=4                # Fixed width (single digit to 3 digits)
    )
    
    # Column 2: Breadcrumb (hierarchical path)
    table.add_column(
        "Breadcrumb",
        min_width=20           # Minimum width, can expand if needed
    )
    
    # Column 3: Token count
    table.add_column(
        "Tokens",
        justify="right",       # Right-align numbers (easier to read)
        width=8                # Fixed width for alignment
    )
    
    # Column 4: Content preview
    table.add_column(
        "Preview",
        min_width=40           # Needs space for text preview
    )
    
    # ────────────────────────────────────────────────────────────────────
    # ADD ROWS
    # ────────────────────────────────────────────────────────────────────
    # enumerate(chunks, 1) gives us: (1, chunk1), (2, chunk2), etc.
    for i, chunk in enumerate(chunks, 1):
        # Create preview text (first 100 characters)
        # - If content > 100 chars: truncate and add "..."
        # - Replace newlines with spaces (keeps preview on one line)
        # 
        # Example:
        # Before: "Revenue increased\nby 217% in Q3..."
        # After:  "Revenue increased by 217% in Q3..."
        if len(chunk.content) > 100:
            preview = chunk.content[:100].replace("\n", " ") + "..."
        else:
            preview = chunk.content.replace("\n", " ")
        
        # Add a row to the table
        # Each argument becomes a cell in the row
        table.add_row(
            str(i),                                    # Convert int to string
            chunk.breadcrumb or "[dim]No headers[/dim]",  # Fallback for missing metadata
            str(chunk.token_count),                    # Convert int to string
            preview
        )
    
    # Print the completed table to terminal
    console.print(table)


# ============================================================================
# DISPLAY FUNCTION: DETAILED CHUNK VIEW
# ============================================================================
# Shows full content of selected chunks in bordered panels.
#
# EXAMPLE OUTPUT:
# ┌─────────────────────────────────────────────────────────────────┐
# │ Chunk 1 - Q3 Report > Revenue                  Tokens: 487      │
# ├─────────────────────────────────────────────────────────────────┤
# │ NVIDIA Corporation reported record revenue of $60.9 billion     │
# │ for the third quarter of fiscal year 2026, representing a       │
# │ 126% increase compared to the same period last year. The        │
# │ growth was primarily driven by unprecedented demand for AI      │
# │ infrastructure and accelerated computing solutions.             │
# │                                                                  │
# │ The Data Center segment led the growth with revenue of          │
# │ $47.5 billion, up 217% year-over-year...                        │
# └─────────────────────────────────────────────────────────────────┘
#
# WHY THIS IS USEFUL:
# - Verify content quality: Is the chunking sensible?
# - Check boundaries: Do chunks start/end at good points?
# - Review structure: Is important context preserved?
# ============================================================================

def display_chunk_details(chunks: list[Chunk], limit: int = 3) -> None:
    """
    Display detailed view of first N chunks in bordered panels.
    
    Parameters:
        chunks: List of chunks to display
        limit: How many chunks to show (default: 3)
               Why 3? Good sample size without overwhelming the terminal
    
    Use case: Deep inspection of chunk quality after splitting
    """
    
    # Only show first 'limit' chunks
    # chunks[:3] is Python slice notation: items 0, 1, 2
    for i, chunk in enumerate(chunks[:limit], 1):
        # Create a Rich Panel (bordered box)
        panel = Panel(
            chunk.content,  # The text to display inside the panel
            
            # Title (top border): Shows chunk number and hierarchy
            # [bold] is Rich markup for bold text
            # Example: "Chunk 1 - Q3 Report > Revenue"
            title=f"[bold]Chunk {i}[/bold] - {chunk.breadcrumb or 'No headers'}",
            
            # Subtitle (bottom border): Shows metadata
            # Example: "Tokens: 487"
            subtitle=f"Tokens: {chunk.token_count}",
            
            # Border color
            border_style="cyan"
        )
        
        console.print(panel)


# ============================================================================
# DISPLAY FUNCTION: ENRICHED CHUNKS WITH CONTEXT
# ============================================================================
# Shows chunks that have been enriched with LLM-generated context.
#
# EXAMPLE OUTPUT:
# ┌─────────────────────────────────────────────────────────────────┐
# │ Enriched Chunk 1 - Q3 Report > Revenue        Tokens: 487       │
# ├─────────────────────────────────────────────────────────────────┤
# │ Context:                                                        │
# │ NVIDIA's Data Center segment experienced unprecedented growth   │
# │ in Q3 FY2026, driven by strong demand for AI infrastructure    │
# │ and accelerated computing solutions. This represents the        │
# │ segment's strongest year-over-year performance.                 │
# │                                                                  │
# │ Content:                                                        │
# │ Revenue increased by 217% year-over-year, reaching $47.5       │
# │ billion. The growth was fueled by hyperscale customers          │
# │ expanding their AI training capabilities...                     │
# └─────────────────────────────────────────────────────────────────┘
#
# COLOR CODING:
# - Yellow: Context (LLM-generated summary)
# - Green: Original content
# - This visual separation helps distinguish what was added vs original
# ============================================================================

def display_enriched_chunks(chunks: list[Chunk], limit: int = 3) -> None:
    """
    Display enriched chunks with context highlighted separately.
    
    Shows both:
    1. The LLM-generated context (what was added)
    2. The original content (truncated to 300 chars for readability)
    
    This helps you evaluate:
    - Is the generated context accurate?
    - Does it add meaningful information?
    - Is it worth the extra cost/processing time?
    """
    
    for i, chunk in enumerate(chunks[:limit], 1):
        # Get the context from metadata
        # .get() returns None if key doesn't exist (safe access)
        # Fallback: "[dim]No context available[/dim]" if missing
        context = chunk.metadata.get("context", "[dim]No context available[/dim]")
        
        # Create formatted content with color coding
        # 
        # Rich markup syntax:
        # [bold yellow]...[/bold yellow] - Bold yellow text
        # [bold green]...[/bold green]   - Bold green text
        # \n\n - Double newline for spacing
        # 
        # Truncate content to 300 chars to keep output manageable
        # chunk.content[:300] + "..." - First 300 chars plus ellipsis
        content_panel = Panel(
            f"[bold yellow]Context:[/bold yellow]\n{context}\n\n"
            f"[bold green]Content:[/bold green]\n{chunk.content[:300]}...",
            
            title=f"[bold]Enriched Chunk {i}[/bold] - {chunk.breadcrumb or 'No headers'}",
            subtitle=f"Tokens: {chunk.token_count}",
            border_style="green"  # Green border to indicate enriched status
        )
        
        console.print(content_panel)


# ============================================================================
# DISPLAY FUNCTION: TOKEN COST ANALYSIS
# ============================================================================
# Compares token counts before and after enrichment.
#
# EXAMPLE OUTPUT:
# ┌─────────────────────────────────────────────────────────────────┐
# │ Token Cost Analysis                                             │
# ├─────────────────────────────────────────────────────────────────┤
# │   • Raw Tokens: 12,450                                          │
# │   • Enriched Tokens (with context): 18,230                      │
# │   • Token Increase: 46.4%                                       │
# │   (This percentage shows the embedding cost increase from       │
# │    adding contextual information)                               │
# └─────────────────────────────────────────────────────────────────┘
#
# WHY THIS MATTERS:
# Embedding APIs charge per token. If enrichment increases tokens by 50%,
# your embedding costs also increase by ~50%. You need to decide if the
# improved search quality is worth the extra cost.
#
# COST EXAMPLE (using OpenAI text-embedding-3-small at $0.02/1M tokens):
# - Raw: 12,450 tokens × $0.02/1M = $0.0002
# - Enriched: 18,230 tokens × $0.02/1M = $0.0004
# - Extra cost: $0.0002 per document
# 
# For 10,000 documents: Extra $2.00 (might be worth it for better search!)
# ============================================================================

def display_cost_analysis(
    basic_chunks: list[Chunk],      # Chunks without enrichment
    enriched_chunks: list[Chunk]    # Chunks with LLM-generated context
) -> None:
    """
    Display token cost comparison between basic and enriched chunks.
    
    Helps answer: "Is enrichment worth the extra embedding cost?"
    """
    
    console.print("\n[bold]Token Cost Analysis[/bold]")
    
    # ────────────────────────────────────────────────────────────────────
    # CALCULATE RAW TOKEN COUNT
    # ────────────────────────────────────────────────────────────────────
    # Sum up token counts from all basic chunks
    # sum([chunk.token_count for chunk in basic_chunks])
    # 
    # Generator expression (more memory efficient than list comprehension):
    # sum(chunk.token_count for chunk in basic_chunks)
    raw_tokens = sum(chunk.token_count for chunk in basic_chunks)
    
    # ────────────────────────────────────────────────────────────────────
    # CALCULATE ENRICHED TOKEN COUNT
    # ────────────────────────────────────────────────────────────────────
    # For enriched chunks, we need to count tokens in vector_text
    # (which includes both context and content)
    # 
    # If vector_text is None, fall back to content
    # This handles edge cases where enrichment might have failed
    tokenizer = tiktoken.get_encoding("cl100k_base")
    enriched_tokens = sum(
        len(tokenizer.encode(chunk.vector_text or chunk.content))
        for chunk in enriched_chunks
    )
    
    # ────────────────────────────────────────────────────────────────────
    # CALCULATE PERCENTAGE INCREASE
    # ────────────────────────────────────────────────────────────────────
    # Formula: (new - old) / old × 100
    # 
    # Example: 
    # - Raw: 10,000 tokens
    # - Enriched: 14,000 tokens
    # - Increase: (14,000 - 10,000) / 10,000 × 100 = 40%
    #
    # Handle division by zero (if no raw tokens for some reason)
    if raw_tokens > 0:
        increase = ((enriched_tokens - raw_tokens) / raw_tokens * 100)
    else:
        increase = 0
    
    # ────────────────────────────────────────────────────────────────────
    # DISPLAY RESULTS
    # ────────────────────────────────────────────────────────────────────
    # {:,} format specifier adds thousands separators
    # Examples: 1234 → "1,234", 1234567 → "1,234,567"
    console.print(f"  • Raw Tokens: {raw_tokens:,}")
    console.print(f"  • Enriched Tokens (with context): {enriched_tokens:,}")
    
    # :.1f format specifier = 1 decimal place
    # Examples: 46.456 → "46.5", 100.0 → "100.0"
    console.print(f"  • Token Increase: {increase:.1f}%")
    
    # Dim text = subtle gray (less prominent)
    console.print(
        "[dim]  (This percentage shows the embedding cost increase from "
        "adding contextual information)[/dim]"
    )



def save_chunks_to_json(chunks: list[Chunk], output_path: Path) -> None:
    """Save chunks to a JSON file"""
    chunks_data = [asdict(chunk) for chunk in chunks]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
    console.print(f"[green]✓[/green] Saved {len(chunks)} chunks to {output_path}")

# Main execution
if __name__ == "__main__":
    console.print("\n[bold]CHUNKING DEMO: Markdown Document Chunking[/bold]\n")

    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    
    # Create a sample markdown file if it doesn't exist
    sample_md_path = DATA_DIR / "sample.md"
    if not sample_md_path.exists():
        console.print("[yellow]Sample file not found. Creating a demo markdown file...[/yellow]\n")
        sample_content = """# NVIDIA Corporation Financial Report

## Executive Summary

NVIDIA Corporation reported strong financial performance for fiscal year 2024, with revenue reaching $60.9 billion, representing a 126% increase year-over-year.

### Revenue Breakdown

| Segment | Revenue (B) | Growth |
|---------|-------------|--------|
| Data Center | $47.5 | 217% |
| Gaming | $10.4 | -2% |
| Professional Visualization | $1.5 | -7% |

## Data Center Performance

The Data Center segment experienced unprecedented growth, driven by demand for AI infrastructure and accelerated computing solutions.

### Key Metrics

Revenue from data center operations increased significantly, with major cloud service providers expanding their GPU deployments. The NVIDIA H100 Tensor Core GPU became the preferred choice for large language model training.

## Gaming Segment

While gaming revenue declined slightly, the segment maintained strong margins. The launch of RTX 40-series GPUs provided premium performance for gaming enthusiasts.

### Market Position

NVIDIA continues to dominate the discrete GPU market with approximately 80% market share in the gaming segment.

## Future Outlook

Management expects continued strong demand for AI computing infrastructure, with data center revenue projected to grow further in fiscal 2025.
"""

        sample_md_path.write_text(sample_content, encoding="utf-8")
        console.print(f"[green]✓[/green] Created sample file at {sample_md_path}\n")


    markdown_content = (DATA_DIR / "NVDA_Q3_FY2026_Earnings_Release.md").read_text(encoding="utf-8")
    console.print(f"Loaded document: [cyan]{len(markdown_content):,}[/cyan] characters\n")

    console.print("[bold cyan]Demo 1: Basic Chunking (Header-Aware)[/bold cyan]\n")

    basic_chunks = chunk_markdown(markdown_content, enrich_with_llm=False)
    console.print(f"Generated {len(basic_chunks)} chunks\n")

    display_chunk_table(basic_chunks)
    console.print("\n" + "-" * 80 + "\n")
    
    console.print("[bold cyan]Demo 2: Detailed Chunk View[/bold cyan]\n")

    display_chunk_details(basic_chunks, limit=3)
    console.print("\n" + "-" * 80 + "\n")
              
    console.print("[bold cyan]Demo 3: LLM-Enriched Chunking[/bold cyan]\n")
    console.print(
    "[yellow]^ This will call the LLM fo each chunk (may take a moment)...[/yellow]\n"
    )

    enriched_chunks = chunk_markdown(markdown_content, enrich_with_llm=True)
    console.print(f"\nGenerated {len(enriched_chunks)} enriched chunks\n")

    display_enriched_chunks(enriched_chunks, limit=3)
                            
    display_cost_analysis(basic_chunks, enriched_chunks)

    output_path = DATA_DIR / "NVDA_Q3_FY2026_Earnings_Release_enriched_chunks.json"
    save_chunks_to_json(enriched_chunks, output_path)      