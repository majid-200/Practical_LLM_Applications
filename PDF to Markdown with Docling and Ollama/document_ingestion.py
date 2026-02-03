from pathlib import Path  # Makes working with file paths easier (cross-platform)
from typing import Any    # Helps with type hints (makes code clearer)

# Docling is the main library we're using - it converts PDFs to markdown
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,        # Main configuration for PDF processing
    TableStructureOptions,     # How to handle tables in PDFs
    PictureDescriptionApiOptions,  # How to describe images using AI
    TableFormerMode,          # Which model to use for table extraction
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend  # PDF reader engine
from docling_core.types.doc import ImageRefMode  # How to handle images in output

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# OLLAMA CONFIGURATION (Local AI Server)                      

OLLAMA_URL = "http://localhost:11434"
# This is the web address where your local AI (Ollama) is running
# localhost = your own computer
# 11434 = the port number

# VISION LANGUAGE MODEL (VLM) SETTINGS                        
VLM_MODEL = "qwen3-vl:2b"
# The AI model that will "look at" images and describe them
# "qwen3-vl:2b" is a specific vision model (2 billion parameters - lightweight)

VLM_PROMPT = "Act as a senior financial analyst. Explain what you see in the image in 1 sentence. Focus on trends, tickers (only if you're sure), and key insights."
# This tells the AI HOW to describe images
# Like giving instructions to an assistant: "describe this chart like an expert would"

# MARKDOWN OUTPUT FORMATTING                                  

PAGE_BREAK_PLACEHOLDER = "<!-- page_break -->"
# When converting PDF → Markdown, use this text to show where pages end
# Example: Page 1 content <!-- page_break --> Page 2 content

IMAGE_DESCRIPTION_START = "<image_description>"
IMAGE_DESCRIPTION_END = "</image_description>"
# These wrap around AI-generated image descriptions
# Example: <image_description>A bar chart showing Q3 revenue growth</image_description>



# ============================================================================
# FUNCTION 1: Configure Picture (Image) Description
# ============================================================================
# Purpose: Sets up how the AI will describe images found in the PDF
# Returns: A configuration object with all the AI settings
# ============================================================================

def create_picture_description_options() -> PictureDescriptionApiOptions:
    # The "-> PictureDescriptionApiOptions" means this function returns 
    # a PictureDescriptionApiOptions object (type hint for clarity)
    
    return PictureDescriptionApiOptions(
        
        # API ENDPOINT                                            
        url=f"{OLLAMA_URL}/v1/chat/completions",
        # Full URL: http://localhost:11434/v1/chat/completions
        # This is the specific "endpoint" (address) where we send images
        
        # AI MODEL PARAMETERS (How the AI should behave)          
        params=dict(
            
            model=VLM_MODEL,  # Which AI model to use (qwen3-vl:2b)
            
            think=False,  # Don't show the AI's "thinking process" 
                         # (just give the final answer)
            
            seed=42,  # Random seed for reproducibility
                     # Using the same seed = same results for same input
                     # 42 is a common choice (Hitchhiker's Guide reference!)
            
            max_completion_tokens=256,  # Maximum length of AI response
                                       # 256 tokens ≈ 1-2 sentences
                                       # Keeps descriptions concise
        ),
        
        # THE INSTRUCTION PROMPT                                  
        prompt=VLM_PROMPT,  # The instruction we defined earlier
                           # Tells AI to act like a financial analyst
        
        # TIMEOUT (How long to wait for AI response)              
        timeout=180,  # Wait up to 180 seconds for the AI to respond
                    # If it takes longer, give up (prevents hanging)
    )


# ============================================================================
# FUNCTION 2: Configure PDF Processing Pipeline
# ============================================================================
# Purpose: Sets up ALL the options for how to convert the PDF
# This is like setting up an assembly line with different stations
# Returns: A complete pipeline configuration object
# ============================================================================

def create_pdf_pipeline_options() -> PdfPipelineOptions:
    # The "-> PdfPipelineOptions" means this function returns 
    # a PdfPipelineOptions object
    
    return PdfPipelineOptions(

        # REMOTE SERVICES (Can we use external/cloud services?)   
        enable_remote_services=True,
        # True = Allow using external APIs (like our Ollama server)
        # False = Only use local processing (no network calls)
        
        
        # OCR - Optical Character Recognition                     
        do_ocr=False,
        # OCR = Converting images of text into actual text
        # False = Don't do OCR (assume PDF already has text layer)
        # True = Would scan images for text (slower but works on scanned PDFs)
        # 
        # Example: 
        #   Scanned document image "Hello" → OCR → Text: "Hello"
        
        # TABLE STRUCTURE EXTRACTION                              
        do_table_structure=True,
        # True = Detect and extract tables from PDF
        # The AI will understand table rows/columns and convert to markdown
        #
        # Example PDF table:     → Markdown table:
        # ┌─────┬─────┐           | Col1 | Col2 |
        # │ A   │ B   │           |------|------|
        # │ C   │ D   │           | A    | B    |
        # └─────┴─────┘           | C    | D    |
        
        # IMAGE GENERATION
        generate_picture_images=True,
        # True = Extract images from the PDF
        # False = Ignore images completely
        # When True, images can be saved or described
        
        # IMAGE DESCRIPTION (Using AI)                            
        do_picture_description=True,
        # True = Use AI to describe what's in the images
        # False = Just extract images but don't describe them
        # This uses the Ollama AI we configured earlier!
        
        # │ TABLE EXTRACTION OPTIONS (Advanced settings)            
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE,  # Uses the high-precision model
            # ACCURATE = Slower but more precise table detection
            # FAST = Faster but might miss complex tables
            # Think: Quality vs Speed trade-off
        ),
        
        # PICTURE DESCRIPTION OPTIONS (From our first function!)  
        picture_description_options=create_picture_description_options(),
        # This calls our first function to get all the AI image settings
        # It's like saying: "Use those image description settings I made earlier"
    )


# ============================================================================
# FUNCTION 3: Process Document (THE MAIN WORKER)
# ============================================================================
# Purpose: Takes a PDF file and converts it to Markdown with AI descriptions
# Input: pdf_path (string) - the file path to your PDF
# Output: content (string) - the final markdown text
# ============================================================================

def process_document(pdf_path: str):
    # pdf_path: str means the parameter must be a string (file path)
    
    # STEP 1: Create the Document Converter
    
    converter = DocumentConverter(
        # format_options = A dictionary that tells the converter:
        # "For each file type, here's how to process it"
        format_options={
            # For PDF files specifically:
            InputFormat.PDF: PdfFormatOption(
                # Use the pipeline options we created earlier
                pipeline_options=create_pdf_pipeline_options(),
                
                # Which PDF "engine" to use for reading the file
                backend=PyPdfiumDocumentBackend,
                # PyPdfium = A fast, reliable PDF reading library
            )
        }
    )
    
    # STEP 2: Actually Convert the PDF
    
    result = converter.convert(pdf_path)
    # This is where the actual conversion happens!
    # The converter:
    #   1. Opens the PDF
    #   2. Extracts text, tables, images
    #   3. Sends images to Ollama AI for descriptions
    #   4. Returns a result object with everything
    
    doc = result.document
    # Extract just the document object from the result
    # doc now contains all the converted content
    
    # STEP 3: Export Document to Markdown Format
    
    content = doc.export_to_markdown(
        # How to handle images in the markdown:
        image_mode=ImageRefMode.PLACEHOLDER,
        # PLACEHOLDER = Don't embed the actual image, use a placeholder
        # Other options: EMBEDDED (base64), REFERENCED (file path)
        
        image_placeholder="",
        # What text to use as the placeholder
        # "" = empty string (so images become invisible in markdown)
        # Could be "[IMAGE]" or "![](image.png)" etc.
        
        # How to mark page breaks:
        page_break_placeholder=PAGE_BREAK_PLACEHOLDER,
        # Use our constant: "<!-- page_break -->"
        # Shows where one PDF page ends and another begins
        
        # Include AI-generated annotations?
        include_annotations=True,
        # True = Include the AI image descriptions
        # False = Skip them
        
        mark_annotations=True,
        # True = Wrap annotations in special HTML comment tags
        # False = Just include them as plain text
        # When True, they look like:
        # <!--<annotation kind="description">-->
        # AI description here
        # <!--</annotation>-->
    )
    
    # STEP 4: Clean Up the Annotation Markers                    
    
    content = content.replace(
        '<!--<annotation kind="description">-->',  # Find this tag
        IMAGE_DESCRIPTION_START  # Replace with: <image_description>
    )
    # BEFORE: <!--<annotation kind="description">-->Chart shows growth<!--</annotation>-->
    # AFTER:  <image_description>Chart shows growth<!--</annotation>-->
    
    content = content.replace(
        '<!--</annotation>-->',  # Find this closing tag
        IMAGE_DESCRIPTION_END    # Replace with: </image_description>
    )
    # FINAL:  <image_description>Chart shows growth</image_description>
    
    return content

if __name__ == "__main__":
    # file_name = Path("sp500-analysis.pdf")
    # file_name = Path("nvidia-q3-2025-press-release.pdf")
    file_name = Path("sp-500-brochure.pdf")
    md_content = process_document(f"./data/{file_name}")
    Path(f"output/{file_name.stem}.md").write_text(md_content, encoding='utf-8')