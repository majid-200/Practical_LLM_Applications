"""
CONFIGURATION MODULE

This module defines the configuration settings for an AI chat application.
It handles model selection, personas, and file path management.
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path



# MODEL PROVIDER ENUM
class ModelProvider(str, Enum):
    """
    Defines which AI service provider to use for running models.
    
    Think of this as choosing between different "brands" of AI:
    ┌─────────────┐       ┌─────────────────┐
    │   OLLAMA    │       │  GOOGLE_GENAI   │
    │  (Local)    │       │    (Cloud)      │
    └─────────────┘       └─────────────────┘
    """
    OLLAMA = "ollama"              # Local AI models running on your machine
    GOOGLE_GENAI = "google_genai"  # Google's cloud-based AI models


# MODEL CONFIGURATION
@dataclass
class ModelConfig:
    """
    Configuration blueprint for an AI model.
    
    Structure visualization:
    ┌─────────────────────────────────────┐
    │        ModelConfig                  │
    ├─────────────────────────────────────┤
    │ name: str         → "qwen3:8b"      │  Model identifier
    │ temperature: float → 0.6            │  Creativity level (0=focused, 1=creative)
    │ provider: enum    → OLLAMA          │  Where the model runs
    │ reasoning: bool   → True            │  Can it explain its thinking?
    └─────────────────────────────────────┘
    
    @dataclass automatically creates __init__, __repr__, and other methods
    """
    name: str                    # Model identifier (e.g., "gpt-4", "qwen3:8b")
    temperature: float           # Controls randomness (0.0 = deterministic, 1.0 = creative)
    provider: ModelProvider      # Which service provides this model
    reasoning: bool             # Whether model supports chain-of-thought reasoning


# PREDEFINED MODEL INSTANCES
# These are "ready-to-use" model configurations you can reference anywhere

QWEN_3 = ModelConfig(
    "qwen3:8b",                          # 8 billion parameter Qwen model
    temperature=0.6,                      # Balanced creativity (moderate randomness)
    provider=ModelProvider.OLLAMA,        # Runs locally via Ollama
    reasoning=True                        # Supports reasoning explanations
)

GEMINI_2_5_FLASH = ModelConfig(
    "gemini-2.5-flash",                  # Google's fast Gemini variant
    temperature=0.0,                      # Deterministic (no randomness)
    provider=ModelProvider.GOOGLE_GENAI,  # Runs on Google's cloud
    reasoning=True                        # Supports reasoning explanations
)


# PERSONA ENUM
class Persona(str, Enum):
    """
    Defines different AI personality types/roles.
    
    Each persona changes how the AI responds:
    
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ NEUROMIND   │  │    CODER    │  │   ROASTER   │  │   TEACHER   │  │  LOGICIAN   │
    │   General   │  │   Coding    │  │   Humor     │  │   Explain   │  │   Logic     │
    └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
    """
    NEUROMIND = "neuromind"  # General-purpose assistant
    CODER = "coder"          # Focused on programming/technical help
    ROASTER = "roaster"      # Humorous/sarcastic responses
    TEACHER = "teacher"      # Educational, patient explanations
    LOGICIAN = "logician"    # Formal logic and reasoning


# MAIN CONFIGURATION CLASS
class Config:
    """
    Central configuration hub for the entire application.
    
    Application Structure:
    ┌─────────────────────────────────────────┐
    │            Config                       │
    ├─────────────────────────────────────────┤
    │ MODEL           → Which AI to use       │
    │ CONTEXT_WINDOW  → Memory size           │
    │ DEFAULT_THREAD  → Conversation name     │
    │ Path            → File locations        │
    └─────────────────────────────────────────┘
    """
    
    # MODEL SETTINGS
    MODEL = QWEN_3  # Default AI model (change to GEMINI_2_5_FLASH to switch)
    # CONTEXT SETTINGS
    CONTEXT_WINDOW = 4096  # Maximum tokens (words/pieces) the AI can remember
                           # Think of this as "short-term memory size"
                           # 4096 ≈ 3000 words of conversation history
    # THREAD SETTINGS
    DEFAULT_THREAD = "master"  # Default conversation thread name
                               # Like a "General" channel in Discord
    

    # FILE PATH CONFIGURATION
    class Path:
        """
        Manages all file and directory paths for the application.
        
        Directory Structure:
        ┌─────────────────────────────────────────┐
        │  APP_HOME (project root)                │
        │  └── data/                              │
        │      ├── neuromind.db    (database)     │
        │      └── personas/       (AI profiles)  │
        └─────────────────────────────────────────┘
        
        Example paths:
        - APP_HOME:       /home/user/neuromind/
        - DATA_DIR:       /home/user/neuromind/data/
        - DATABASE_FILE:  /home/user/neuromind/data/neuromind.db
        - PERSONAS_DIR:   /home/user/neuromind/data/personas/
        """
        
        # Root directory of the application
        # Uses environment variable APP_HOME if set, otherwise uses parent of this file
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        
        # Data directory (stores all persistent data)
        DATA_DIR = APP_HOME / "data"
        
        # SQLite database file (stores conversations, messages, etc.)
        DATABASE_FILE = DATA_DIR / "neuromind.db"
        
        # Directory containing persona definition files
        PERSONAS_DIR = DATA_DIR / "personas"



# USAGE EXAMPLES
"""
How to use this configuration module:

1. Import the config:
   from config import Config, ModelProvider, Persona

2. Access the current model:
   current_model = Config.MODEL
   print(f"Using: {current_model.name}")  # Output: Using: qwen3:8b

3. Get database path:
   db_path = Config.Path.DATABASE_FILE
   print(db_path)  # Output: /path/to/app/data/neuromind.db

4. Switch models:
   Config.MODEL = GEMINI_2_5_FLASH  # Now using Google's model

5. Work with personas:
   active_persona = Persona.TEACHER
   persona_file = Config.Path.PERSONAS_DIR / f"{active_persona.value}.txt"

6. Check model properties:
   if Config.MODEL.reasoning:
       print("This model supports reasoning!")
"""