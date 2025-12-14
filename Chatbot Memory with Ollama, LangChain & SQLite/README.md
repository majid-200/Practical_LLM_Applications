# Project: NeuroMind (Persistent CLI Chatbot)

**NeuroMind** is a robust, full-stack AI application that runs entirely locally. Unlike simple scripts, this project simulates a production-grade architecture by separating the **Backend (API Server)** from the **Frontend (CLI Client)**.

It features persistent memory using **SQLite**, real-time streaming using **Server-Sent Events (SSE)**, and a beautiful terminal interface built with **Rich**.

## Core Concepts

This project moves beyond simple "input/output" scripts to teach system design patterns:

1.  **Client-Server Architecture:**
    *   **Server (`FastAPI`):** Handles the "Brain" (LLM), "Memory" (Database), and "Personality" (Personas).
    *   **Client (`Rich`):** Handles the "Presentation." It knows nothing about the DB or LLM; it just talks to the API.
2.  **Server-Sent Events (SSE):** Instead of waiting for the full answer, the server streams tokens to the client in real-time using standard HTTP streaming.
3.  **State Management:** Uses **SQLite** to store threads and messages, allowing you to close the app and resume conversations later.
4.  **Persona Injection:** Dynamically swaps system prompts (Coder, Teacher, Roaster) based on user selection.

## Architecture

```mermaid
graph TD
    User[User (Terminal)] <-->|Interacts| Client[CLI App (app.py)]
    
    subgraph "Frontend"
        Client <-->|Display| UI[UI Manager (Rich)]
    end
    
    Client <-->|HTTP/SSE| Server[API Server (server.py)]
    
    subgraph "Backend"
        Server <-->|Retrieves| DB[(SQLite Database)]
        Server <-->|Loads| Personas[Persona Files]
        Server <-->|Invokes| LLM[Ollama (Local AI)]
    end
```

## Project Structure

```text
Chatbot_Memory_with_Ollama_LangChain_SQLite/
├── start_server.py           # Entry point to launch the API
├── server.py                 # FastAPI backend logic
├── app.py                    # Entry point to launch the CLI Client
├── requirements.txt          # Dependencies
│
├── data/                     # Data storage
│   ├── neuromind.db          # (Created automatically) SQLite DB
│   └── personas/             # Personality definitions
│       ├── neuromind.md
│       ├── coder.md
│       ├── teacher.md
│       └── ...
│
└── Project/                  # Main application package
    ├── __init__.py           # (Empty file to make it a package)
    ├── api_client.py         # Handles HTTP/SSE communication
    ├── config.py             # Global settings & paths
    ├── thread_manager.py     # Database CRUD operations
    └── ui_manager.py         # Terminal UI logic
```

## Setup & Installation

### 1. Install Dependencies

```bash
pip install fastapi uvicorn requests rich langchain langchain-ollama pydantic python-dotenv
```

### 3. Pull Ollama Model
The default configuration uses **Qwen 3** (specifically `qwen3:8b` in the code, but you can map this to `qwen2.5` or `llama3` in `config.py`).

```bash
ollama pull qwen3:8b
# Note: Ensure the model name in Project/config.py matches the one you pull!
# Default in config.py is "qwen3:8b"
```

## Running the Application

You need two separate terminal windows (or tabs) to run this application: one for the **Server** and one for the **Client**.

### Terminal 1: The Server
This starts the REST API.
```bash
python start_server.py
```
*You will see: `Uvicorn running on http://0.0.0.0:8000`*

### Terminal 2: The Client
This starts the user interface.
```bash
python app.py
```

##  How to Use

Once the CLI is running, you can interact with the bot using standard chat or special commands.

### Commands
*   `/new <name>`: Create a new conversation thread (e.g., `/new python_study`). You will be prompted to select a persona (e.g., Coder, Teacher).
*   `/switch <name>`: Switch to an existing thread.
*   `/list`: Show all active threads and message counts in a table.
*   `/clear`: Wipe the memory of the current thread.
*   `/exit`: Close the client (Server keeps running).

### Example Workflow
1.  **Start:** `python app.py`
2.  **Create:** `/new coding_help` -> Select "2" (Coder).
3.  **Chat:** "How do I use Asyncio in Python?"
4.  **Observe:** The "Reasoning" (if supported by model) appears in a dim panel, followed by the code.
5.  **Exit:** `/exit`
6.  **Resume:** Run `python app.py` again -> `/switch coding_help`. Context is remembered!

## Technical Deep Dive

### Server-Sent Events (SSE)
Unlike standard web requests where you wait for the whole page to load, this app uses streaming. 
*   **Server:** Yields chunks of data (`{"type": "content", "content": "..."}`) using Python `generators`.
*   **Client:** Iterates over these chunks as they arrive, updating the UI instantly.

### Dependency Injection
The server uses **FastAPI's** `Depends()` system. This ensures that every API request gets its own fresh Database Connection (`get_db`) and Model Instance (`get_llm`), making the application thread-safe and easier to test.

### Pydantic Validation
Files like `server.py` use **Pydantic** models (`ThreadCreate`, `MessageResponse`) to strictly validate data. If the client sends bad data, the server rejects it automatically with clear errors before it ever reaches your logic.

---

## Acknowledgements
This project is heavily inspired by and based on the excellent tutorial by Venelin Valkov:
- **[Build Private AI Assistant That Actually Remembers | Chatbot Memory with Ollama, LangChain & SQLite](https://youtu.be/XRfCfDQb1zc?si=5FJrTZPwk5mx_rXf)**
