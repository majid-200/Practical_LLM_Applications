"""
FASTAPI SERVER MODULE

This module implements the REST API server using FastAPI.
It's the "backend" that handles AI chat, thread management, and database operations.

Server Architecture:
┌──────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Server                                  │
├──────────────────────────────────────────────────────────────────────────┤
│  REST API Endpoints:                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │   /personas     │  │    /threads     │  │     /health     │           │
│  │   (List AIs)    │  │  (Manage chats) │  │  (Status check) │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│                                                                          │
│  Dependencies:                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │ ThreadManager   │  │   LLM (AI)      │  │    Personas     │           │
│  │  (Database)     │  │  (LangChain)    │  │  (System msgs)  │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
└──────────────────────────────────────────────────────────────────────────┘

Request Flow Example:
┌─────────────┐    POST /threads/master/chat      ┌──────────────┐
│   Client    │ ─────────────────────────────────>│    Server    │
│             │                                   │              │
│             │ ←─────────────────────────────────│ SSE Stream:  │
│ (displays)  │   data: {"type": "content"...}    │ - Load msgs  │
└─────────────┘                                   │ - Call AI    │
                                                  │ - Stream back│
                                                  │ - Save to DB │
                                                  └──────────────┘
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv  # Load environment variables from .env file
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain.chat_models import init_chat_model  # Initialize AI models
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field  # Data validation

from Project.config import Config, Persona
from Project.thread_manager import Thread, ThreadManager


# LOGGING & ENVIRONMENT

# Logger for tracking errors and events
# Usage: logger.error("Something went wrong!")
logger = logging.getLogger(__name__)

# Load environment variables from .env file (API keys, etc.)
# Example .env file:
#   GOOGLE_API_KEY=your_key_here
#   OLLAMA_HOST=http://localhost:11434
load_dotenv()



# PYDANTIC MODELS (DATA VALIDATION)

"""
Pydantic models define the "shape" of data coming in and going out.
Think of them as contracts: "This is what I expect to receive/send"

Benefits:
✓ Automatic validation (wrong data types = automatic error)
✓ Auto-generated API documentation
✓ Type safety
✓ Clear data structures

Example:
    Without Pydantic:
        data = request.json()  # Could be anything!
        name = data.get("name")  # Might not exist
    
    With Pydantic:
        data: ThreadCreate  # Guaranteed to have valid 'name' and 'persona'
        name = data.name    # Type-safe, always exists
"""


class ThreadCreate(BaseModel):
    """
    Data structure for creating a new thread.
    
    ┌──────────────────────────────────────┐
    │        ThreadCreate                  │
    ├──────────────────────────────────────┤
    │ name: str (1-100 chars, required)    │
    │ persona: Persona (default: NEUROMIND)│
    └──────────────────────────────────────┘
    
    Example JSON:
    {
        "name": "coding_help",
        "persona": "coder"
    }
    
    Field() Validation:
    • ... means "required" (no default value)
    • min_length=1: Name must be at least 1 character
    • max_length=100: Name can't exceed 100 characters
    """
    name: str = Field(..., min_length=1, max_length=100)
    persona: Persona = Persona.NEUROMIND  # Default persona if not specified


class ThreadResponse(BaseModel):
    """
    Data structure for returning thread information.
    
    Example JSON Response:
    {
        "id": 1,
        "name": "master",
        "persona": "neuromind"
    }
    """
    id: int          # Database ID
    name: str        # Thread name
    persona: str     # Persona name


class ThreadListItem(BaseModel):
    """
    Data structure for listing threads with message counts.
    
    Example JSON Response:
    {
        "name": "master",
        "persona": "neuromind",
        "message_count": 42
    }
    """
    name: str
    persona: str
    message_count: int


class MessageCreate(BaseModel):
    """
    Data structure for sending a message.
    
    Example JSON:
    {
        "content": "What is async programming?"
    }
    
    Validation: Content must be at least 1 character (no empty messages)
    """
    content: str = Field(..., min_length=1)


class MessageResponse(BaseModel):
    """
    Data structure for returning a message.
    
    Example JSON Response:
    {
        "role": "human",
        "content": "Hello!"
    }
    """
    role: str        # "human" or "ai"
    content: str     # Message text


class ChatResponse(BaseModel):
    """
    Data structure for complete chat response (not used in streaming).
    
    This would be for non-streaming endpoints (if we had them).
    """
    response: str
    reasoning: str | None = None  # Optional reasoning


class ErrorDetail(BaseModel):
    """
    Data structure for error responses.
    
    Example JSON Response:
    {
        "error": "thread_not_found",
        "detail": "Thread 'xyz' does not exist"
    }
    """
    error: str              # Error code
    detail: str | None = None  # Optional detailed message


class PersonaResponse(BaseModel):
    """
    Data structure for persona information.
    
    Example JSON Response:
    {
        "name": "coder",
        "description": "Coder persona"
    }
    """
    name: str
    description: str



# DEPENDENCY INJECTION FUNCTIONS

"""
Dependency Injection (DI) is a fancy term for "provide what's needed automatically"

Instead of:
    def my_function():
        db = ThreadManager("path/to/db")  # Create every time
        # use db

We do:
    def my_function(db: ThreadManager = Depends(get_db)):  # Provided automatically
        # use db

Benefits:
✓ Don't repeat database/LLM initialization
✓ Easy to test (can inject mock objects)
✓ Clean code (FastAPI handles the wiring)
✓ Resource management (connections cleaned up automatically)
"""


def get_db() -> ThreadManager:
    """
    Dependency function that provides a ThreadManager instance.
    
    FastAPI will call this function whenever an endpoint needs a database.
    
    Flow:
    ┌─────────────────────────────────────────┐
    │ Endpoint called                         │
    │ def list_threads(db = Depends(get_db))  │
    └────────────────┬────────────────────────┘
                     │
                     ↓ FastAPI calls get_db()
    ┌────────────────────────────────────────┐
    │ get_db() returns ThreadManager         │
    └────────────────┬───────────────────────┘
                     │
                     ↓ Inject into endpoint
    ┌────────────────────────────────────────┐
    │ Endpoint uses db parameter             │
    └────────────────────────────────────────┘
    
    Why not create globally?
    • Each request gets its own DB connection
    • Prevents connection conflicts in async code
    • Easier to test
    """
    return ThreadManager(Config.Path.DATABASE_FILE)


def get_llm():
    """
    Dependency function that provides an AI language model.
    
    This initializes the LangChain chat model based on config settings.
    
    Parameters Explained:
    • model.name: Which model to use (e.g., "qwen3:8b")
    • model_provider: Where to get it (ollama, google_genai)
    • reasoning: Enable chain-of-thought reasoning
    • num_ctx: Context window size (how much history to remember)
    
    Example:
    ┌──────────────────────────────────────┐
    │ Config says: QWEN_3                  │
    │ - name: "qwen3:8b"                   │
    │ - provider: OLLAMA                   │
    │ - reasoning: True                    │
    └────────────┬─────────────────────────┘
                 │
                 ↓ get_llm() initializes
    ┌──────────────────────────────────────┐
    │ Returns: LangChain ChatModel         │
    │ Connected to local Ollama server     │
    └──────────────────────────────────────┘
    """
    model = Config.MODEL
    return init_chat_model(
        model.name,                      # Model identifier
        model_provider=model.provider.value,  # Provider (ollama/google_genai)
        reasoning=True,                  # Enable reasoning mode
        num_ctx=Config.CONTEXT_WINDOW,   # Context window size
    )


def get_personas() -> dict[str, str]:
    """
    Load all persona system prompts from markdown files.
    
    Returns:
        Dictionary mapping persona names to their system prompts
    
    Example:
    {
        "neuromind": "You are a helpful AI assistant...",
        "coder": "You are an expert programmer...",
        "teacher": "You are a patient educator..."
    }
    
    File Structure:
    ┌─────────────────────────────────────────┐
    │ data/personas/                          │
    │ ├── neuromind.md  ─→ "You are..."       │
    │ ├── coder.md      ─→ "You are..."       │
    │ ├── teacher.md    ─→ "You are..."       │
    │ └── logician.md   ─→ "You are..."       │
    └─────────────────────────────────────────┘
    
    Dictionary Comprehension Breakdown:
    {
        p.value: (Config.Path.PERSONAS_DIR / f"{p.value}.md").read_text()
        for p in Persona
    }
    
    Translates to:
    for p in Persona:
        filename = f"{p.value}.md"  # "neuromind.md"
        filepath = Config.Path.PERSONAS_DIR / filename
        content = filepath.read_text()
        result[p.value] = content
    """
    return {
        p.value: (Config.Path.PERSONAS_DIR / f"{p.value}.md").read_text()
        for p in Persona
    }



# APPLICATION LIFESPAN MANAGEMENT

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown logic.
    
    This is called once when the server starts and once when it stops.
    Think of it like __init__ and __del__ for the entire application.
    
    Flow:
    ┌──────────────────────────────────────┐
    │ Server Starting...                   │
    └────────────┬─────────────────────────┘
                 │
                 ↓ Load personas (before yield)
    ┌──────────────────────────────────────┐
    │ app.state.personas = {...}           │
    │ Personas loaded into memory          │
    └────────────┬─────────────────────────┘
                 │
                 ↓ yield (server runs)
    ┌──────────────────────────────────────┐
    │ Server Running...                    │
    │ (handle requests)                    │
    └────────────┬─────────────────────────┘
                 │
                 ↓ Server Stopping (after yield)
    ┌──────────────────────────────────────┐
    │ Cleanup code would go here           │
    │ (close connections, save state, etc.)│
    └──────────────────────────────────────┘
    
    app.state is a place to store app-wide data:
    • Loaded once at startup
    • Accessible in all endpoints
    • Avoids reading files on every request
    
    @asynccontextmanager is like a context manager (with statement) but async:
    async with lifespan():
        # do startup
        yield
        # do cleanup
    """
    # STARTUP: Load all persona prompts into memory
    app.state.personas = get_personas()
    
    yield  # Server runs here (yield = pause and let server run)
    
    # SHUTDOWN: Cleanup code would go here (we don't need any currently)



# FASTAPI APPLICATION

app = FastAPI(
    title="NeuroMind API",                    # Shows in API docs
    description="AI Assistant REST API",      # Shows in API docs
    version="0.1.0",                          # API version
    lifespan=lifespan,                        # Startup/shutdown handler
)

# FastAPI automatically creates interactive documentation at:
# • http://localhost:8000/docs  (Swagger UI)
# • http://localhost:8000/redoc (ReDoc)


# API ENDPOINTS

"""
Endpoints are the "functions" of your API - the operations clients can call.

Endpoint Anatomy:
@app.get("/path")                          ← Decorator: HTTP method + URL path
def function_name(                         ← Function name (for documentation)
    param: type,                           ← Path/query parameters
    db: Type = Depends(get_db)             ← Injected dependencies
) -> ResponseType:                         ← Return type (for validation)
    '''Docstring appears in API docs'''
    return data                            ← Return value (auto-converted to JSON)
"""



# LIST PERSONAS ENDPOINT

@app.get("/personas", response_model=list[PersonaResponse])
def list_personas():
    """
    List all available personas.
    
    HTTP Request:
    GET /personas
    
    Response: 200 OK
    [
        {"name": "neuromind", "description": "Neuromind persona"},
        {"name": "coder", "description": "Coder persona"},
        ...
    ]
    
    @app.get() decorator:
    • Registers this function as a GET endpoint
    • URL path: /personas
    • response_model: Validates and documents return type
    
    List comprehension creates PersonaResponse for each Persona enum:
    [PersonaResponse(...) for p in Persona]
    """
    return [
        PersonaResponse(
            name=p.value,  # "neuromind", "coder", etc.
            description=f"{p.value.title()} persona"  # "Neuromind persona"
        )
        for p in Persona
    ]



# LIST THREADS ENDPOINT

@app.get("/threads", response_model=list[ThreadListItem])
def list_threads(db: ThreadManager = Depends(get_db)):
    """
    List all conversation threads.
    
    HTTP Request:
    GET /threads
    
    Response: 200 OK
    [
        {"name": "master", "persona": "neuromind", "message_count": 42},
        {"name": "coding", "persona": "coder", "message_count": 15}
    ]
    
    Flow:
    ┌─────────────────────────────────────────┐
    │ Client: GET /threads                    │
    └────────────┬────────────────────────────┘
                 │
                 ↓ FastAPI injects db
    ┌─────────────────────────────────────────┐
    │ db.list_threads() → tuples              │
    │ [("master", "neuromind", 42), ...]      │
    └────────────┬────────────────────────────┘
                 │
                 ↓ Convert to Pydantic models
    ┌─────────────────────────────────────────┐
    │ [ThreadListItem(...), ...]              │
    └────────────┬────────────────────────────┘
                 │
                 ↓ FastAPI converts to JSON
    ┌─────────────────────────────────────────┐
    │ Client receives JSON array              │
    └─────────────────────────────────────────┘
    
    Depends(get_db):
    • FastAPI calls get_db() to get ThreadManager
    • Injects it as 'db' parameter
    • Happens automatically for every request
    """
    threads = db.list_threads()  # Returns list of tuples
    return [
        ThreadListItem(name=name, persona=persona, message_count=count)
        for name, persona, count in threads
    ]



# CREATE THREAD ENDPOINT

@app.post("/threads", response_model=ThreadResponse, status_code=201)
def create_thread(data: ThreadCreate, db: ThreadManager = Depends(get_db)):
    """
    Create a new conversation thread.
    
    HTTP Request:
    POST /threads
    Content-Type: application/json
    {
        "name": "new_thread",
        "persona": "coder"
    }
    
    Response: 201 Created
    {
        "id": 5,
        "name": "new_thread",
        "persona": "coder"
    }
    
    status_code=201:
    • 201 = "Created" (standard for POST that creates a resource)
    • Different from 200 (OK) - signals resource creation
    
    Request Body (data: ThreadCreate):
    • FastAPI automatically parses JSON
    • Validates against ThreadCreate model
    • If validation fails, returns 422 error automatically
    
    Example Invalid Request:
    {
        "name": "",  ← Too short (min_length=1)
        "persona": "invalid"  ← Not a valid Persona
    }
    Response: 422 Unprocessable Entity
    {
        "detail": [validation errors]
    }
    """
    thread = db.get_or_create_thread(data.name, data.persona)
    return ThreadResponse(id=thread.id, name=thread.name, persona=thread.persona)



# GET THREAD ENDPOINT

@app.get("/threads/{thread_name}", response_model=ThreadResponse)
def get_thread_endpoint(thread_name: str, db: ThreadManager = Depends(get_db)):
    """
    Get a thread by name.
    
    HTTP Request:
    GET /threads/master
    
    Response: 200 OK
    {
        "id": 1,
        "name": "master",
        "persona": "neuromind"
    }
    
    Or if not found:
    Response: 404 Not Found
    {
        "detail": "Thread not found"
    }
    
    Path Parameter:
    • {thread_name} in URL path becomes function parameter
    • GET /threads/master → thread_name = "master"
    • GET /threads/coding → thread_name = "coding"
    
    HTTPException:
    • FastAPI's way of returning error responses
    • status_code=404 → HTTP 404 Not Found
    • detail → Error message in response
    
    URL Examples:
    GET /threads/master    ✓ Valid
    GET /threads/coding    ✓ Valid
    GET /threads/my%20chat ✓ Valid (space encoded as %20)
    """
    thread = db.get_thread(thread_name)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadResponse(id=thread.id, name=thread.name, persona=thread.persona)



# GET MESSAGES ENDPOINT

@app.get("/threads/{thread_name}/messages", response_model=list[MessageResponse])
def get_messages(thread_name: str, db: ThreadManager = Depends(get_db)):
    """
    Get message history for a thread.
    
    HTTP Request:
    GET /threads/master/messages
    
    Response: 200 OK
    [
        {"role": "human", "content": "Hello"},
        {"role": "ai", "content": "Hi there!"},
        {"role": "human", "content": "How are you?"},
        {"role": "ai", "content": "I'm doing well!"}
    ]
    
    Flow:
    ┌─────────────────────────────────────────┐
    │ GET /threads/master/messages            │
    └────────────┬────────────────────────────┘
                 │
                 ↓ Get thread from DB
    ┌─────────────────────────────────────────┐
    │ thread = db.get_thread("master")        │
    └────────────┬────────────────────────────┘
                 │
                 ↓ Get message history
    ┌─────────────────────────────────────────┐
    │ history = db.get_history(thread.id)     │
    │ Returns: [HumanMessage, AIMessage, ...] │
    └────────────┬────────────────────────────┘
                 │
                 ↓ Convert to API response format
    ┌─────────────────────────────────────────┐
    │ [MessageResponse(role, content), ...]   │
    └─────────────────────────────────────────┘
    
    Why check if thread exists?
    • Prevent crashes if someone requests non-existent thread
    • Return clear 404 error instead of cryptic database error
    """
    thread = db.get_thread(thread_name)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    history = db.get_history(thread.id)
    return [MessageResponse(role=msg.type, content=msg.content) for msg in history]


# CLEAR MESSAGES ENDPOINT

@app.delete("/threads/{thread_name}/messages", status_code=204)
def clear_messages(thread_name: str, db: ThreadManager = Depends(get_db)):
    """
    Clear all messages in a thread.
    
    HTTP Request:
    DELETE /threads/master/messages
    
    Response: 204 No Content
    (empty response body)
    
    status_code=204:
    • 204 = "No Content" (success, but no data to return)
    • Standard for DELETE operations
    • Different from 200 which includes response body
    
    @app.delete():
    • DELETE method for removing resources
    • RESTful convention for deletions
    
    Why DELETE method?
    • Clearly signals destructive operation
    • Browsers/tools can warn users
    • Follows REST best practices
    
    The function doesn't explicitly return anything:
    • Python functions with no return statement return None
    • FastAPI sees status_code=204 and returns empty response
    """
    thread = db.get_thread(thread_name)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    db.clear_messages(thread.id)
    # No return statement → empty 204 response


# CHAT STREAMING ENDPOINT

def _build_context(thread: Thread, user_input: str, personas: dict, db: ThreadManager):
    """
    Build the conversation context for the AI model.
    
    Args:
        thread: Current thread
        user_input: User's new message
        personas: Dictionary of persona system prompts
        db: Database manager
    
    Returns:
        List of messages to send to AI model
    
    Message Structure:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. SystemMessage (persona prompt)                               │
    │    "You are a helpful coding assistant..."                      │
    ├─────────────────────────────────────────────────────────────────┤
    │ 2. Previous conversation history                                │
    │    HumanMessage: "What is async?"                               │
    │    AIMessage: "Async is..."                                     │
    │    HumanMessage: "Can you explain more?"                        │
    │    AIMessage: "Sure..."                                         │
    ├─────────────────────────────────────────────────────────────────┤
    │ 3. New user message                                             │
    │    HumanMessage: "Give me an example"                           │
    └─────────────────────────────────────────────────────────────────┘
    
    Why this order?
    1. System message sets AI personality/behavior
    2. History provides context for conversation
    3. New message is what AI responds to
    
    personas.get() with fallback:
    • Gets persona prompt for thread's persona
    • If not found, defaults to NEUROMIND persona
    • Prevents crashes if persona file is missing
    """
    # Get system prompt for this persona (with fallback)
    sys_prompt = personas.get(thread.persona, personas[Persona.NEUROMIND.value])
    
    # Start with system message
    messages = [SystemMessage(content=sys_prompt)]
    
    # Add conversation history
    messages.extend(db.get_history(thread.id))
    
    # Add new user message
    messages.append(HumanMessage(content=user_input))
    
    return messages


@app.post("/threads/{thread_name}/chat")
async def chat(
    thread_name: str,
    data: MessageCreate,
    db: ThreadManager = Depends(get_db),
    llm=Depends(get_llm),
):
    """
    Send a message and stream the AI response via Server-Sent Events.
    
    HTTP Request:
    POST /threads/master/chat
    Content-Type: application/json
    {
        "content": "What is async programming?"
    }
    
    Response: Streaming (Server-Sent Events)
    Content-Type: text/event-stream
    
    data: {"type": "reasoning", "content": "Let me explain..."}
    
    data: {"type": "content", "content": "Async"}
    
    data: {"type": "content", "content": " programming"}
    
    data: {"type": "content", "content": " allows"}
    
    data: {"type": "done"}
    
    Why async def?
    • This endpoint uses async streaming (llm.astream)
    • async/await allows handling multiple requests concurrently
    • Non-blocking: server can handle other requests while streaming
    
    StreamingResponse:
    • Special FastAPI response for Server-Sent Events
    • Keeps connection open and sends data incrementally
    • media_type="text/event-stream" tells client it's SSE
    
    Flow Diagram:
    ┌──────────────────────────────────────────────────────────────┐
    │ 1. Get or create thread                                      │
    └────────────┬─────────────────────────────────────────────────┘
                 │
                 ↓
    ┌──────────────────────────────────────────────────────────────┐
    │ 2. Build context (system prompt + history + new message)     │
    └────────────┬─────────────────────────────────────────────────┘
                 │
                 ↓
    ┌──────────────────────────────────────────────────────────────┐
    │ 3. Stream AI response chunk by chunk                         │
    │    • Send reasoning (if available)                           │
    │    • Send content as it's generated                          │
    │    • Accumulate full response                                │
    └────────────┬─────────────────────────────────────────────────┘
                 │
                 ↓
    ┌──────────────────────────────────────────────────────────────┐
    │ 4. Save messages to database                                 │
    │    • Save user's message                                     │
    │    • Save AI's complete response                             │
    └────────────┬─────────────────────────────────────────────────┘
                 │
                 ↓
    ┌──────────────────────────────────────────────────────────────┐
    │ 5. Send 'done' event and close stream                        │
    └──────────────────────────────────────────────────────────────┘
    
    Error Handling:
    • ConnectionError: LLM service unavailable
    • TimeoutError: LLM request took too long
    • Exception: Any other unexpected error
    All errors sent as SSE events with error type and message
    """
    thread = db.get_or_create_thread(thread_name)
    context = _build_context(thread, data.content, app.state.personas, db)
    # This async function yields data chunks over time.
    async def generate() -> AsyncGenerator[str, None]:
        full_content = ""

        try:
            # Call LangChain's async stream method
            async for chunk in llm.astream(context):
                
                # Check if chunk is empty (can happen during handshakes)
                if not chunk.content and not chunk.additional_kwargs.get(
                    "reasoning_content"
                ):
                    continue

                # Accumulate full answer for database storage later
                if chunk.content:
                    full_content += chunk.content

                # ─── STREAMING PROTOCOL (SSE) ───
                # Format: "data: {json_payload}\n\n"
                
                # CASE 1: Reasoning (Thinking tokens)
                # Specific to reasoning models
                if chunk.additional_kwargs.get("reasoning_content"):
                    payload = {
                        "type": "reasoning", 
                        "content": chunk.additional_kwargs['reasoning_content']
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                
                # CASE 2: Standard Content
                elif chunk.content:
                    payload = {
                        "type": "content", 
                        "content": chunk.content
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

            # ─── POST-PROCESSING ───
            # Stream is done. Save interaction to database.
            db.add_message(thread.id, "human", data.content)
            db.add_message(thread.id, "ai", full_content)

            # Signal completion to client
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        # ─── ERROR HANDLING INSIDE STREAM ───
        # Since we already returned a 200 OK to start the stream,
        # we cannot raise HTTPExceptions. We must yield error events.
        
        except ConnectionError as e:
            logger.error(f"LLM connection failed during stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': 'connection_failed', 'message': f'AI model unavailable. Please ensure {Config.MODEL.name} is running.'})}\n\n"
        
        except TimeoutError as e:
            logger.error(f"LLM request timed out during stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': 'timeout', 'message': 'AI model request timed out. Please try again.'})}\n\n"
        
        except Exception as e:
            logger.exception(f"Unexpected error during stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': 'internal_error', 'message': str(e)})}\n\n"

    # Return the stream wrapper
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
def health_check():
    """Simple probe to verify server is up and model is configured."""
    return {"status": "ok", "model": Config.MODEL.name}
