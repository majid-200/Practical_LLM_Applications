"""
API CLIENT MODULE

This module provides a Python client for interacting with the NeuroMind REST API.

Architecture Overview:
┌─────────────────┐         HTTP/REST API          ┌──────────────────┐
│  Your App/CLI   │ <────────────────────────────> │  NeuroMind API   │
│                 │      (NeuroMindClient)         │   (FastAPI)      │
│  - Uses client  │                                │  - Processes AI  │
│  - Makes calls  │                                │  - Manages DB    │
└─────────────────┘                                └──────────────────┘

This client handles:
✓ Making HTTP requests to the API
✓ Parsing responses
✓ Streaming Server-Sent Events (SSE)
✓ Error handling
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Generator, List, Tuple

import requests  # HTTP library for making API calls

from Project.config import Persona


# STREAM EVENT TYPES

class StreamEventType(str, Enum):
    """
    Types of events that can be streamed from the chat endpoint.
    
    Event Flow (typical chat):
    ┌──────────────┐
    │ User sends   │ "Explain async"
    │ message      │
    └──────┬───────┘
           │
           ↓
    ┌──────────────┐  Event Type: REASONING
    │ AI thinks    │  "Let me break this down..."
    └──────┬───────┘
           │
           ↓
    ┌──────────────┐  Event Type: CONTENT
    │ AI responds  │  "Async is..."
    │ (streaming)  │  "It allows..."
    └──────┬───────┘  "Programs to..."
           │
           ↓
    ┌──────────────┐  Event Type: DONE
    │ Stream ends  │  (no more data)
    └──────────────┘
    
    If something goes wrong:
    ┌──────────────┐  Event Type: ERROR
    │ Error occurs │  "Connection failed"
    └──────────────┘
    """
    REASONING = "reasoning"  # AI's internal thought process (optional)
    CONTENT = "content"      # The actual response text
    DONE = "done"           # Stream finished successfully
    ERROR = "error"         # Something went wrong


# DATA CLASSES

@dataclass
class ThreadInfo:
    """
    Simplified thread information returned from API.
    
    ┌─────────────────────────────────────┐
    │         ThreadInfo                  │
    ├─────────────────────────────────────┤
    │ id: int       → 1                   │
    │ name: str     → "coding_help"       │
    │ persona: str  → "coder"             │
    └─────────────────────────────────────┘
    
    This is similar to the Thread class in thread_manager.py,
    but used specifically for API responses.
    """
    id: int          # Thread's database ID
    name: str        # Thread name
    persona: str     # AI personality being used


@dataclass
class StreamEvent:
    """
    Represents a single event in the streaming response.
    
    Event Examples:
    
    1. Reasoning Event:
       ┌────────────────────────────────────────┐
       │ type: REASONING                        │
       │ content: "I'll explain step by step"   │
       └────────────────────────────────────────┘
    
    2. Content Event:
       ┌────────────────────────────────────────┐
       │ type: CONTENT                          │
       │ content: "Async allows concurrent..."  │
       └────────────────────────────────────────┘
    
    3. Error Event:
       ┌────────────────────────────────────────┐
       │ type: ERROR                            │
       │ error: "timeout"                       │
       │ message: "Request timed out"           │
       └────────────────────────────────────────┘
    
    4. Done Event:
       ┌────────────────────────────────────────┐
       │ type: DONE                             │
       │ (signals end of stream)                │
       └────────────────────────────────────────┘
    """
    type: StreamEventType        # What kind of event is this?
    content: str = ""           # Text content (for REASONING/CONTENT)
    error: str | None = None    # Error code (for ERROR events)
    message: str | None = None  # Error message (for ERROR events)



# CUSTOM EXCEPTION

class APIError(Exception):
    """
    Custom exception for API-related errors.
    
    Why create a custom exception?
    - More specific than generic Exception
    - Can store additional info (status_code)
    - Makes error handling cleaner
    
    Example:
        try:
            client.health_check()
        except APIError as e:
            print(f"API Error: {e.message}")
            if e.status_code:
                print(f"Status: {e.status_code}")
    """
    def __init__(self, message: str, status_code: int | None = None):
        self.message = message        # Human-readable error description
        self.status_code = status_code  # HTTP status code (404, 500, etc.)
        super().__init__(message)     # Call parent Exception class



# NEUROMIND API CLIENT

class NeuroMindClient:
    """
    Client for interacting with the NeuroMind REST API.
    
    This class acts as a bridge between your application and the API server:
    
    ┌──────────────────────────────────────────────────────────────────┐
    │                    NeuroMindClient                               │
    ├──────────────────────────────────────────────────────────────────┤
    │  Methods:                          API Endpoints:                │
    │  • health_check()          →       GET  /health                  │
    │  • list_personas()         →       GET  /personas                │
    │  • list_threads()          →       GET  /threads                 │
    │  • get_or_create_thread()  →       GET  /threads/{name}          │
    │  • clear_messages()        →       DELETE /threads/{name}/msgs   │
    │  • stream_chat()           →       POST /threads/{name}/chat     │
    └──────────────────────────────────────────────────────────────────┘
    
    Usage:
        client = NeuroMindClient("http://localhost:8000")
        client.health_check()  # Verify server is running
        for event in client.stream_chat("master", "Hello!"):
            print(event.content)
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 60):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
                     Examples: "http://localhost:8000"
                              "https://api.neuromind.com"
            timeout: How long to wait for responses (seconds)
                    Default: 60 seconds
        
        Connection Setup:
        ┌────────────────────┐
        │  NeuroMindClient   │
        │                    │
        │  base_url:         │ "http://localhost:8000"
        │  timeout:          │ 60 seconds
        └────────┬───────────┘
                 │
                 ↓ Makes HTTP requests to
        ┌────────────────────┐
        │   API Server       │
        │   (FastAPI app)    │
        └────────────────────┘
        """
        # Remove trailing slash to avoid double slashes in URLs
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    
    # HEALTH CHECK
   
    def health_check(self) -> dict:
        """
        Check if the API server is healthy and responsive.
        
        Returns:
            Dictionary with server status info
        
        Flow:
        ┌──────────────┐
        │ health_check │
        └──────┬───────┘
               │
               ↓ GET /health
        ┌──────────────────┐
        │   API Server     │
        ├──────────────────┤
        │ Status: OK       │ <─── Returns {"status": "ok"}
        │ Uptime: 5h       │
        └──────────────────┘
        
        Error Handling:
        • ConnectionError → Server not running
        • Timeout        → Server not responding
        
        Example:
            try:
                status = client.health_check()
                print("Server is healthy!")
            except APIError as e:
                print(f"Server unreachable: {e.message}")
        """
        try:
            # Short timeout for health checks (5 seconds)
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()  # Raise exception for 4xx/5xx status
            return response.json()
        except requests.exceptions.ConnectionError:
            raise APIError("Could not connect to API server. Is it running?")
        except requests.exceptions.Timeout:
            raise APIError("Health check timed out.")

    
    # LIST PERSONAS

    def list_personas(self) -> List[dict]:
        """
        Get all available AI personas from the server.
        
        Returns:
            List of persona dictionaries with their properties
        
        Example Response:
            [
                {
                    "name": "neuromind",
                    "description": "General assistant",
                    "temperature": 0.7
                },
                {
                    "name": "coder",
                    "description": "Programming expert",
                    "temperature": 0.2
                }
            ]
        
        API Call:
        ┌──────────────────┐
        │ list_personas()  │
        └────────┬─────────┘
                 │
                 ↓ GET /personas
        ┌─────────────────────────────┐
        │      API Server             │
        ├─────────────────────────────┤
        │ Returns list of personas    │
        │ [{name, desc, temp}, ...]   │
        └─────────────────────────────┘
        """
        response = requests.get(f"{self.base_url}/personas", timeout=self.timeout)
        response.raise_for_status()
        return response.json()


    # LIST THREADS
    
    def list_threads(self) -> List[Tuple[str, str, int]]:
        """
        Get all conversation threads with their message counts.
        
        Returns:
            List of tuples: (thread_name, persona, message_count)
        
        Example Output:
            [
                ("master", "neuromind", 42),
                ("coding_help", "coder", 15),
                ("math_tutor", "teacher", 8)
            ]
        
        Data Transformation:
        ┌───────────────────────────────────────────┐
        │  API Response (JSON)                      │
        ├───────────────────────────────────────────┤
        │  [                                        │
        │    {                                      │
        │      "name": "master",                    │
        │      "persona": "neuromind",              │
        │      "message_count": 42                  │
        │    }                                      │
        │  ]                                        │
        └─────────────────┬─────────────────────────┘
                          │
                          ↓ Convert to tuples
        ┌─────────────────────────────────────────┐
        │  [("master", "neuromind", 42)]          │
        └─────────────────────────────────────────┘
        """
        response = requests.get(f"{self.base_url}/threads", timeout=self.timeout)
        response.raise_for_status()
        threads = response.json()
        # Convert list of dicts to list of tuples
        return [(t["name"], t["persona"], t["message_count"]) for t in threads]

    
    # GET OR CREATE THREAD
    
    def get_or_create_thread(
        self, name: str, persona: Persona = Persona.NEUROMIND
    ) -> ThreadInfo:
        """
        Get existing thread or create new one if it doesn't exist.
        
        Args:
            name: Thread name (e.g., "master", "coding_help")
            persona: AI personality to use (default: NEUROMIND)
        
        Returns:
            ThreadInfo object with id, name, and persona
        
        Two-Step Process:
        
        Step 1: Try to GET existing thread
        ┌─────────────────────────────────────┐
        │ GET /threads/{name}                 │
        └───────────┬─────────────────────────┘
                    │
            ┌───────┴────────┐
            │                │
        ┌───▼────┐       ┌───▼────┐
        │ 200 OK │       │  404   │
        │ Found! │       │ Not    │
        │ Return │       │ Found  │
        └────────┘       └───┬────┘
                             │
      Step 2: POST to create │
        ┌────────────────────▼─────────────┐
        │ POST /threads                    │
        │ Body: {name, persona}            │
        │ Returns: new ThreadInfo          │
        └──────────────────────────────────┘
        
        Example:
            # First call creates thread
            thread = client.get_or_create_thread("debug", Persona.LOGICIAN)
            print(f"Thread ID: {thread.id}")
            
            # Second call returns existing
            same = client.get_or_create_thread("debug")
        """
        # Step 1: Try to get existing thread
        response = requests.get(f"{self.base_url}/threads/{name}", timeout=self.timeout)

        if response.status_code == 200:
            # Thread exists, return it
            data = response.json()
            return ThreadInfo(id=data["id"], name=data["name"], persona=data["persona"])

        # Step 2: Thread doesn't exist, create it
        response = requests.post(
            f"{self.base_url}/threads",
            json={"name": name, "persona": persona.value},  # .value gets enum string
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return ThreadInfo(id=data["id"], name=data["name"], persona=data["persona"])


    # CLEAR MESSAGES

    def clear_messages(self, thread_name: str) -> None:
        """
        Delete all messages from a thread (thread itself remains).
        
        Args:
            thread_name: Name of thread to clear
        
        Example:
            client.clear_messages("master")
            print("All messages deleted!")
        
        HTTP Method: DELETE
        ┌──────────────────────────────────────┐
        │ DELETE /threads/{name}/messages      │
        └────────────┬─────────────────────────┘
                     │
                     ↓
        ┌──────────────────────────────────────┐
        │ Thread: "master"                     │
        │ ├─ Message 1    Deleted              │
        │ ├─ Message 2    Deleted              │
        │ └─ Message 3    Deleted              │
        │                                      │
        │ Thread still exists (empty)          │
        └──────────────────────────────────────┘
        """
        response = requests.delete(
            f"{self.base_url}/threads/{thread_name}/messages", 
            timeout=self.timeout
        )
        response.raise_for_status()

    
    # STREAM CHAT (SSE)
    
    def stream_chat(
        self, thread_name: str, content: str
    ) -> Generator[StreamEvent, None, None]:
        """
        Send a message and stream the AI response in real-time.
        
        Args:
            thread_name: Which thread to send message to
            content: The message text
        
        Yields:
            StreamEvent objects as they arrive
        
        Server-Sent Events (SSE) Explained:
        ════════════════════════════════════════════════════════════════
        
        Traditional HTTP:
        ┌────────────┐  Request   ┌────────────┐
        │   Client   │ ─────────→ │   Server   │
        │            │ ←───────── │            │
        └────────────┘  Response  └────────────┘
        (wait for full response)
        
        Server-Sent Events (SSE):
        ┌────────────┐            ┌────────────┐
        │   Client   │ ─────────→ │   Server   │
        │            │ ←───────── │ "Async"    │  (event 1)
        │            │ ←───────── │ " is"      │  (event 2)
        │            │ ←───────── │ " useful"  │  (event 3)
        │            │ ←───────── │ [DONE]     │  (event 4)
        └────────────┘            └────────────┘
        (receive chunks as they're generated)
        
        SSE Format:
        ════════════════════════════════════════════════════════════════
        data: {"type": "reasoning", "content": "Let me think..."}
        
        data: {"type": "content", "content": "Async allows"}
        
        data: {"type": "content", "content": " concurrent"}
        
        data: {"type": "done"}
        ════════════════════════════════════════════════════════════════
        
        Usage Example:
            for event in client.stream_chat("master", "What is async?"):
                if event.type == StreamEventType.REASONING:
                    print(f"[Thinking] {event.content}")
                elif event.type == StreamEventType.CONTENT:
                    print(event.content, end="", flush=True)
                elif event.type == StreamEventType.ERROR:
                    print(f"Error: {event.message}")
                elif event.type == StreamEventType.DONE:
                    print("[Complete]")
        
        Event Flow:
        ┌─────────────────────────────────────────────────────────────┐
        │ 1. POST /threads/{name}/chat                                │
        │    Body: {"content": "What is async?"}                      │
        │    stream=True (keep connection open)                       │
        └───────────────────┬─────────────────────────────────────────┘
                            │
                            ↓
        ┌─────────────────────────────────────────────────────────────┐
        │ 2. Server processes and streams back                        │
        │    • Reasoning (if enabled)                                 │
        │    • Content chunks (word by word or sentence by sentence)  │
        │    • Done signal                                            │
        └───────────────────┬─────────────────────────────────────────┘
                            │
                            ↓
        ┌─────────────────────────────────────────────────────────────┐
        │ 3. Client yields StreamEvent for each chunk                 │
        │    Your code processes events in real-time                  │
        └─────────────────────────────────────────────────────────────┘
        """
        
        # STEP 1: Make POST request with streaming enabled

        try:
            response = requests.post(
                f"{self.base_url}/threads/{thread_name}/chat",
                json={"content": content},
                stream=True,  # CRITICAL: Keeps connection open for streaming
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            # Server not reachable
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error="connection_failed",
                message="Could not connect to API server.",
            )
            return
        except requests.exceptions.Timeout:
            # Request took too long
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error="timeout",
                message="Request timed out.",
            )
            return

        
        # STEP 2: Process incoming SSE lines

        # iter_lines() yields one line at a time as they arrive
        for line in response.iter_lines(decode_unicode=True):
            # Skip empty lines and non-data lines
            if not line or not line.startswith("data:"):
                continue

            try:
                # Parse JSON from "data: {...}" format
                # line[5:] removes "data:" prefix
                event_data = json.loads(line[5:].strip())
                event_type = event_data.get("type", "unknown")

                
                # STEP 3: Convert to StreamEvent based on type
                
                if event_type == "reasoning":
                    # AI's thought process (optional)
                    yield StreamEvent(
                        type=StreamEventType.REASONING, 
                        content=event_data["content"]
                    )
                elif event_type == "content":
                    # Actual response text
                    yield StreamEvent(
                        type=StreamEventType.CONTENT, 
                        content=event_data["content"]
                    )
                elif event_type == "error":
                    # Something went wrong
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=event_data.get("error"),
                        message=event_data.get("message"),
                    )
                elif event_type == "done":
                    # Stream finished
                    yield StreamEvent(type=StreamEventType.DONE)
            except json.JSONDecodeError:
                # Malformed JSON, skip this line
                continue



# USAGE EXAMPLES

"""
Complete workflow examples:

1. Basic Setup:
   ───────────────────────────────────────────────────────────────────────
   from api_client import NeuroMindClient, StreamEventType
   
   client = NeuroMindClient("http://localhost:8000")
   
   # Verify server is running
   try:
       status = client.health_check()
       print("Server is online!")
   except APIError as e:
       print(f"Server error: {e.message}")
       exit(1)

2. List Available Resources:
   ───────────────────────────────────────────────────────────────────────
   # See available personas
   personas = client.list_personas()
   for p in personas:
       print(f"- {p['name']}: {p['description']}")
   
   # See existing threads
   threads = client.list_threads()
   for name, persona, count in threads:
       print(f"{name} ({persona}): {count} messages")

3. Create/Get Thread and Chat:
   ───────────────────────────────────────────────────────────────────────
   from neuromind.config import Persona
   
   # Create or get thread
   thread = client.get_or_create_thread("my_chat", Persona.CODER)
   print(f"Using thread: {thread.name}")
   
   # Send message and stream response
   print("You: What is async in Python?")
   print("AI: ", end="", flush=True)
   
   for event in client.stream_chat(thread.name, "What is async in Python?"):
       if event.type == StreamEventType.REASONING:
           # Optional: show AI's thinking
           print(f"\n[Thinking: {event.content}]")
       
       elif event.type == StreamEventType.CONTENT:
           # Print response as it arrives
           print(event.content, end="", flush=True)
       
       elif event.type == StreamEventType.ERROR:
           print(f"\nError: {event.message}")
           break
       
       elif event.type == StreamEventType.DONE:
           print("\n")  # Newline after complete response

4. Thread Management:
   ───────────────────────────────────────────────────────────────────────
   # Clear conversation history
   client.clear_messages("my_chat")
   print("Thread cleared!")
   
   # Start fresh conversation in same thread
   for event in client.stream_chat("my_chat", "Hello again!"):
       if event.type == StreamEventType.CONTENT:
           print(event.content, end="")

5. Error Handling:
   ───────────────────────────────────────────────────────────────────────
   try:
       for event in client.stream_chat("master", "Hello"):
           if event.type == StreamEventType.ERROR:
               print(f"Stream error: {event.message}")
               if event.error == "timeout":
                   print("Try again with shorter input")
               elif event.error == "connection_failed":
                   print("Check if server is running")
   except APIError as e:
       print(f"Request failed: {e.message}")
       if e.status_code:
           print(f"HTTP Status: {e.status_code}")

Key Concepts:
═══════════════════════════════════════════════════════════════════════════
• REST API: Web service that uses HTTP methods (GET, POST, DELETE)
• SSE: Server-Sent Events - one-way streaming from server to client
• Generator: Function that yields values one at a time (using yield)
• Streaming: Receiving data incrementally rather than all at once
• Timeout: Maximum time to wait for response before giving up
"""