"""
THREAD MANAGER MODULE

This module manages conversation threads and message history using SQLite.

Think of it as a filing cabinet for AI conversations:
┌─────────────────────────────────────────┐
│         THREAD MANAGER                  │
│  ┌───────────┐  ┌──────────────────┐    │
│  │  Thread 1 │  │  Message History │    │
│  │  Thread 2 │  │  - User: "Hi"    │    │
│  │  Thread 3 │  │  - AI: "Hello!"  │    │
│  └───────────┘  └──────────────────┘    │
└─────────────────────────────────────────┘
"""

import sqlite3
from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.messages import (
    AIMessage,       # Represents messages from the AI
    BaseMessage,     # Base class for all message types
    HumanMessage,    # Represents messages from the user
)

from Project.config import Persona


# THREAD DATA CLASS

@dataclass
class Thread:
    """
    Represents a single conversation thread.
    
    Structure:
    ┌─────────────────────────────────────┐
    │           Thread                    │
    ├─────────────────────────────────────┤
    │ id: int       → 1                   │  Unique identifier (database ID)
    │ name: str     → "master"            │  Human-readable thread name
    │ persona: str  → "neuromind"         │  AI personality for this thread
    └─────────────────────────────────────┘
    
    Example:
        Thread(id=1, name="coding_help", persona="coder")
    """
    id: int          # Primary key from database (auto-generated)
    name: str        # Thread name (must be unique, like "master" or "coding_help")
    persona: str     # Which AI personality to use (e.g., "neuromind", "teacher")



# THREAD MANAGER CLASS
class ThreadManager:
    """
    Manages conversation threads and their message history in SQLite.
    
    Database Schema:
    ┌──────────────────────────────────────────────────────────────┐
    │                        threads                               │
    ├──────────────────────────────────────────────────────────────┤
    │ id (PK)  │  name (UNIQUE)  │  persona   │  created_at        │
    │    1     │    "master"     │ "neuromind"│  2025-01-15 10:00  │
    │    2     │    "coding"     │  "coder"   │  2025-01-15 11:30  │
    └──────────────────────────────────────────────────────────────┘
                            │
                            │ (one-to-many relationship)
                            ↓
    ┌──────────────────────────────────────────────────────────────┐
    │                       messages                               │
    ├──────────────────────────────────────────────────────────────┤
    │ id (PK)  │ thread_id (FK) │  role   │  content  │ timestamp  │
    │    1     │       1        │ "human" │  "Hello"  │  10:01     │
    │    2     │       1        │  "ai"   │  "Hi!"    │  10:01     │
    │    3     │       2        │ "human" │  "Debug?" │  11:31     │
    └──────────────────────────────────────────────────────────────┘
    
    Responsibilities:
    - Create and retrieve conversation threads
    - Store and fetch message history
    - Initialize database schema
    - Manage SQLite connection
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the ThreadManager with a database connection.
        
        Args:
            db_path: Path to SQLite database file (e.g., "data/neuromind.db")
        
        Connection Setup:
        ┌─────────────────────────────────────────┐
        │  ThreadManager                          │
        │    │                                    │
        │    └──> SQLite Connection               │
        │           └──> neuromind.db             │
        │                  ├── threads table      │
        │                  └── messages table     │
        └─────────────────────────────────────────┘
        """
        # check_same_thread=False allows connection use across async thread boundaries
        # Safe here since FastAPI creates one ThreadManager per request via Depends()
        # Without this, you'd get errors when using async/await with SQLite
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # row_factory lets us access columns by name instead of index
        # Instead of: row[0], row[1], row[2]
        # We can do:  row["id"], row["name"], row["persona"]
        self.conn.row_factory = sqlite3.Row
        
        # Create database tables if they don't exist
        self._migrate()


    # DATABASE INITIALIZATION
    def _migrate(self):
        """
        Create database tables if they don't already exist.
        
        This is called "migration" because it sets up the database schema.
        It's safe to run multiple times - won't recreate if tables exist.
        
        Tables Created:
        1. threads  - Stores conversation threads
        2. messages - Stores individual messages within threads
        """
        with self.conn:  # Auto-commits on success, auto-rollbacks on error
            
            # CREATE THREADS TABLE
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Auto-incrementing ID
                    name TEXT UNIQUE NOT NULL,             -- Thread name (must be unique)
                    persona TEXT NOT NULL,                 -- AI personality type
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Auto-set creation time
                )
            """)
            
            # CREATE MESSAGES TABLE
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Auto-incrementing ID
                    thread_id INTEGER,                     -- Which thread this belongs to
                    role TEXT NOT NULL,                    -- "human" or "ai"
                    content TEXT NOT NULL,                 -- The actual message text
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- When message was sent
                    FOREIGN KEY(thread_id) REFERENCES threads(id)   -- Link to threads table
                )
            """)


    # THREAD RETRIEVAL
    def get_thread(self, name: str) -> Thread | None:
        """
        Retrieve a thread by name.
        
        Args:
            name: Thread name to search for (e.g., "master")
        
        Returns:
            Thread object if found, None if not found
        
        Example:
            thread = manager.get_thread("master")
            if thread:
                print(f"Found thread #{thread.id}")
            else:
                print("Thread doesn't exist yet")
        
        SQL Query Flow:
        ┌──────────────────────────────────────────┐
        │ SELECT * FROM threads WHERE name = ?     │
        │                                          │
        │ Input:  "master"                         │
        │ Output: (1, "master", "neuromind", ...)  │
        └──────────────────────────────────────────┘
        """
        # ? is a placeholder - prevents SQL injection attacks
        row = self.conn.execute(
            "SELECT * FROM threads WHERE name = ?", (name,)
        ).fetchone()  # Get first matching row (or None)

        if row:
            # Convert database row to Thread object
            return Thread(id=row["id"], name=row["name"], persona=row["persona"])
        return None


    # THREAD CREATION OR RETRIEVAL
    def get_or_create_thread(
        self, name: str, persona: Persona = Persona.NEUROMIND
    ) -> Thread:
        """
        Get existing thread or create new one if it doesn't exist.
        
        Args:
            name: Thread name
            persona: AI personality (defaults to NEUROMIND)
        
        Returns:
            Thread object (either existing or newly created)
        
        Flow Diagram:
        ┌─────────────────────────────────────────────┐
        │  get_or_create_thread("master")             │
        └─────────────────┬───────────────────────────┘
                          │
                ┌─────────▼─────────┐
                │ Does thread exist?│
                └─────────┬─────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
        ┌───▼────┐                 ┌───▼────┐
        │  YES   │                 │   NO   │
        │ Return │                 │ Create │
        │existing│                 │  new   │
        └────────┘                 └────────┘
        
        Example:
            # First call creates thread
            thread = manager.get_or_create_thread("coding", Persona.CODER)
            
            # Second call returns existing thread
            same_thread = manager.get_or_create_thread("coding")
        """
        # Try to get existing thread
        thread = self.get_thread(name)
        if thread:
            return thread

        # Thread doesn't exist, create it
        cursor = self.conn.execute(
            "INSERT INTO threads (name, persona) VALUES (?, ?)", 
            (name, persona.value)
        )
        self.conn.commit()  # Save changes to database
        
        # Return newly created thread with auto-generated ID
        return Thread(id=cursor.lastrowid, name=name, persona=persona.value)


    # LIST ALL THREADS
    def list_threads(self) -> List[Tuple[str, str, int]]:
        """
        Get all threads with their message counts.
        
        Returns:
            List of tuples: (thread_name, persona, message_count)
        
        Example Output:
            [
                ("master", "neuromind", 42),
                ("coding", "coder", 15),
                ("debug", "logician", 8)
            ]
        
        SQL Query Explanation:
        ┌──────────────────────────────────────────────────────────┐
        │ SELECT t.name, t.persona, COUNT(m.id) as count           │
        │ FROM threads t                                           │
        │ LEFT JOIN messages m ON t.id = m.thread_id               │
        │ GROUP BY t.id                                            │
        └──────────────────────────────────────────────────────────┘
        
        What this does:
        - SELECT: Get thread name, persona, and count of messages
        - LEFT JOIN: Include threads even if they have no messages
        - COUNT(m.id): Count how many messages in each thread
        - GROUP BY: Group results by thread
        
        Visual:
        threads         messages               Result
        ┌────┬────┐    ┌────┬──────┐         ┌────┬─────┬───┐
        │ id │name│    │ id │thread│         │name│pers │cnt│
        ├────┼────┤    ├────┼──────┤         ├────┼─────┼───┤
        │ 1  │ A  │───→│ 1  │  1   │    ───→ │ A  │ n   │ 2 │
        │ 2  │ B  │    │ 2  │  1   │         │ B  │ c   │ 0 │
        └────┴────┘    │ 3  │  3   │         └────┴─────┴───┘
                       └────┴──────┘
        """
        query = """
            SELECT t.name, t.persona, COUNT(m.id) as count
            FROM threads t 
            LEFT JOIN messages m ON t.id = m.thread_id 
            GROUP BY t.id
        """
        # Convert Row objects to tuples
        return [(r["name"], r["persona"], r["count"]) for r in self.conn.execute(query)]


    # ADD MESSAGE TO THREAD
    def add_message(self, thread_id: int, role: str, content: str):
        """
        Add a new message to a thread.
        
        Args:
            thread_id: Which thread to add message to
            role: "human" or "ai"
            content: The message text
        
        Example:
            manager.add_message(
                thread_id=1,
                role="human",
                content="What's the weather?"
            )
            manager.add_message(
                thread_id=1,
                role="ai",
                content="I don't have real-time data."
            )
        
        Message Flow:
        ┌─────────────┐
        │ User types  │ "Hello"
        └──────┬──────┘
               │
               ↓
        ┌──────────────────────────┐
        │ add_message()            │
        │ - thread_id: 1           │
        │ - role: "human"          │
        │ - content: "Hello"       │
        └──────┬───────────────────┘
               │
               ↓
        ┌──────────────────────────┐
        │ INSERT INTO messages     │
        │ Saved to database!       │
        └──────────────────────────┘
        """
        with self.conn:  # Auto-commit on success
            self.conn.execute(
                "INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)",
                (thread_id, role, content),
            )


    # GET CONVERSATION HISTORY
    def get_history(self, thread_id: int) -> List[BaseMessage]:
        """
        Retrieve all messages for a thread in chronological order.
        
        Args:
            thread_id: Which thread's history to retrieve
        
        Returns:
            List of HumanMessage and AIMessage objects
        
        Example:
            history = manager.get_history(thread_id=1)
            for msg in history:
                if isinstance(msg, HumanMessage):
                    print(f"User: {msg.content}")
                else:
                    print(f"AI: {msg.content}")
        
        Database → LangChain Conversion:
        ┌─────────────────────────────────────────────────────┐
        │ Database Row                                        │
        ├─────────────────────────────────────────────────────┤
        │ role: "human", content: "Hello"                     │
        └──────────────────┬──────────────────────────────────┘
                           ↓
                  ┌────────────────────┐
                  │ Convert to Object  │
                  └────────┬───────────┘
                           ↓
        ┌──────────────────────────────────────────────────────┐
        │ HumanMessage(content="Hello")                        │
        └──────────────────────────────────────────────────────┘
        
        Why convert to LangChain messages?
        - LangChain expects messages in this format
        - Makes it easy to pass history to AI models
        - Provides useful methods and properties
        """
        # Get all messages for this thread, oldest first
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,),
        ).fetchall()

        history = []
        for r in rows:
            # Convert database rows to LangChain message objects
            if r["role"] == "human":
                history.append(HumanMessage(content=r["content"]))
            elif r["role"] == "ai":
                history.append(AIMessage(content=r["content"]))
        return history


    # CLEAR THREAD HISTORY
    def clear_messages(self, thread_id: int):
        """
        Delete all messages from a thread (but keep the thread itself).
        
        Args:
            thread_id: Which thread to clear
        
        Example:
            manager.clear_messages(thread_id=1)
            # Thread still exists, but all messages are gone
        
        Before:
        ┌─────────────────────────────────────┐
        │ Thread "master" (id=1)              │
        │  ├─ Message 1: "Hello"              │
        │  ├─ Message 2: "Hi there!"          │
        │  └─ Message 3: "How are you?"       │
        └─────────────────────────────────────┘
        
        After clear_messages(1):
        ┌─────────────────────────────────────┐
        │ Thread "master" (id=1)              │
        │  (no messages)                      │
        └─────────────────────────────────────┘
        
        Note: This only deletes messages, not the thread.
              To delete the thread itself, you'd need a delete_thread() method.
        """
        with self.conn:  # Auto-commit
            self.conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))



# USAGE EXAMPLES
"""
Complete usage workflow:

1. Initialize manager:
   manager = ThreadManager("data/neuromind.db")

2. Create or get a thread:
   thread = manager.get_or_create_thread("my_conversation", Persona.CODER)

3. Add messages to the conversation:
   manager.add_message(thread.id, "human", "How do I use async/await?")
   manager.add_message(thread.id, "ai", "Async/await is used for...")

4. Retrieve conversation history:
   history = manager.get_history(thread.id)
   for msg in history:
       print(f"{type(msg).__name__}: {msg.content}")

5. List all threads:
   threads = manager.list_threads()
   for name, persona, count in threads:
       print(f"{name} ({persona}): {count} messages")

6. Start fresh:
   manager.clear_messages(thread.id)

Database Design Benefits:
✓ Persistent storage (survives app restarts)
✓ Multiple conversations organized by thread
✓ Full history for context-aware AI responses
✓ Easy to search, filter, and manage conversations
✓ Timestamps for every message and thread
"""