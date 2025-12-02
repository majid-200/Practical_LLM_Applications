"""
MAIN APPLICATION MODULE

This is the entry point and main orchestrator of the NeuroMind CLI application.

Application Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                            NeuroApp                                     │
│                        (Main Controller)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐           │
│  │ UIManager    │      │ APIClient    │      │ Config       │           │
│  │              │      │              │      │              │           │
│  │ • Display    │      │ • HTTP calls │      │ • Settings   │           │
│  │ • Prompts    │      │ • Streaming  │      │ • Paths      │           │
│  │ • Formatting │      │ • Errors     │      │ • Models     │           │
│  └──────────────┘      └──────────────┘      └──────────────┘           │
│         ↑                      ↑                      ↑                 │
│         └──────────────────────┴──────────────────────┘                 │
│                          Orchestrated by                                │
│                           NeuroApp                                      │
└─────────────────────────────────────────────────────────────────────────┘

Responsibilities:
✓ Initialize all components
✓ Handle user commands (/new, /switch, /list, /clear, /exit)
✓ Manage conversation flow
✓ Coordinate between UI and API
✓ Error handling and recovery
"""

import sys
from typing import List

from Project.api_client import APIError, NeuroMindClient, StreamEventType, ThreadInfo
from Project.config import Config, Persona
from Project.ui_manager import UIManager



# MAIN APPLICATION CLASS

class NeuroApp:
    """
    Main application controller for NeuroMind CLI.
    
    This class ties everything together and manages the application lifecycle:
    
    Lifecycle Flow:
    ┌────────────────────────────────────────────────────────────┐
    │ 1. __init__()                                              │
    │    ├─ Create API client                                    │
    │    ├─ Create UI manager                                    │
    │    ├─ Health check server                                  │
    │    └─ Load/create default thread                           │
    ├────────────────────────────────────────────────────────────┤
    │ 2. run()                                                   │
    │    ├─ Show header                                          │
    │    └─ Main loop:                                           │
    │       ├─ Get user input                                    │
    │       ├─ Handle commands OR                                │
    │       └─ Stream AI response                                │
    ├────────────────────────────────────────────────────────────┤
    │ 3. Exit                                                    │
    │    └─ Graceful shutdown                                    │
    └────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the NeuroMind application.
        
        Args:
            base_url: URL of the API server
        
        Initialization Steps:
        ┌──────────────────────────────────────────────────────┐
        │ Step 1: Create Components                            │
        │ ┌──────────────┐  ┌──────────────┐                   │
        │ │  API Client  │  │  UI Manager  │                   │
        │ └──────────────┘  └──────────────┘                   │
        ├──────────────────────────────────────────────────────┤
        │ Step 2: Health Check                                 │
        │ ┌──────────────────────────────────────┐             │
        │ │ Is server running?                   │             │
        │ │ ├─ Yes: Get model name               │             │
        │ │ └─ No: Show error & exit             │             │
        │ └──────────────────────────────────────┘             │
        ├──────────────────────────────────────────────────────┤
        │ Step 3: Load Default Thread                          │
        │ ┌──────────────────────────────────────┐             │
        │ │ Get or create "master" thread        │             │
        │ │ (from Config.DEFAULT_THREAD)         │             │
        │ └──────────────────────────────────────┘             │
        └──────────────────────────────────────────────────────┘
        
        Why health check first?
        - Fail fast if server not running
        - Get model info for display
        - Better UX than cryptic errors later
        """

        # STEP 1: Initialize core components
   
        self.client = NeuroMindClient(base_url)  # API communication
        self.ui = UIManager()                    # Terminal UI
        self.active_thread: ThreadInfo | None = None  # Current conversation


        # STEP 2: Verify server is running and get model info

        try:
            health = self.client.health_check()  # Returns: {"status": "ok", "model": "..."}
            self.model_name = health.get("model", "unknown")
        except APIError as e:
            # Server not reachable - show helpful error and exit
            self.ui.print_critical_error(
                f"{e.message}\nMake sure the server is running: python start_server.py"
            )
            sys.exit(1)  # Exit code 1 indicates error


        # STEP 3: Load or create the default thread

        # Config.DEFAULT_THREAD is typically "master"
        self.active_thread = self.client.get_or_create_thread(Config.DEFAULT_THREAD)


    # COMMAND: /list

    def _cmd_list(self):
        """
        Display all available threads in a table.
        
        Command: /list
        
        Execution Flow:
        ┌────────────────────────────────────────┐
        │ User types: /list                      │
        └──────────┬─────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────────────────────────┐
        │ 1. Call API to get all threads           │
        │    client.list_threads()                 │
        └──────────┬───────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────────────────────────┐
        │ 2. Display in formatted table            │
        │    ui.show_thread_list(...)              │
        │                                          │
        │ Output:                                  │
        │ ┌─ Memory Banks ──────────────┐          │
        │ │ ➤ master   | neuromind | 42│          │
        │ │   coding    | coder     | 15│          │
        │ └─────────────────────────────┘          │
        └──────────────────────────────────────────┘
        
        Why underscore prefix (_cmd_list)?
        - Convention for "private" methods
        - Not meant to be called from outside
        - Just for internal command handling
        """
        # Get list of all threads from API
        threads = self.client.list_threads()
        
        # Display in formatted table with active thread marked
        self.ui.show_thread_list(threads, self.active_thread.name)


    # COMMAND: /new

    def _cmd_new(self, args: List[str]):
        """
        Create a new thread with a chosen persona.
        
        Command: /new <thread_name>
        Example: /new coding_session
        
        Execution Flow:
        ┌────────────────────────────────────────────────────────┐
        │ User types: /new coding_session                        │
        └──────────┬─────────────────────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────────────────────────────────────┐
        │ 1. Validate: Did user provide thread name?           │
        │    ├─ No: Show error, return                         │
        │    └─ Yes: Continue                                  │
        └──────────┬───────────────────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────────────────────────────────────┐
        │ 2. Prompt user to select persona                     │
        │    Select Persona:                                   │
        │      1. neuromind                                    │
        │      2. coder                                        │
        │      3. teacher                                      │
        │      4. logician                                     │
        │    Choice [1]: _                                     │
        └──────────┬───────────────────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────────────────────────────────────┐
        │ 3. Create thread with selected persona               │
        │    client.get_or_create_thread(name, persona)        │
        └──────────┬───────────────────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────────────────────────────────────┐
        │ 4. Switch to new thread                              │
        │    self.active_thread = new_thread                   │
        │    Show: "Switched to 'coding_session' (coder)"      │
        └──────────────────────────────────────────────────────┘
        
        Args Validation:
        • args is a list of words after the command
        • /new my_thread → args = ["my_thread"]
        • /new → args = [] (empty, show error)
        """

        # STEP 1: Validate input

        if not args:
            self.ui.print_error("Usage: /new <thread_name>")
            return  # Exit early if no thread name provided


        # STEP 2: Get thread name and prepare persona choices

        name = args[0]  # First argument is the thread name
        all_personas = list(Persona)  # Get all available personas from enum
        

        # STEP 3: Let user choose persona interactively

        choice = self.ui.prompt_choice(
            "Select Persona", 
            [p.value for p in all_personas]  # Convert enum to string list
        )
        persona = all_personas[choice]  # Get selected persona enum


        # STEP 4: Create thread and switch to it

        self.active_thread = self.client.get_or_create_thread(name, persona)
        self.ui.print_info(f"Switched to '{name}' ({persona.value})")


    # COMMAND: /switch

    def _cmd_switch(self, args: List[str]):
        """
        Switch to an existing thread (or create if doesn't exist).
        
        Command: /switch <thread_name>
        Example: /switch master
        
        Difference from /new:
        ┌──────────────────────────────────────────────────────┐
        │ /new                  vs    /switch                  │
        ├──────────────────────────────────────────────────────┤
        │ • Always prompts for     • Uses existing persona     │
        │   persona                  or default                │
        │ • Creates new thread     • Gets existing or creates  │
        │ • Interactive            • Quick switch              │
        │ • Use for: New convos    • Use for: Known threads    │
        └──────────────────────────────────────────────────────┘
        
        Execution Flow:
        ┌────────────────────────────────────────┐
        │ User types: /switch master             │
        └──────────┬─────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────────────────────────┐
        │ 1. Validate: thread name provided?       │
        │    ├─ No: Show error                     │
        │    └─ Yes: Continue                      │
        └──────────┬───────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────────────────────────┐
        │ 2. Get or create thread (default persona)│
        │    client.get_or_create_thread(name)     │
        └──────────┬───────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────────────────────────┐
        │ 3. Switch active thread                  │
        │    Show: "Active thread: master"         │
        └──────────────────────────────────────────┘
        """
        # Validate input
        if not args:
            self.ui.print_error("Usage: /switch <thread_name>")
            return

        # Get thread name
        name = args[0]
        
        # Get or create thread (uses default persona if creating)
        self.active_thread = self.client.get_or_create_thread(name)
        
        # Confirm switch
        self.ui.print_info(f"Active thread: {name}")


    # COMMAND: /clear

    def _cmd_clear(self):
        """
        Clear all messages from the current thread.
        
        Command: /clear
        
        Safety Feature: Confirmation Required
        ┌────────────────────────────────────────────────────┐
        │ Why require confirmation?                          │
        │ • Destructive operation (can't undo!)              │
        │ • User might type /clear accidentally              │
        │ • Better safe than sorry                           │
        └────────────────────────────────────────────────────┘
        
        Execution Flow:
        ┌────────────────────────────────────────┐
        │ User types: /clear                     │
        └──────────┬─────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────────────────────────┐
        │ 1. Ask for confirmation                  │
        │    Wipe memory for 'master'? (y/n): _    │
        └──────────┬───────────────────────────────┘
                   │
           ┌───────┴────────┐
           │                │
        Yes│                │No
           ↓                ↓
        ┌─────────────┐  ┌──────────────┐
        │ 2. Delete   │  │ 2. Cancel    │
        │    messages │  │    (do       │
        │             │  │    nothing)  │
        │ 3. Show     │  └──────────────┘
        │   "Memory   │
        │    wiped"   │
        └─────────────┘
        
        What gets deleted?
        • All messages in the thread
        • Thread itself remains (just empty)
        • Other threads not affected
        """
        # Ask for confirmation before deleting
        if self.ui.confirm(f"Wipe memory for '{self.active_thread.name}'?"):
            # User confirmed, delete messages
            self.client.clear_messages(self.active_thread.name)
            self.ui.print_info("Memory wiped.")
        # If user said no, do nothing (implicit else)


    # PROCESS STREAMING RESPONSE

    def _process_stream(self, events) -> None:
        """
        Process and display streaming AI response events.
        
        Args:
            events: Generator yielding StreamEvent objects
        
        
        Stream Processing Architecture:
        ═══════════════════════════════════════════════════════════════
        
        ┌────────────────────────────────────────────────────────┐
        │ API Server                                             │
        │ Generates response word-by-word                        │
        └───────────────────┬────────────────────────────────────┘
                            │ Server-Sent Events (SSE)
                            ↓
        ┌────────────────────────────────────────────────────────┐
        │ events Generator (from api_client.stream_chat)         │
        │ Yields: StreamEvent objects                            │
        └───────────────────┬────────────────────────────────────┘
                            │
                            ↓
        ┌────────────────────────────────────────────────────────┐
        │ _process_stream (this method)                          │
        │ ┌────────────────────────────────────┐                 │
        │ │ Buffers:                           │                 │
        │ │ • thought_buffer  = ""             │                 │
        │ │ • response_buffer = ""             │                 │
        │ └────────────────────────────────────┘                 │
        │                                                        │
        │ For each event:                                        │
        │ ├─ REASONING → append to thought_buffer                │
        │ ├─ CONTENT   → append to response_buffer               │
        │ ├─ ERROR     → raise exception                         │
        │ └─ DONE      → break loop                              │
        │                                                        │
        │ After each event:                                      │
        │ └─ Update live display with both buffers               │
        └────────────────────┬───────────────────────────────────┘
                             │
                             ↓
        ┌────────────────────────────────────────────────────────┐
        │ Terminal Display (Live updating)                       │
        │ ╭─── Reasoning ────────────────────╮                   │
        │ │ Let me think about this...       │                   │
        │ ╰──────────────────────────────────╯                   │
        │                                                        │
        │ **Response**: Async is a pattern█                      │
        └────────────────────────────────────────────────────────┘
        
        Why use buffers?
        ════════════════════════════════════════════════════════════
        • Events come in small chunks: "Async", " is", " a", "..."
        • Buffers accumulate: "Async" → "Async is" → "Async is a"
        • Display shows full accumulated text
        • Without buffers, you'd only see the latest chunk!
        
        Event Types Handling:
        ════════════════════════════════════════════════════════════
        1. REASONING: Optional AI thinking process
           → Append to thought_buffer
           → Displayed in dim panel above response
        
        2. CONTENT: Main response text
           → Append to response_buffer
           → Displayed as markdown below reasoning
        
        3. ERROR: Something went wrong
           → Raise exception to stop processing
           → Caught by main loop's error handler
        
        4. DONE: Stream complete
           → Break loop, done processing
           → Live display remains on screen
        """


        # Initialize string buffers to hold the accumulating text
        thought_buffer = ""
        response_buffer = ""


        # LIVE DISPLAY CONTEXT MANAGER

        # self.ui.stream_response creates a Live display context.
        # It keeps the UI active and rewritable until the block exits.
        with self.ui.stream_response(self.active_thread.name) as live:
            
            # Loop through the generator yielding Server-Sent Events
            for event in events:
                
                # Case 1: Reasoning (Inner monologue/Chain of thought)
                if event.type == StreamEventType.REASONING:
                    thought_buffer += event.content
                    
                # Case 2: Content (The actual answer)
                elif event.type == StreamEventType.CONTENT:
                    response_buffer += event.content
                    
                # Case 3: Error (Server reported an issue)
                elif event.type == StreamEventType.ERROR:
                    # Raise immediately to be caught by the run() loop
                    raise APIError(event.message or "Unknown error")
                    
                # Case 4: Done (Stream finished)
                elif event.type == StreamEventType.DONE:
                    break

                
                # UPDATE THE SCREEN
                
                # 1. Create a renderable object (Panel/Group) from current buffers
                renderable = self.ui.render_stream_group(
                    thought_buffer, response_buffer
                )
                
                # 2. Refresh the terminal display with the new state
                live.update(renderable)



    # MAIN EVENT LOOP (REPL)
    
    def run(self):
        """
        The main application loop (Read-Eval-Print Loop).
        
        This method keeps the application running until the user exits.
        
        The Loop Architecture:
        ┌─────────────────────────────────────────────────────────────┐
        │                        START                                │
        │                  (Show App Header)                          │
        └──────────────────────────┬──────────────────────────────────┘
                                   │
                                   ↓
        ┌─────────────────────────────────────────────────────────────┐
        │ 1. READ INPUT                                               │
        │    prompt > _                                               │
        └──────────────────────────┬──────────────────────────────────┘
                                   │
                                   ↓
        ┌─────────────────────────────────────────────────────────────┐
        │ 2. EVALUATE                                                 │
        │    Is it a command? (/new, /exit)                           │
        │    ├─ YES ──> Execute Command ──> (Exit if /exit)           │
        │    └─ NO  ──> Call API (Stream Chat)                        │
        └──────────────────────────┬──────────────────────────────────┘
                                   │
                                   ↓
        ┌─────────────────────────────────────────────────────────────┐
        │ 3. PRINT / DISPLAY                                          │
        │    Show command result OR Stream AI response                │
        └──────────────────────────┬──────────────────────────────────┘
                                   │
                                   └──────────────────────────────────┘
                                           Loop back to 1.
        
        Error Handling Strategy:
        • The loop is wrapped in try/except to prevent crashes.
        • API Errors → Printed nicely, loop continues.
        • Ctrl+C     → Caught nicely, loop continues (user must use /exit).
        • Unexpected → Printed, loop continues.
        """
        # Show the welcome banner
        self.ui.show_header(self.model_name, self.active_thread.name)

        while True:
            try:

                # 1. GET USER INPUT

                # ui.get_user_input handles the prompt styling (e.g., [master] >)
                user_input = self.ui.get_user_input(self.active_thread.name)


                # 2. CHECK FOR COMMANDS (starts with "/")

                if user_input.startswith("/"):
                    # Parse input: "/new coding" -> cmd="/new", args=["coding"]
                    parts = user_input.strip().split()
                    cmd = parts[0].lower()
                    args = parts[1:]

                    # ─── Command Routing ───
                    if cmd == "/exit":
                        self.ui.print_info("Goodbye.")
                        break  # Break the while loop to terminate program
                        
                    elif cmd == "/list":
                        self._cmd_list()
                        
                    elif cmd == "/new":
                        self._cmd_new(args)
                        
                    elif cmd == "/switch":
                        self._cmd_switch(args)
                        
                    elif cmd == "/clear":
                        self._cmd_clear()
                        
                    else:
                        self.ui.print_error("Unknown command.")
                    
                    # Skip the rest of the loop (don't send commands to LLM)
                    continue


                # 3. SEND TO AI (If not a command)

                # Start the generator
                events = self.client.stream_chat(self.active_thread.name, user_input)
                
                # Hand over to the stream processor defined earlier
                self._process_stream(events)


            # EXCEPTION HANDLING

            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                self.ui.print_info("\nUse /exit to quit.")
                
            except APIError as e:
                # Handle expected API errors (e.g., server down, validation)
                self.ui.print_error(e.message)
                
            except Exception as e:
                # Handle unexpected crashes without killing the app
                self.ui.print_error(str(e))



# ENTRY POINT

if __name__ == "__main__":
    """
    Standard Python entry point.
    
    Responsibilities:
    1. Parse command line arguments (host/port configuration)
    2. Instantiate the application
    3. Start the run loop
    """
    import argparse

    # Set up argument parser
    # Allows running like: python app.py --server http://192.168.1.5:8000
    parser = argparse.ArgumentParser(description="NeuroMind CLI")
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="API server URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    # Initialize app with provided URL
    app = NeuroApp(base_url=args.server)
    
    # Start the application
    app.run()