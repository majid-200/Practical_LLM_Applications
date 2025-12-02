"""
UI MANAGER MODULE

This module handles all terminal UI rendering using the Rich library.

Think of this as the "presentation layer" - it makes the CLI look beautiful:

Plain Terminal:                Rich Terminal (This module):
┌─────────────────┐           ┌──────────────────────────────────┐
│ > Hello         │           │ ╭────── NeuroMind ──────╮        │
│ AI: Hi there    │           │ │ [Thinking] Let me...  │        │
│ > What is 2+2?  │           │ │ **Response**: Here... │        │
│ AI: 4           │           │ ╰───────────────────────╯        │
└─────────────────┘           └──────────────────────────────────┘

Rich Library Features Used:
✓ Colors and styling
✓ Borders and panels
✓ Tables with alignment
✓ Markdown rendering
✓ Live updating displays
✓ Progress indicators
"""

from typing import List, Tuple

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table



# UI MANAGER CLASS

class UIManager:
    """
    Manages all terminal UI operations using Rich library.
    
    Responsibilities:
    ┌────────────────────────────────────────────────────┐
    │  UIManager                                         │
    ├────────────────────────────────────────────────────┤
    │  • Display headers and banners                     │
    │  • Show thread lists in tables                     │
    │  • Get user input with prompts                     │
    │  • Stream AI responses in real-time                │
    │  • Display errors and info messages                │
    │  • Handle confirmations and choices                │
    └────────────────────────────────────────────────────┘
    
    Rich Console is the core output handler - think of it as
    an enhanced print() that supports colors, styles, and layouts.
    """
    
    def __init__(self):
        """
        Initialize the UI Manager with a Rich Console.
        
        Console Object:
        ┌──────────────────────────────────────┐
        │      Rich Console                    │
        ├──────────────────────────────────────┤
        │  • Handles all terminal output       │
        │  • Manages colors and styles         │
        │  • Renders rich content (tables,     │
        │    panels, markdown, etc.)           │
        │  • Thread-safe for async operations  │
        └──────────────────────────────────────┘
        """
        self._console = Console()

    
    # SHOW HEADER

    def show_header(self, model: str, thread: str):
        """
        Display the application header with current settings.
        
        Args:
            model: AI model name (e.g., "qwen3:8b")
            thread: Current thread name (e.g., "master")
        
        Visual Output:
        ╭──────────────────────────────────────────────────────╮
        │ NeuroMind | CLI AI Assistant                         │
        │ Model: qwen3:8b | Thread: master                     │
        │ Cmds: /new, /switch, /list, /clear, /exit            │
        ╰──────────────────────────────────────────────────────╯
        
        Rich Markup Explained:
        • [bold cyan]text[/bold cyan]  → Blue, bold text
        • [dim]text[/dim]              → Dimmed/faded text
        • [green]text[/green]          → Green text
        • [yellow]text[/yellow]        → Yellow text
        • [magenta]text[/magenta]      → Magenta/purple text
        
        Why use Panel?
        - Adds a border around content
        - Groups related information
        - Makes UI more organized and professional
        """
        # Clear screen for fresh display
        self._console.clear()
        
        # Create and display header panel
        self._console.print(
            Panel(
                # Multi-line content with Rich markup
                f"[bold cyan]NeuroMind[/bold cyan] [dim]| CLI AI Assistant [/dim]\n"
                f"Model: [green]{model}[/green] | Thread: [yellow]{thread}[/yellow]\n"
                "Cmds: [magenta]/new, /switch, /list, /clear, /exit[/magenta]",
                border_style="cyan",  # Cyan colored border
            )
        )


    # SHOW THREAD LIST

    def show_thread_list(self, threads: List[Tuple[str, str, int]], active_name: str):
        """
        Display all threads in a formatted table.
        
        Args:
            threads: List of (name, persona, message_count) tuples
            active_name: Currently active thread name (marked with arrow)
        
        Visual Output:
        ┌─ Memory Banks ──────────────────────────┐
        │ Status │ Name         │ Persona  │ Msgs │
        ├────────┼──────────────┼──────────┼──────┤
        │   ➤   │ master       │ neuromind│  42  │
        │        │ coding_help  │ coder    │  15  │
        │        │ debug_session│ logician │   8  │
        └────────┴──────────────┴──────────┴──────┘
        
        Table Structure:
        ┌──────────────────────────────────────────────────┐
        │ Status Column (width=3)                          │
        │  • Shows "➤" for active thread                  │
        │  • Empty for inactive threads                    │
        ├──────────────────────────────────────────────────┤
        │ Name Column (cyan color)                         │
        │  • Thread name                                   │
        ├──────────────────────────────────────────────────┤
        │ Persona Column (magenta color)                   │
        │  • AI personality type                           │
        ├──────────────────────────────────────────────────┤
        │ Msgs Column (right-aligned)                      │
        │  • Message count as integer                      │
        └──────────────────────────────────────────────────┘
        """
        # Create table with title and styling
        table = Table(title="Memory Banks", border_style="dim")
        
        # Define columns with specific properties
        table.add_column("Status", width=3)           # Fixed width, no style
        table.add_column("Name", style="cyan")        # Cyan text
        table.add_column("Persona", style="magenta")  # Magenta text
        table.add_column("Msgs", justify="right")     # Right-aligned numbers

        # Add a row for each thread
        for name, persona, count in threads:
            # Show arrow (➤) for active thread, empty for others
            marker = "➤" if name == active_name else ""
            table.add_row(marker, name, persona, str(count))

        # Display the complete table
        self._console.print(table)


    # GET USER INPUT
    
    def get_user_input(self, thread_name: str) -> str:
        """
        Prompt user for input with styled prompt.
        
        Args:
            thread_name: Current thread name to display
        
        Returns:
            User's input string
        
        Visual Output:
        [master] User: █  ← Cursor here, waiting for input
        
        Prompt.ask() Features:
        • Styled prompt text
        • Input validation (optional)
        • Default values (optional)
        • Blocks until user presses Enter
        
        Example Flow:
        ┌─────────────────────────────────────┐
        │ [master] User: What is async?       │ ← User types
        └───────────────┬─────────────────────┘
                        │
                        ↓ User presses Enter
        ┌─────────────────────────────────────┐
        │ Returns: "What is async?"           │
        └─────────────────────────────────────┘
        """
        return Prompt.ask(f"\n[bold cyan][{thread_name}][/bold cyan] User")

    
    # STREAM RESPONSE SETUP
    
    def stream_response(self, thread_name: str) -> Live:
        """
        Set up a Live display for streaming AI responses.
        
        Args:
            thread_name: Current thread name
        
        Returns:
            Live object for real-time updates
        
        What is Live Display?
        ═══════════════════════════════════════════════════════════════
        Live allows updating the SAME area of the terminal repeatedly
        without creating new lines. Perfect for streaming text!
        
        Without Live (bad):
        AI: Async
        AI: Async is
        AI: Async is a
        AI: Async is a way
        (Keeps adding new lines - messy!)
        
        With Live (good):
        AI: Async█                    (frame 1)
        AI: Async is█                 (frame 2, replaces frame 1)
        AI: Async is a way█           (frame 3, replaces frame 2)
        (Updates in place - smooth!)
        ═══════════════════════════════════════════════════════════════
        
        Live Parameters Explained:
        • Markdown(""): Initial empty content (will be updated)
        • refresh_per_second=15: Update display 15 times per second
        • transient=False: Keep content after streaming ends
        • vertical_overflow="visible": Allow scrolling if too long
        
        Usage Pattern:
        ┌──────────────────────────────────────────────┐
        │ live = ui.stream_response("master")          │
        │ live.start()                                 │
        │                                              │
        │ for event in stream:                         │
        │     text += event.content                    │
        │     live.update(Markdown(text))  ← Updates!  │
        │                                              │
        │ live.stop()                                  │
        └──────────────────────────────────────────────┘
        """
        # Print the AI response header
        self._console.print(f"[bold magenta]Neuro ({thread_name}) > [/bold magenta]")
        
        # Return Live object (caller will start/stop it)
        return Live(
            Markdown(""),                    # Start with empty markdown
            refresh_per_second=15,           # Smooth updates (15 FPS)
            transient=False,                 # Keep text after done
            vertical_overflow="visible",     # Allow scrolling
        )

    
    # RENDER STREAM GROUP
  
    def render_stream_group(self, thought: str, response: str) -> Group:
        """
        Create a grouped renderable for displaying reasoning + response.
        
        Args:
            thought: AI's reasoning/thinking process (optional)
            response: AI's actual response text
        
        Returns:
            Group containing panels and markdown
        
        Visual Layout:
        ╭─── Reasoning ──────────────────────────╮
        │ Let me break this down step by step... │
        │ First, I'll explain the concept...     │
        ╰────────────────────────────────────────╯
        
        **Async** is a programming pattern that
        allows code to run concurrently without
        blocking the main thread.
        
        Group Benefits:
        • Combines multiple renderables into one
        • Updates atomically (all at once)
        • Maintains layout structure
        • Clean separation of reasoning vs content
        
        Component Breakdown:
        ┌────────────────────────────────────────────┐
        │  if thought exists:                        │
        │    ┌─────────────────────────────────┐     │
        │    │  Panel with dim border          │     │
        │    │  ├─ Title: "Reasoning"          │     │
        │    │  └─ Content: thought (italic)   │     │
        │    └─────────────────────────────────┘     │
        │                                            │
        │  if response exists:                       │
        │    ┌─────────────────────────────────┐     │
        │    │  Markdown content               │     │
        │    │  (formatted with bold, lists,   │     │
        │    │   code blocks, etc.)            │     │
        │    └─────────────────────────────────┘     │
        └────────────────────────────────────────────┘
        """
        # List to collect renderable components
        renderables: List[RenderableType] = []

        # Add reasoning panel if thought exists
        if thought:
            renderables.append(
                Panel(
                    # Render thought as markdown with italic dim style
                    Markdown(thought, style="italic dim white"),
                    title="[dim]Reasoning[/dim]",  # Dimmed title
                    border_style="dim",            # Dimmed border
                    expand=False,                  # Don't expand to full width
                )
            )

        # Add response markdown if exists
        if response:
            renderables.append(Markdown(response))

        # Combine all renderables into a group
        # *renderables unpacks the list into arguments
        return Group(*renderables)

    
    # ERROR DISPLAY

    def print_error(self, msg: str):
        """
        Display a standard error message.
        
        Args:
            msg: Error message to display
        
        Output:
        Error: Connection failed
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Red text with "Error:" prefix
        
        Example:
            ui.print_error("Could not connect to server")
        """
        self._console.print(f"[bold red]Error:[/bold red] {msg}")

    def print_critical_error(self, msg: str):
        """
        Display a critical error message (more severe).
        
        Args:
            msg: Critical error message
        
        Output:
        Critical Error: Database corrupted
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Bright red text with "Critical Error:" prefix
        
        When to use:
        • Database failures
        • Config file missing
        • Unrecoverable errors
        """
        self._console.print(f"[bold red]Critical Error:[/bold red] {msg}")

    
    # INFO DISPLAY
    
    def print_info(self, msg: str):
        """
        Display an informational message.
        
        Args:
            msg: Info message to display
        
        Output:
        Thread cleared successfully
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Green text (positive/success indication)
        
        Example:
            ui.print_info("Thread cleared successfully")
            ui.print_info("Connected to API server")
        """
        self._console.print(f"[green]{msg}[/green]")


    # CONFIRMATION PROMPT

    def confirm(self, message: str) -> bool:
        """
        Ask user for yes/no confirmation.
        
        Args:
            message: Question to ask
        
        Returns:
            True if user confirms (yes), False otherwise
        
        Visual Output:
        Clear all messages? (y/n): █
        
        User Input Options:
        • y, yes, Y, Yes → Returns True
        • n, no, N, No   → Returns False
        • Enter          → Default (usually no)
        
        Usage Example:
        ┌──────────────────────────────────────────┐
        │ if ui.confirm("Delete thread?"):         │
        │     delete_thread()                      │
        │     ui.print_info("Thread deleted")      │
        │ else:                                    │
        │     ui.print_info("Cancelled")           │
        └──────────────────────────────────────────┘
        """
        return Confirm.ask(message)

    
    # CHOICE PROMPT

    def prompt_choice(self, title: str, options: List[str], default: int = 1) -> int:
        """
        Prompt user to select from a numbered list of options.
        
        Args:
            title: Question/prompt to display
            options: List of choice strings
            default: Default choice (1-indexed, default=1)
        
        Returns:
            Index of selected option (0-indexed)
        
        Visual Output:
        Select a persona:
          1. neuromind
          2. coder
          3. teacher
          4. logician
        Choice [1]: █
        
        Data Flow:
        ┌───────────────────────────────────────────────┐
        │ Input:                                        │
        │   options = ["neuromind", "coder", "teacher"] │
        │   default = 1                                 │
        └────────────────┬──────────────────────────────┘
                         │
                         ↓ Display numbered list
        ┌────────────────────────────────────────────────┐
        │ User sees:                                     │
        │   1. neuromind                                 │
        │   2. coder                                     │
        │   3. teacher                                   │
        │ Choice [1]: _                                  │
        └────────────────┬───────────────────────────────┘
                         │
                         ↓ User enters "2"
        ┌────────────────────────────────────────────────┐
        │ Returns: 1                                     │
        │ (index 1 = "coder")                            │
        │                                                │
        │ Note: Returns 0-indexed!                       │
        │ Display is 1-indexed, return is 0-indexed      │
        └────────────────────────────────────────────────┘
        
        Why 0-indexed return?
        • Python lists are 0-indexed
        • Makes it easy to use: options[result]
        • Consistent with Python conventions
        
        Usage Example:
        ┌──────────────────────────────────────────────┐
        │ personas = ["neuromind", "coder", "teacher"] │
        │ choice = ui.prompt_choice(                   │
        │     "Select persona",                        │
        │     personas,                                │
        │     default=1                                │
        │ )                                            │
        │ selected = personas[choice]                  │
        │ print(f"You chose: {selected}")              │
        └──────────────────────────────────────────────┘
        
        Validation:
        • Only accepts numbers 1 through len(options)
        • Invalid input → prompts again
        • Enter key → uses default
        """
        # Display the title
        self._console.print(f"\n[bold]{title}:[/bold]")
        
        # Display numbered options (1-indexed for humans)
        for idx, option in enumerate(options, 1):
            self._console.print(f"  [green]{idx}.[/green] {option}")

        # Get user's choice
        choice = Prompt.ask(
            "Choice",
            # Valid choices as strings: "1", "2", "3", etc.
            choices=[str(i) for i in range(1, len(options) + 1)],
            default=str(default),  # Default choice as string
        )
        
        # Convert to 0-indexed integer
        return int(choice) - 1



# USAGE EXAMPLES

"""
Complete usage workflow:

1. Initialize UI Manager:
   ───────────────────────────────────────────────────────────────────────
   ui = UIManager()

2. Display Header:
   ───────────────────────────────────────────────────────────────────────
   ui.show_header(model="qwen3:8b", thread="master")
   # Shows app banner with current settings

3. Show Thread List:
   ───────────────────────────────────────────────────────────────────────
   threads = [
       ("master", "neuromind", 42),
       ("coding", "coder", 15),
       ("debug", "logician", 8)
   ]
   ui.show_thread_list(threads, active_name="master")
   # Displays formatted table with arrow on active thread

4. Get User Input:
   ───────────────────────────────────────────────────────────────────────
   message = ui.get_user_input("master")
   # Waits for user to type and press Enter
   # Returns: "What is async?"

5. Stream AI Response:
   ───────────────────────────────────────────────────────────────────────
   live = ui.stream_response("master")
   thought = ""
   response = ""
   
   with live:  # Starts the live display
       for event in stream_events:
           if event.type == "reasoning":
               thought += event.content
           elif event.type == "content":
               response += event.content
           
           # Update display with both reasoning and response
           live.update(ui.render_stream_group(thought, response))
   # Live display automatically stops when exiting 'with' block

6. Display Messages:
   ───────────────────────────────────────────────────────────────────────
   ui.print_info("Connection established")
   ui.print_error("Failed to send message")
   ui.print_critical_error("Database file missing")

7. Get Confirmation:
   ───────────────────────────────────────────────────────────────────────
   if ui.confirm("Clear all messages?"):
       clear_messages()
       ui.print_info("Messages cleared")

8. Get User Choice:
   ───────────────────────────────────────────────────────────────────────
   personas = ["neuromind", "coder", "teacher", "logician"]
   choice = ui.prompt_choice("Select persona", personas, default=1)
   selected_persona = personas[choice]
   ui.print_info(f"Using persona: {selected_persona}")

Rich Library Features:
═══════════════════════════════════════════════════════════════════════════
• Console: Main output handler with color/style support
• Panel: Bordered boxes around content
• Table: Formatted tables with columns and styling
• Markdown: Renders markdown with formatting
• Live: Real-time updating displays (for streaming)
• Group: Combines multiple renderables into one
• Prompt: Styled input prompts with validation

Color Markup Reference:
═══════════════════════════════════════════════════════════════════════════
[bold]text[/bold]           → Bold text
[italic]text[/italic]       → Italic text
[dim]text[/dim]             → Dimmed/faded text
[red]text[/red]             → Red text
[green]text[/green]         → Green text
[blue]text[/blue]           → Blue text
[yellow]text[/yellow]       → Yellow text
[magenta]text[/magenta]     → Magenta text
[cyan]text[/cyan]           → Cyan text
[bold red]text[/bold red]   → Bold red text (combine styles)

Why Separate UI Logic?
═══════════════════════════════════════════════════════════════════════════
✓ Single Responsibility: UI Manager only handles display
✓ Testability: Easy to test UI separately from business logic
✓ Reusability: Can use same UI for different backends
✓ Maintainability: All UI code in one place
✓ Swappability: Could replace Rich with another library easily
"""