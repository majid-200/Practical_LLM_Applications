"""
═══════════════════════════════════════════════════════════════════════════════
    LONG-RUNNING AI CODING AGENT
═══════════════════════════════════════════════════════════════════════════════

This system creates an AI agent that:
1. Takes a project request (e.g., "build a calculator")
2. Breaks it into features
3. Implements each feature with tests
4. Commits working code to git

FLOW DIAGRAM:
┌─────────────────┐
│  User Request   │  "Create factorial and fibonacci functions"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  INITIALIZER    │  Breaks request into features
│     Agent       │  Creates feature_list.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CODING Agent   │  ◄─┐ Loops until all features done
│                 │    │
│  1. Pick next   │    │
│     feature     │    │
│  2. Write code  │    │
│  3. Run tests   │    │
│  4. Commit if   │    │
│     passing     │────┘
└─────────────────┘

"""

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

MODEL_NAME = "qwen3:8b"  # The LLM model used for AI reasoning
WORK_DIR = Path("./agent_workspace")  # Where all code files will be created

# Initialize the LLM (Large Language Model) that powers the agent
llm = init_chat_model(MODEL_NAME, model_provider="ollama", reasoning=False)

# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS - Define the structure of our data
# ═══════════════════════════════════════════════════════════════════════════

class Feature(BaseModel):
    """
    Represents a single feature to be implemented.
    
    Example:
        {
            "name": "factorial",
            "description": "Calculate factorial of a number",
            "implementation_hints": ["Use recursion", "Handle base case"],
            "test_file": "test_factorial.py",
            "passes": False
        }
    """
    name: str
    description: str
    implementation_hints: list[str] = Field(
        description="Step-by-step hints for implementing this feature"
    )
    test_file: str = Field(description="Name of the test file to create")
    passes: bool = False  # Tracks if tests have passed

class ProjectState(BaseModel):
    """
    Container for all features in the project.
    
    Visual Structure:
        ProjectState
        └── features: [Feature, Feature, Feature, ...]
    """
    features: list[Feature]

class CodingAction(BaseModel):
    """
    Represents a single coding action the agent will take.
    
    Contains:
    - The code files to write
    - The command to run tests
    - A git commit message
    
    Example:
        {
            "code_files": {
                "factorial.py": "def factorial(n): ...",
                "test_factorial.py": "import unittest ..."
            },
            "test_command": "python -m pytest test_factorial.py -v",
            "commit_message": "Implement factorial function"
        }
    """
    code_files: dict[str, str] = Field(description="filename -> content mapping")
    test_command: str = Field(description="Command to run tests")
    commit_message: str

# ═══════════════════════════════════════════════════════════════════════════
#  SHELL RESULT - Captures output from running commands
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ShellResult:
    """
    Stores the result of running a shell command.
    
    Diagram:
        ┌──────────────────┐
        │  Shell Command   │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │   ShellResult    │
        ├──────────────────┤
        │ stdout: "..."    │ ← Standard output
        │ stderr: "..."    │ ← Error messages
        │ returncode: 0    │ ← 0 = success, non-zero = failure
        └──────────────────┘
    """
    stdout: str      # Normal output
    stderr: str      # Error output
    returncode: int  # Exit code (0 = success)

    @property
    def success(self) -> bool:
        """Returns True if command succeeded (returncode 0)"""
        return self.returncode == 0
    
    def __str__(self) -> str:
        """Pretty prints the output for debugging"""
        out = []
        if self.stdout.strip():
            out.append(f"STDOUT:\n{self.stdout}")
        if self.stderr.strip():
            out.append(f"STDERR:\n{self.stderr}")
        return "\n".join(out) if out else "(no output)"
    
# ═══════════════════════════════════════════════════════════════════════════
#  AGENT ENVIRONMENT - The workspace where the agent operates
# ═══════════════════════════════════════════════════════════════════════════

class AgentEnvironment:
    """
    Provides tools for the agent to interact with the file system and run commands.
    
    Think of this as the agent's "hands" - it can:
    - Write files
    - Read files
    - Run shell commands
    - Commit to git
    
    File Structure Created:
        agent_workspace/
        ├── feature_list.json      ← Tracks all features and their status
        ├── agent_progress.txt     ← Human-readable progress log
        ├── factorial.py           ← Implementation files
        ├── test_factorial.py      ← Test files
        ├── fibonacci.py
        ├── test_fibonacci.py
        └── .git/                  ← Git repository
    """
    
    def __init__(self, work_dir: Path):
        """
        Initialize the agent's workspace.
        
        Args:
            work_dir: Path where all files will be created
        """
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True) # If folder exists, don't raise an error
        self.python = sys.executable  # Path to Python interpreter

    def run_shell(self, command: str, timeout: int = 60) -> ShellResult:
        """
        Execute a shell command in the workspace.
        
        Flow:
            command → subprocess → ShellResult
        
        Args:
            command: Shell command to run (e.g., "pytest test.py")
            timeout: Max seconds to wait before killing command
            
        Returns:
            ShellResult with stdout, stderr, and return code
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.work_dir,        # Run in agent's workspace
                capture_output=True,       # Capture stdout and stderr
                text=True,                 # Return as strings, not bytes
                timeout=timeout,
            )
            return ShellResult(result.stdout, result.stderr, result.returncode)
        except subprocess.TimeoutExpired:
            return ShellResult("", "Command timed out", 1)
        except Exception as e:
            return ShellResult("", str(e), 1)
        
    def write_file(self, filename: str, content: str):
        """
        Write content to a file in the workspace.
        Creates parent directories if needed.
        
        Example:
            env.write_file("factorial.py", "def factorial(n): return 1")
        """
        file_path = self.work_dir / filename
        file_path.parent.mkdir(exist_ok=True, parents=True)
        file_path.write_text(content)

    def read_file(self, filename: str) -> str:
        """
        Read a file from the workspace.
        Returns empty string if file doesn't exist.
        """
        path = self.work_dir / filename
        return path.read_text() if path.exists() else ""
    
    def list_files(self) -> list[str]:
        """
        List all Python files in the workspace.
        
        Returns:
            List of relative paths to .py files
        """
        return [str(p.relative_to(self.work_dir)) for p in self.work_dir.rglob("*.py")]
    
    def git_commit(self, message: str):
        """
        Stage all changes and commit to git.
        
        Flow:
            git add . → git commit -m "message"
        """
        self.run_shell("git add .")
        self.run_shell(f'git commit -m "{message}"')

    def get_context(self) -> str:
        """
        Gather context about the current state of the workspace.
        This is sent to the LLM so it knows what's already been done.
        
        Returns formatted string with:
        - Python interpreter path
        - List of existing files
        - Feature status
        - Git commit history
        """
        git_log = self.run_shell("git log --oneline -5")
        files = self.list_files()
        features = self.read_file("feature_list.json")

        return f"""
=== PYTHON PATH ===
{self.python}

=== EXISTING FILES ===
{files}

=== FEATURES STATUS ===
{features}

=== GIT HISTORY ===
{git_log}
"""
    
# ═══════════════════════════════════════════════════════════════════════════
#  INITIALIZER AGENT - Breaks down the project into features
# ═══════════════════════════════════════════════════════════════════════════

def run_initializer(request: str, env: AgentEnvironment):
    """
    The INITIALIZER takes a user request and creates a project plan.
    
    Process:
        1. User Request → LLM
        2. LLM generates list of features
        3. Save features to feature_list.json
        4. Initialize git repository
    
    Example:
        Input:  "Create factorial and fibonacci functions"
        Output: feature_list.json with 2 features:
                - factorial (with test_factorial.py)
                - fibonacci (with test_fibonacci.py)
    
    Args:
        request: User's project description
        env: Agent environment to work in
    """
    print(f"\n[Initializer] setting up project: {request}")

    # Create a prompt template for the LLM
    # This tells the AI how to break down the request
    prompt = ChatPromptTemplate.from_template("""
You are a project architect. Create a feature list for this request.
                                              
RULES:
- Each feature is a python module to implement
- Each feature needs a corresponding test file
- Features should be simple and focused (one function per feature)
- Do NOT include setup tasks like "install python" - only code features

User Request: {request}

Example features for "calculator with add multiply":
- name: "add", description: "Add two numbers", test_file:"test_add.py"
- name: "multiply", description: "Multiply two numbers", test_file: "test_multiply.py"                                   
""")
    
    # Chain: prompt → LLM → structured output (ProjectState)
    chain = prompt | llm.with_structured_output(ProjectState)
    project_plan = chain.invoke({"request": request})

    # Save the project plan as JSON
    env.write_file("feature_list.json", project_plan.model_dump_json(indent=2))
    env.write_file("agent_progress.txt", "Project initialized.\n")

    # Initialize git repository
    env.run_shell("git init")
    env.git_commit("Initial commit")

    # Print summary of created features
    print(f"[OK] Created {len(project_plan.features)} features.")
    for f in project_plan.features:
        print(f"  - {f.name}: {f.description}")


# ═══════════════════════════════════════════════════════════════════════════
#  CODING AGENT - Implements one feature at a time
# ═══════════════════════════════════════════════════════════════════════════

def run_coding_session(env: AgentEnvironment, max_retries: int = 3) -> bool:
    """
    The CODING AGENT implements one feature per session.
    
    Algorithm:
        ┌─────────────────────────────────┐
        │ 1. Load feature_list.json       │
        │ 2. Find next incomplete feature │
        │ 3. FOR attempt in 1..max_retries│
        │    ├─ Ask LLM to write code     │
        │    ├─ Write files to disk       │
        │    ├─ Run tests                 │
        │    └─ IF tests pass:            │
        │       ├─ Mark feature complete  │
        │       ├─ Git commit             │
        │       └─ RETURN True            │
        │ 4. IF all attempts fail:        │
        │    └─ RETURN True (continue)    │
        └─────────────────────────────────┘
    
    Args:
        env: Agent environment
        max_retries: How many times to try if tests fail
        
    Returns:
        True: Continue to next feature
        False: All features complete, stop
    """
    print("\n[Coding Agent] Starting session...")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: Load feature list
    # ─────────────────────────────────────────────────────────────────────
    try:
        features_data = json.loads(env.read_file("feature_list.json"))
        features = features_data.get("features", [])
    except Exception as e:
        print(f"[ERROR] Reading features: {e}")
        return False
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: Find next incomplete feature
    # ─────────────────────────────────────────────────────────────────────
    next_feature = next((f for f in features if not f["passes"]), None)
    if not next_feature:
        print("[DONE] All features completed!")
        return False  # Signal to stop the main loop
    
    print(f"Working on: {next_feature['name']}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: Attempt to implement feature (with retries)
    # ─────────────────────────────────────────────────────────────────────
    for attempt in range(1, max_retries + 1):
        print(f"  Attempt {attempt}/{max_retries}")
        
        # Ask LLM to write code for this feature
        prompt = ChatPromptTemplate.from_template("""
You are a Python developer. Implement this feature.
                                                  
CONTEXT:
{context}
                                                
TASK: {feature_name}
Description: {feature_desc}
Test file: {test_file}
                                                  
RULES:
1. Create the Implementation file (e.g., {feature_name}.py)
2. Create the test file ({test_file}) using unittest
3. Tests must be runnable with: {python_path} -m pytest {test_file} -v
4. Keep code simple and correct
                                                  
IMPORTANT
- Use '{python_path}' as the Python interpreter
- Make sure imports work (use relative imports or put files in same directory)
""")
        
        # Chain: prompt → LLM → CodingAction (files + test command)
        chain = prompt | llm.with_structured_output(CodingAction)
        action = chain.invoke(
            {
                "context": env.get_context(),  # Current state of workspace
                "feature_name": next_feature["name"],
                "feature_desc": next_feature["description"],
                "test_file": next_feature["test_file"],
                "python_path": env.python,
            }
        )

        # Write all code files to disk
        for filename, content in action.code_files.items():
            env.write_file(filename, content)
            print(f"  wrote: {filename}")

        # Run the tests
        print(f"  Running: {action.test_command}")
        result = env.run_shell(action.test_command)
        print(f"  {result}")

        # ─────────────────────────────────────────────────────────────────
        # Check if tests passed
        # ─────────────────────────────────────────────────────────────────
        if result.success:
            # Mark feature as complete
            for f in features:
                if f["name"] == next_feature["name"]:
                    f["passes"] = True
                    break

            # Save updated feature list
            features_data["features"] = features
            env.write_file("feature_list.json", json.dumps(features_data, indent=2))

            # Update progress log
            progress = env.read_file("agent_progress.txt")
            env.write_file(
                "agent_progress.txt", progress + f"[x] {next_feature['name']}\n"
            )

            # Commit to git
            env.git_commit(action.commit_message)
            print("  [OK] Tests passed, committed.")
            return True  # Success! Move to next feature
        else:
            print("  [FAIL] Tests failed.")
            # Loop continues to next attempt

    # ─────────────────────────────────────────────────────────────────────
    # All attempts failed
    # ─────────────────────────────────────────────────────────────────────
    print(f"[ERROR] Failed after {max_retries} attempts.")
    return True  # Continue to next feature anyway


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION - Orchestrates the entire agent system
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Main execution flow:
    
    ┌──────────────────────────────────────────────────────┐
    │                    START                             │
    └──────────────┬───────────────────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────────────────┐
    │  1. Setup: Clean workspace and create environment    │
    └──────────────┬───────────────────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────────────────┐
    │  2. Initialize: Break request into features          │
    └──────────────┬───────────────────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────────────────┐
    │  3. Loop: Run coding sessions (max 3 iterations)     │
    │     Each session implements one feature              │
    │     Stop if all features complete                    │
    └──────────────┬───────────────────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────────────────┐
    │  4. Summary: Print completion statistics             │
    └──────────────────────────────────────────────────────┘
    """
    
    # Initialize the agent environment
    env = AgentEnvironment(WORK_DIR)

    # Clean slate: Remove old workspace if it exists
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir()

    # Define what we want the agent to build
    user_request = (
        "Create Python functions for factorial and fibonacci. Include unit tests."
    )
    
    # PHASE 1: Break down the request into features
    run_initializer(user_request, env)

    # PHASE 2: Implement features one by one (max 3 sessions)
    for _ in range(3):
        if not run_coding_session(env):
            break  # All features complete!

    # PHASE 3: Print final summary
    print("\n[Summary]")
    features = json.loads(env.read_file("feature_list.json")).get("features", [])
    passed = sum(1 for f in features if f["passes"])
    print(f"  Completed: {passed}/{len(features)} features")
    