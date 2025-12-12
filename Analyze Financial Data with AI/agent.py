"""
Multi-Agent Financial Analysis System
======================================
This module implements a supervisor-worker pattern using LangGraph to create
a team of AI agents that collaborate to answer financial queries.

Agent Architecture:
    Supervisor → coordinates the team
    ├── Price Analyst → fetches stock price data
    ├── Filing Analyst → fetches SEC filing data
    └── Synthesizer → creates final answer from gathered data

The system uses a graph-based workflow where agents pass information back
and forth until the query is fully answered.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from config import Config
from tools import fetch_sec_filing_sections, get_historical_stock_price

# AGENT PROMPTS (SYSTEM INSTRUCTIONS)

# These prompts define the behavior and personality of each agent.
# They are templates that get filled with dynamic content at runtime.

# SUPERVISOR PROMPT: The "Manager" Agent

# Role: Coordinates the team, decides which agent should work next
# Input: User's query + conversation history
# Output: JSON with {next_agent, question}
#
# Decision Tree:
#   Need stock prices? → Price Analyst
#   Need SEC filings?  → Filing Analyst
#   Have enough info?  → Synthesizer (creates final answer)

SUPERVISOR_PROMPT = """You are a supervisor managing a team of financial analyst agents to answer user queries.
Create a delegation plan to answer the user's query.

<current query>
{most_recent_query}
</current query>

<conversation_history>
{conversation_history}
</conversation_history>

<instructions>
- Price Analyst: Retrieves historical stock price data
- Filing Analyst: Retrieves SEC filing information (10-K, 10-Q, risk factors, MD&A)
- Synthesizer: Create final answer from gathered information

Decision logic:
1. If more data is needed -> Delegate a specific question to Price Analyst or Filing Analyst
2. If you have sufficient information -> Delegate to Synthesizer with: "Provide a concise final answer."

For follow-up questions: Use context from earlier in the conversation and possibly call other agents to get more information.
When calling agents, always ask them to provide the analysis for your query.
</instructions>

<formatting>
JSON object with 'next_agent' and 'question' fields.
</formatting>
"""

# WORKER PROMPT: The "Data Gatherer" Agents

# Role: Execute specific tasks (fetch data), summarize findings
# Used by: Price Analyst, Filing Analyst
# Input: Supervisor's instruction + raw tool data
# Output: Concise bullet-point summary
#
# Flow:
#   Supervisor gives instruction → Worker calls tool → Tool returns raw data
#   → Worker summarizes data → Returns to Supervisor

WORKER_PROMPT = """You are the {agent_name}. Summarize tool data to answer the supervisor's request.
Create a concise report that directly addresses the supervisor's instruction.

<supervisor_instruction>
{supervisor_instruction}
</supervisor_instruction>

<tool_data>
{tool_data}
</tool_data>

<instructions>
- Use bullet points for key findings
- Highlight important numbers and dates with bold
- Maximum 3-5 key points
- Include ONLY critical information relevant to the instruction
- No preamble or conclusions, just facts
</instructions>

Output your summary now:
"""

# SYNTHESIS PROMPT: The "Report Writer" Agent

# Role: Create final user-facing answer from all gathered information
# Input: Original user query + full conversation history
# Output: Polished, formatted answer (with BUY/HOLD/SELL for stocks)
#
# This is the LAST agent in the chain - its output goes directly to the user

SYNTHESIS_PROMPT = """You are an expert financial analyst.
Provide a concise, data-driven answer to the user's query.

<user_query>
{most_recent_user_query}
</user_query>

<conversation_history>
{conversation_history}
</conversation_history>

<instructions>
LENGTH LIMITS (strictly enforce):
- Initial company analysis: 5-8 sentences maximum
- Follow-up questions: 2-4 sentences maximum
- Data requests: Present facts concisely

Every word must add value. No fluff, no hedging, no unnecessary qualifiers.
</instructions>

<output_structure>
Choose the appropriate structure based on the query type:

TYPE 1 - Initial Company Analysis:
**Overview** (2-3 sentences)
Key business metrics, major risks or opportunities

**Price Action** (2-3 sentences)
Recent trends, key price levels, volatility

**Recommendation** (1-2 sentences)
BUY/HOLD/SELL with core reasoning based on data

TYPE 2 - Follow-up Questions:
Directly answer the question (2-4 sentences)
Reference prior context only if relevant

TYPE 3 - Data Requests:
Present key numbers/facts clearly
Add brief context if essential
</output_structure>

<formatting>
- Use markdown: **bold** for emphasis, bullets for lists
- Be direct and data-driven
- Make it scannable and easy to read
</formatting>

Write your answer now:
"""

# AGENT NAMES ENUMERATION

# Purpose: Define the names of all agents in the system
#
# Agent Team Structure:
#   ┌─────────────┐
#   │ SUPERVISOR  │ ← Orchestrates everything
#   └──────┬──────┘
#          │
#          ├──→ Price Analyst    (fetches stock prices)
#          ├──→ Filing Analyst   (fetches SEC filings)
#          └──→ Synthesizer      (creates final answer)
#
# Why StrEnum instead of regular Enum?
# - StrEnum values are strings by default
# - Easier to use in routing and conditional logic
# - Better for serialization/logging

class AgentName(StrEnum):
    """Names of all agents in the system."""
    PRICE_ANALYST = "Price Analyst"
    FILING_ANALYST = "Filing Analyst"
    SYNTHESIZER = "Synthesizer"
    SUPERVISOR = "Supervisor"
    
# STATE MANAGEMENT CLASSES

# These classes define the "memory" that flows through the agent system

# AGENT STATE: The "Shared Memory" of the System

# This state object is passed between all agents and contains:
# - All messages (conversation history)
# - Iteration counter (prevents infinite loops)
# - Next agent to execute
#
# Visual Flow of State:
#   User Input → [State] → Supervisor → [State] → Worker → [State] → Supervisor
#                  ↓                                                     ↓
#              messages: []                                         messages: [msg1, msg2]
#              iteration: 0                                         iteration: 1

@dataclass
class AgentState:
    """
    The shared state that flows through all agents.
    
    Attributes:
        messages: Conversation history (HumanMessage, AIMessage, ToolMessage)
                  The Annotated[list, add_messages] means messages are APPENDED,
                  not replaced, when state is updated
        iteration_count: How many times we've gone through the supervisor
                         (prevents infinite loops via Config.MAX_ITERATIONS)
        next_agent: Which agent should execute next (set by router function)
    """
    messages: Annotated[list, add_messages]  # add_messages = append to list
    iteration_count: int = 0
    next_agent: AgentName | None = None

# CONTEXT SCHEMA: Runtime Configuration

# Purpose: Store runtime dependencies that all nodes need access to
# Currently: Just the LLM model, but could include databases, APIs, etc.

@dataclass
class ContextSchema:
    """
    Runtime context available to all agent nodes.
    
    This is separate from AgentState because:
    - AgentState = changes during execution (messages, iteration count)
    - ContextSchema = stays constant (model, config, connections)
    """
    model: BaseChatModel  # The LLM that powers all agents (e.g., Qwen, GPT-4)

# SUPERVISOR PLAN: Structured Output from Supervisor

# Purpose: Force supervisor to output structured JSON (not free text)
# 
# Pydantic BaseModel ensures the LLM returns:
#   {
#     "next_agent": "Price Analyst",  ← Must be one of the 3 choices
#     "question": "Get NVDA prices"   ← Specific instruction
#   }
#
# Why structured output?
# - Reliable parsing (no need to extract from markdown)
# - Type safety (next_agent must be valid)
# - Easy to route (just read next_agent field)

class SupervisorPlan(BaseModel):
    """
    A structured plan for the supervisor to delegate tasks.
    
    Using Pydantic forces the LLM to output valid JSON that matches this schema.
    """

    next_agent: Literal[
        AgentName.PRICE_ANALYST, AgentName.FILING_ANALYST, AgentName.SYNTHESIZER
    ] = Field(
        description="The next agent to delegate the task to, or 'Synthesizer' if enough information is gathered."
    )
    # Literal[] means next_agent can ONLY be one of these 3 values
    # Prevents hallucinated agent names like "Data Analyst" or "Researcher"
    
    question: str = Field(
        description="A specific, focused question or instruction for the chosen agent."
    )
    # The instruction given to the next agent (e.g., "Fetch NVDA stock prices")

# HELPER FUNCTIONS

def format_history(messages: list) -> str:
    """
    Convert message list to human-readable conversation history string.
    
    Message types and formatting:
        HumanMessage → "User: <content>" or "SupervisorInstruction: <content>"
        AIMessage → "Assistant: <content>"
        ToolMessage → (skipped - too verbose for LLM context)
    
    Example output:
        User: What's NVDA's stock price?
        Assistant: Let me check that for you.
        User: Also check their SEC filings.
        Assistant: Here's what I found...
    
    Why format history?
    - LLMs understand natural conversation format better than raw message objects
    - Removes noise (tool outputs are redundant with summaries)
    - Keeps context concise for better reasoning
    """
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            # Check if message has a "name" attribute (like "SupervisorInstruction")
            role = "User" if not hasattr(msg, "name") else msg.name
            formatted.append(f"{role}: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
        elif isinstance(msg, ToolMessage):
            # Skip tool messages - they're raw data, not conversational
            continue
    return "\n".join(formatted)

# AGENT NODE FUNCTIONS (THE ACTUAL AGENTS)

# Each function represents one node in the execution graph.
# Nodes take state, process it, and return updated state.

# SUPERVISOR NODE: The Orchestrator

# Flow:
#   1. Extract user's most recent query
#   2. Format conversation history
#   3. Ask LLM: "What should we do next?"
#   4. LLM returns structured plan: {next_agent, question}
#   5. Add supervisor's instruction to messages
#   6. Return updated state with next_agent set
#
# This node is the "brain" - it decides the workflow

def supervisor_node(state: AgentState, runtime: Runtime[ContextSchema]):
    """
    The supervisor agent that coordinates all other agents.
    
    Input: Current state with all messages
    Output: Updated state with:
        - New message (supervisor's instruction to next agent)
        - next_agent (which agent should execute)
        - iteration_count (incremented)
    
    Process:
        User Query → Supervisor → Decides → Delegates to Worker/Synthesizer
    """
    
    # STEP 1: Configure LLM to Output Structured Data

    # .with_structured_output() forces LLM to return SupervisorPlan object
    # Instead of free text, we get: {"next_agent": "...", "question": "..."}

    supervisor_llm = runtime.context.model.with_structured_output(SupervisorPlan)

    # STEP 2: Extract User's Most Recent Query

    # Filter for HumanMessages that don't have a "name" attribute
    # (messages with name="SupervisorInstruction" are internal, not from user)

    user_messages = [
        msg
        for msg in state.messages
        if isinstance(msg, HumanMessage) and not hasattr(msg, "name")
    ]
    most_recent_query = user_messages[-1].content if user_messages else ""

    # STEP 3: Format Conversation History

    conversation_history = format_history(state.messages)

    # STEP 4: Ask Supervisor LLM to Create a Plan

    # LLM receives the prompt with query + history filled in
    # Returns: SupervisorPlan(next_agent="Price Analyst", question="...")

    plan = supervisor_llm.invoke(
        SUPERVISOR_PROMPT.format(
            most_recent_query=most_recent_query,
            conversation_history=conversation_history,
        )
    )

    # STEP 5: Create Message with Supervisor's Instruction

    # name="SupervisorInstruction" distinguishes this from user messages
    # The next agent will receive this as their task

    new_message = HumanMessage(content=plan.question, name="SupervisorInstruction")
    
    # STEP 6: Return Updated State

    # The returned dict updates the state:
    # - messages: appends new_message (due to add_messages annotation)
    # - next_agent: tells router which node to execute next
    # - iteration_count: prevents infinite loops

    return {
        "messages": [new_message],
        "next_agent": plan.next_agent,
        "iteration_count": state.iteration_count + 1,
    }

# WORKER NODE FACTORY: Creates Price Analyst and Filing Analyst

# Why a factory function?
# - Price Analyst and Filing Analyst have identical logic
# - Only difference: which tools they have access to
# - Factory avoids code duplication
#
# Flow:
#   1. Receive supervisor's instruction
#   2. Call appropriate tool (stock prices or SEC filing)
#   3. Summarize tool output
#   4. Return to supervisor

def create_worker_node(agent_name: AgentName, tools: list):
    """
    Factory function that creates a worker agent node.
    
    This is a closure that captures agent_name and tools, then returns
    a function that can be used as a graph node.
    
    Args:
        agent_name: Display name for this worker (e.g., "Price Analyst")
        tools: List of LangChain tools this worker can use
    
    Returns:
        worker_node function configured with the given tools
    
    Usage:
        price_node = create_worker_node("Price Analyst", [get_stock_price])
        filing_node = create_worker_node("Filing Analyst", [get_sec_filing])
    """
    
    def worker_node(state: AgentState, runtime: Runtime[ContextSchema]):
        """
        A worker agent that executes tools and summarizes results.
        
        Process Flow:
        ┌─────────────────────────────────────────────────────────────┐
        │ 1. Get supervisor's instruction from last message           │
        │ 2. Call LLM to decide which tool to use                     │
        │ 3. Execute the tool (fetch stock data or SEC filing)        │
        │ 4. Ask LLM to summarize tool output                         │
        │ 5. Return summary to supervisor                             │
        └─────────────────────────────────────────────────────────────┘
        """
        
        # STEP 1: Bind Tools to LLM

        # .bind_tools() makes the LLM aware of available tools
        # LLM can then decide to call these tools based on the instruction

        agent = runtime.context.model.bind_tools(tools)
        
        # STEP 2: Get Supervisor's Instruction

        # Last message is always the supervisor's instruction
        # (e.g., "Get NVDA's stock prices for last 90 days")

        supervisor_instruction = state.messages[-1]
        
        # STEP 3: Let LLM Decide Which Tool to Call

        # LLM reads the instruction and decides if it needs to call a tool
        # If yes: response.tool_calls contains [{name, args, id}, ...]
        # If no: response.tool_calls is empty (LLM answered directly)

        response = agent.invoke([supervisor_instruction])

        # STEP 4: Handle Case Where No Tool Call is Needed

        # Sometimes the instruction is already answered by LLM's knowledge
        # In that case, just return the response and go back to supervisor

        if not response.tool_calls:
            return {"messages": [response], "next_agent": AgentName.SUPERVISOR}
        
        # STEP 5: Execute All Requested Tools

        # tool_calls structure: [
        #   {
        #     "name": "get_historical_stock_price",
        #     "args": {"ticker": "NVDA"},
        #     "id": "call_abc123"
        #   }
        # ]

        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]

            # Find the actual tool function by name
            selected_tool = next((t for t in tools if t.name == tool_name), None)

            # Execute tool with the provided arguments
            # e.g., get_historical_stock_price(ticker="NVDA")
            tool_output = selected_tool.invoke(tool_call["args"])
            
            # Wrap output in ToolMessage (required for LangChain)
            tool_messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
            )

        # STEP 6: Summarize Tool Output

        # Tool output is often verbose (XML with lots of data)
        # We ask LLM to extract and summarize the key points
        # This makes it easier for the supervisor to understand

        summary_response = runtime.context.model.invoke(
            WORKER_PROMPT.format(
                agent_name=agent_name,
                supervisor_instruction=supervisor_instruction.content,
                tool_data=tool_messages[0].content,  # Raw tool output
                ),
        )

        # STEP 7: Return Summary and Route Back to Supervisor

        # Worker's job is done - send summary to supervisor for next decision

        return {
            "messages": [summary_response],
            "next_agent": AgentName.SUPERVISOR,
        }
    
    return worker_node

# SYNTHESIZER NODE: The Final Answer Generator

# Role: Take ALL gathered information and create polished user-facing answer
# When: Called by supervisor when enough information is collected
# Output: Goes directly to user (END of graph)
#
# This is the LAST agent in the workflow
        
def synthesizer_node(state: AgentState, runtime: Runtime[ContextSchema]):
    """
    The synthesizer agent that creates the final answer for the user.
    
    This agent:
    1. Reviews the entire conversation history
    2. Identifies the original user query
    3. Synthesizes all gathered data into a coherent answer
    4. Formats it professionally (markdown, bullet points, etc.)
    
    This is the TERMINAL node - after this, the graph ends.
    """
    
    # STEP 1: Extract Original User Query

    # Need to answer the user's question, not supervisor's instructions
    # Filter for actual user messages (no name attribute)

    user_messages = [
        msg
        for msg in state.messages
        if isinstance(msg, HumanMessage) and not hasattr(msg, "name")
    ]

    most_recent_user_query = user_messages[-1].content if user_messages else ""

    # STEP 2: Format Full Conversation History

    # Synthesizer needs ALL context:
    # - User's question
    # - Supervisor's delegations
    # - Workers' findings

    conversation_history = format_history(state.messages)

    # STEP 3: Generate Final Answer

    # LLM reads everything and creates a polished, formatted answer
    # Follows the SYNTHESIS_PROMPT structure (Overview, Price Action, etc.)

    response = runtime.context.model.invoke(
        SYNTHESIS_PROMPT.format(
            most_recent_user_query = most_recent_user_query,
            conversation_history = conversation_history,
        )
    )

    # STEP 4: Return Final Answer

    # No next_agent specified = graph will use router to determine next step
    # Router will see we're at Synthesizer and route to END

    return {"messages": [response]}

# ROUTER FUNCTION: Traffic Controller

# Purpose: Decide which node to execute next (or if we should stop)
#
# Decision Logic:
#   1. If iteration_count >= MAX_ITERATIONS → END (prevent infinite loops)
#   2. Otherwise → state.next_agent (set by supervisor or workers)
#
# This is called after EVERY node execution

def router(state: AgentState):
    """
    Router function that decides which agent executes next.
    
    Safety Check:
        If we've iterated too many times (Config.MAX_ITERATIONS),
        force stop to prevent infinite loops.
    
    Otherwise:
        Route to whatever agent is specified in state.next_agent
        (set by supervisor_node or worker_nodes)
    
    Returns:
        - END (special constant) to stop execution
        - AgentName to route to that agent's node
    """

    # Safety: Prevent Infinite Loops

    # If supervisor keeps delegating without synthesizing, we'd loop forever
    # MAX_ITERATIONS (from config.py) provides a hard stop

    if state.iteration_count >= Config.MAX_ITERATIONS:
        return END  # END is a special LangGraph constant that stops execution
    
    # Normal Routing: Follow state.next_agent

    # state.next_agent was set by the previous node
    # Could be: PRICE_ANALYST, FILING_ANALYST, SYNTHESIZER, or SUPERVISOR
    return state.next_agent

# GRAPH BUILDER: Assembles the Agent System

# This function creates the actual execution graph that defines:
# - Which agents exist (nodes)
# - How they connect (edges)
# - Where to start (entry point)
# - How to route between them (conditional edges)

def create_agent():
    """
    Creates and compiles the agent execution graph.
    
    Graph Structure:
    ┌───────────────────────────────────────────────────────────────────┐
    │                                                                   │
    │  START → SUPERVISOR → (router decides) → [Worker or Synthesizer]  │
    │             ↑                                     ↓               │
    │             └─────────── Workers return here ────┘                │
    │                                                                   │
    │  Synthesizer → END                                                │
    └───────────────────────────────────────────────────────────────────┘
    
    Execution Flow Example:
        User: "Analyze NVDA stock"
        ↓
        Supervisor: "Need prices and filings"
        ↓
        Router: Routes to Price Analyst
        ↓
        Price Analyst: Fetches data, summarizes
        ↓
        Router: Back to Supervisor
        ↓
        Supervisor: "Now need filings"
        ↓
        Router: Routes to Filing Analyst
        ↓
        Filing Analyst: Fetches data, summarizes
        ↓
        Router: Back to Supervisor
        ↓
        Supervisor: "Have enough info, synthesize"
        ↓
        Router: Routes to Synthesizer
        ↓
        Synthesizer: Creates final answer
        ↓
        Router: Routes to END
        ↓
        User receives final answer
    
    Returns:
        Compiled LangGraph that can be executed with .invoke()
    """

    # STEP 1: Create Worker Nodes Using Factory Function

    # Each worker gets specific tools:
    # - Price Analyst → [get_historical_stock_price]
    # - Filing Analyst → [fetch_sec_filing_sections]

    price_agent_node = create_worker_node(
        AgentName.PRICE_ANALYST, 
        [get_historical_stock_price]  # Only this tool available
    )
    
    filing_agent_node = create_worker_node(
        AgentName.FILING_ANALYST, 
        [fetch_sec_filing_sections]  # Only this tool available
    )

    # STEP 2: Initialize the State Graph

    # StateGraph is LangGraph's main class for building agent workflows
    # 
    # Type parameters explained:
    # StateGraph[AgentState, ContextSchema, AgentState, AgentState]
    #            ↑          ↑              ↑            ↑
    #            │          │              │            └─ Output type
    #            │          │              └─ Update type (partial state updates)
    #            │          └─ Context type (runtime dependencies)
    #            └─ State type (the shared memory)

    graph = StateGraph[AgentState, ContextSchema, AgentState, AgentState](
        AgentState,      # The state schema
        ContextSchema    # The context schema (model config)
    )

    # STEP 3: Add All Nodes to the Graph

    # Each node is a function that:
    # - Takes (state, runtime) as input
    # - Returns updated state as output
    # 
    # Node names (strings) are used for routing and visualization

    graph.add_node(AgentName.SUPERVISOR, supervisor_node)
    graph.add_node(AgentName.PRICE_ANALYST, price_agent_node)
    graph.add_node(AgentName.FILING_ANALYST, filing_agent_node)
    graph.add_node(AgentName.SYNTHESIZER, synthesizer_node)

    # STEP 4: Set Entry Point

    # When graph.invoke() is called, execution starts here
    # Supervisor always goes first to analyze the user's query

    graph.set_entry_point(AgentName.SUPERVISOR)

    # STEP 5: Add Conditional Edges from Supervisor

    # After Supervisor runs, the router function decides where to go next
    # 
    # Routing logic:
    #   router(state) returns one of:
    #   - AgentName.PRICE_ANALYST → go to Price Analyst node
    #   - AgentName.FILING_ANALYST → go to Filing Analyst node
    #   - AgentName.SYNTHESIZER → go to Synthesizer node
    #   - END → stop execution
    # 
    # The dictionary maps router's return value to actual node names

    graph.add_conditional_edges(
        AgentName.SUPERVISOR,  # Source node
        router,                # Function that decides next node
        {
            # Mapping: router return value → target node
            AgentName.PRICE_ANALYST: AgentName.PRICE_ANALYST,
            AgentName.FILING_ANALYST: AgentName.FILING_ANALYST,
            AgentName.SYNTHESIZER: AgentName.SYNTHESIZER,
            # If router returns END, execution stops (no mapping needed)
        },
    )

    # STEP 6: Add Fixed Edges from Workers Back to Supervisor

    # Workers always return to Supervisor (no conditional logic)
    # Supervisor then decides if more workers are needed or if ready to synthesize
    # 
    # Flow:
    #   Price Analyst → (always) → Supervisor
    #   Filing Analyst → (always) → Supervisor

    graph.add_edge(AgentName.PRICE_ANALYST, AgentName.SUPERVISOR)
    graph.add_edge(AgentName.FILING_ANALYST, AgentName.SUPERVISOR)

    # STEP 7: Add Fixed Edge from Synthesizer to END

    # Synthesizer creates the final answer - nothing comes after it
    # Execution stops here

    graph.add_edge(AgentName.SYNTHESIZER, END)

    # STEP 8: Compile the Graph

    # .compile() validates the graph and creates an executable object
    # 
    # Compilation checks:
    # - All referenced nodes exist
    # - No orphaned nodes (unreachable from entry point)
    # - No invalid edge configurations
    # 
    # Returns: CompiledGraph that can be invoked with:
    #   graph.invoke(initial_state, context=...)

    return graph.compile()

# USAGE EXAMPLES

"""
How to use this agent system:

# ───────────────────────────────────────────────────────────────────────────
# Example 1: Basic Usage
# ───────────────────────────────────────────────────────────────────────────
from agent import create_agent, AgentState, ContextSchema
from langchain_ollama import ChatOllama
from config import Config

# Initialize the model
model = ChatOllama(model=Config.MODEL.name, temperature=Config.MODEL.temperature)

# Create the agent graph
agent_graph = create_agent()

# Prepare initial state
initial_state = AgentState(
    messages=[HumanMessage(content="Analyze NVIDIA stock (NVDA)")],
    iteration_count=0,
    next_agent=None
)

# Prepare context
context = ContextSchema(model=model)

# Run the agent system
result = agent_graph.invoke(initial_state, context=context)

# Get final answer
final_answer = result["messages"][-1].content
print(final_answer)

# ───────────────────────────────────────────────────────────────────────────
# Example 2: Multi-Turn Conversation
# ───────────────────────────────────────────────────────────────────────────

# First query
state1 = AgentState(
    messages=[HumanMessage(content="What's NVDA's recent price trend?")],
    iteration_count=0
)
result1 = agent_graph.invoke(state1, context=context)

# Follow-up query (uses conversation history)
state2 = AgentState(
    messages=result1["messages"] + [HumanMessage(content="What are their main risks?")],
    iteration_count=0
)
result2 = agent_graph.invoke(state2, context=context)

# ───────────────────────────────────────────────────────────────────────────
# Example 3: Understanding the Execution Flow
# ───────────────────────────────────────────────────────────────────────────

Visual representation of what happens internally:

Query: "Analyze NVDA stock"

┌─────────────────────────────────────────────────────────────────┐
│ Iteration 1                                                     │
│ ───────────                                                     │
│ Supervisor: "Need stock prices. Delegate to Price Analyst."     │
│ Router: Routes to PRICE_ANALYST                                 │
│ Price Analyst:                                                  │
│   - Calls get_historical_stock_price("NVDA")                    │
│   - Gets XML with weekly prices                                 │
│   - Summarizes: "NVDA up 35% in 90 days, $480→$650"             │
│ Router: Routes to SUPERVISOR                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Iteration 2                                                     │
│ ───────────                                                     │
│ Supervisor: "Need SEC filings. Delegate to Filing Analyst."     │
│ Router: Routes to FILING_ANALYST                                │
│ Filing Analyst:                                                 │
│   - Calls fetch_sec_filing_sections("NVDA", [MDA, RISK])        │
│   - Gets MD&A and Risk Factors from 10-Q                        │
│   - Summarizes: "Key risks: supply chain, competition"          │
│ Router: Routes to SUPERVISOR                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Iteration 3                                                     │
│ ───────────                                                     │
│ Supervisor: "Have all info. Delegate to Synthesizer."           │
│ Router: Routes to SYNTHESIZER                                   │
│ Synthesizer:                                                    │
│   - Reviews all gathered data                                   │
│   - Creates formatted final answer:                             │
│                                                                 │
│     **Overview**                                                │
│     NVIDIA (NVDA) is a leading AI chip maker...                 │
│                                                                 │
│     **Price Action**                                            │
│     Strong uptrend, +35% in 90 days...                          │
│                                                                 │
│     **Recommendation**                                          │
│     BUY - Strong fundamentals, positive momentum                │
│                                                                 │
│ Router: Routes to END                                           │
└─────────────────────────────────────────────────────────────────┘

Final output delivered to user!

# ───────────────────────────────────────────────────────────────────────────
# Example 4: Error Handling and Edge Cases
# ───────────────────────────────────────────────────────────────────────────

# Case 1: Invalid ticker
state = AgentState(
    messages=[HumanMessage(content="Analyze XYZ123 stock")],
    iteration_count=0
)
result = agent_graph.invoke(state, context=context)
# Tools will return error messages, agents will handle gracefully

# Case 2: Max iterations reached
# If supervisor keeps delegating without synthesizing, router will
# stop execution at Config.MAX_ITERATIONS to prevent infinite loop

# Case 3: Follow-up without context
state = AgentState(
    messages=[HumanMessage(content="What about their competitors?")],
    iteration_count=0
)
# Supervisor will ask for clarification or make best effort with context
"""

# KEY DESIGN PATTERNS EXPLAINED

"""
1. SUPERVISOR-WORKER PATTERN
   ─────────────────────────
   Why: Separates coordination (supervisor) from execution (workers)
   Benefit: Easy to add new workers without changing core logic
   
   Alternative: Could have single agent do everything, but that's less modular

2. STRUCTURED OUTPUT (Pydantic)
   ────────────────────────────
   Why: Forces LLM to return parseable JSON instead of free text
   Benefit: No regex parsing, type safety, predictable routing
   
   Example: SupervisorPlan ensures next_agent is always valid

3. FACTORY PATTERN (create_worker_node)
   ────────────────────────────────────
   Why: Avoid code duplication for similar agents
   Benefit: Workers have identical logic, only tools differ
   
   Without factory: Would need separate price_analyst_node and 
                    filing_analyst_node functions with duplicated code

4. STATE FLOW (add_messages annotation)
   ──────────────────────────────────────
   Why: Messages accumulate rather than replace
   Benefit: Full conversation history preserved automatically
   
   Technical: Annotated[list, add_messages] tells LangGraph to append

5. GRAPH-BASED ROUTING
   ────────────────────
   Why: Explicit visual representation of agent workflow
   Benefit: Easy to understand, debug, and modify agent interactions
   
   Alternative: Nested if/else statements (much harder to understand)

6. CONTEXT SEPARATION
   ──────────────────
   Why: AgentState vs ContextSchema separation
   Benefit: State changes during execution, context stays constant
   
   State: messages, iteration_count (dynamic)
   Context: model, config (static)
"""

# DEBUGGING TIPS

"""
1. Trace message flow:
   for msg in result["messages"]:
       print(f"{type(msg).__name__}: {msg.content[:100]}...")

2. Check iteration count:
   print(f"Iterations: {result['iteration_count']}")

3. Visualize the graph:
   from langgraph.graph import Graph
   graph.get_graph().draw_mermaid()  # Generates diagram

4. Add logging to nodes:
   def supervisor_node(state, runtime):
       print(f"[SUPERVISOR] Query: {state.messages[-1].content}")
       # ... rest of code

5. Test individual components:
   # Test worker node separately
   test_state = AgentState(
       messages=[HumanMessage(content="Get NVDA prices", name="SupervisorInstruction")],
       iteration_count=0
   )
   result = price_agent_node(test_state, runtime)
"""