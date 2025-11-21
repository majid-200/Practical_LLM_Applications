# Project: LangChain Conceptual Tutorials

This project contains a series of Jupyter notebooks that follow the official LangChain documentation. The notebooks are adapted to use a fully local, open-source stack powered by **Ollama**.

They showcase a progression from basic LLM interactions to complex, stateful agents and Retrieval-Augmented Generation (RAG) systems.

## 🚀 Key Technologies

- **[LangChain](https://www.langchain.com/):** The core framework for orchestrating LLM workflows, from simple chains to complex agents.
- **[Ollama](https://ollama.com/):** For running and managing local LLMs and embedding models.
- **[LangGraph](https://langchain-ai.github.io/langgraph/):** A LangChain library for building stateful, multi-actor applications like agents and chatbots.
- **[Pydantic](https://docs.pydantic.dev/):** For defining data schemas to enable reliable, structured output from LLMs.
- **[Tavily](https://tavily.com/):** A search API optimized for LLM agents, used in the agent notebook.

## 🔧 Project-Specific Setup

Before running these notebooks, ensure you have completed the general setup in the main repository README.

1.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

2.  **Pull Required Ollama Models:**
    These notebooks use the `qwen` model for generation and `nomic-embed-text` for embeddings. Pull them using the Ollama CLI:
    ```bash
    ollama pull qwen
    ollama pull nomic-embed-text
    ```

3.  **Set Tavily API Key (for Agent Notebook):**
    The `Build_a_RAG.ipynb` notebook uses the Tavily search tool for its agent.
    - Get a free API key from [Tavily](https://tavily.com/).
    - Set it as an environment variable:
      ```bash
      export TAVILY_API_KEY="your-tavily-api-key"
      ```
    - Alternatively, you can paste the key directly into the notebook where `TAVILY_API_KEY = ""` is defined.
---

## 📚 Notebooks Overview

This repository contains six notebooks, each focusing on a different aspect of building with LangChain.

### 1. Simple LLM Application (`Simple_LLM_application_with_chat_models_and_prompt_templates.ipynb`)

This notebook covers the fundamentals of interacting with an LLM using LangChain. It's the "Hello, World!" of building with local models.

**Key Concepts Demonstrated:**
- **Initializing a Local LLM:** How to instantiate `ChatOllama` to connect to a locally running model (`qwen3`).
- **Basic Invocation:** Sending a list of messages (System and Human) to the model to get a response.
- **Streaming:** How to stream the model's response token by token, which is essential for creating a responsive, real-time user experience.
- **Prompt Templates:** Using `ChatPromptTemplate` to create dynamic and reusable prompts. This separates the prompt logic from the application code and allows for easy insertion of variables like `{language}` and `{text}`.

### 2. Building a Stateful Chatbot (`Build_a_Chatbot.ipynb`)

This notebook addresses a core challenge in conversational AI: maintaining memory. LLMs are stateless by default, and this notebook shows how to build a true chatbot that remembers previous turns in a conversation.

**Key Concepts Demonstrated:**
- **State Management with LangGraph:** Introduces `LangGraph` to build a stateful graph. The state (`MessagesState`) is explicitly designed to hold a sequence of messages.
- **Conversation Memory:** Uses `MemorySaver` to persist the conversation history for a given `thread_id`, allowing the application to manage multiple, separate conversations.
- **Incorporating History:** The `MessagesPlaceholder` is used within a prompt template to dynamically inject the conversation history, giving the LLM the context it needs to respond appropriately.
- **Managing Context Window:** Implements `trim_messages` to strategically prune the conversation history, ensuring it doesn't exceed the model's context window limit.

### 3. Text Classification with Structured Output (`Classify_Text_into_Labels.ipynb`)

This notebook demonstrates how to force an LLM to provide its output in a specific, reliable JSON format. This is crucial for integrating LLMs into larger software systems where predictable data structures are required.

**Key Concepts Demonstrated:**
- **Structured Output:** Using the `.with_structured_output()` method on an LLM to ensure its response conforms to a predefined schema.
- **Pydantic Schemas:** Defining a Pydantic `BaseModel` to specify the desired output structure, including field names, types, and descriptions (e.g., `sentiment`, `aggressiveness`, `language`).
- **Finer Control with `enum`:** The notebook shows how to constrain the LLM's output to a predefined set of choices by using `enum` within the Pydantic schema. This is perfect for classification tasks where you only want specific labels.

### 4. Building an Extraction Chain (`Build_an_Extraction_Chain.ipynb`)

Building on the previous notebook, this one tackles a more advanced extraction task: identifying and extracting a list of multiple entities from a block of text.

**Key Concepts Demonstrated:**
- **Extracting Lists of Objects:** By nesting a Pydantic `Person` schema within a `Data` schema (`people: List[Person]`), the LLM is instructed to find and extract all instances of a person in the text.
- **Few-Shot Prompting for Reliability:** The notebook introduces "reference examples" to improve the model's performance. By providing examples of both successful extractions and cases where nothing should be extracted (e.g., text about the ocean), the model learns to be more accurate and avoid false positives.

### 5. Semantic Search Engine (Vector Store) (`Sementic_search_engine.ipynb`)

This notebook builds the foundational "Retrieval" component of a RAG system. It demonstrates the complete pipeline for converting a document into a searchable vector store.

**The Four Key Steps of Indexing:**
1.  **Load:** Uses `PyPDFLoader` to load an external document (`.pdf` file) into memory.
2.  **Split:** Employs `RecursiveCharacterTextSplitter` to break the document into smaller, semantically meaningful chunks.
3.  **Embed:** Uses `OllamaEmbeddings` with the `nomic-embed-text` model to convert each text chunk into a numerical vector (an embedding).
4.  **Store:** Stores these embeddings in an `InMemoryVectorStore`, creating an index that can be queried for semantic similarity. The notebook also shows how to perform similarity searches to find the most relevant chunks for a given query.

### 6. Building a RAG Application (`Build_a_RAG.ipynb`)

This is the capstone notebook, bringing everything together to build two types of RAG applications that can answer questions based on the content of the document indexed in the previous notebook.

**Two RAG Architectures are Explored:**
1.  **LangGraph-based RAG:** A straightforward, two-step graph is created using `LangGraph`:
    - **Retrieve:** The user's question is used to retrieve relevant document chunks from the vector store.
    - **Generate:** The retrieved chunks are passed as context to the LLM, along with the original question, to generate a factually grounded answer.
    - An advanced version shows "Query Analysis," where the LLM first decides *which section* of the document to search in, making retrieval more targeted.
2.  **Agent-based RAG:** This demonstrates a more intelligent and autonomous approach.
    - The `retrieve` function is wrapped as a `@tool` that the agent can choose to use.
    - The `create_react_agent` function builds an agent that, for any given query, can decide whether to:
        - Answer directly from its own knowledge (if the question is general, like "Hello").
        - Call the `retrieve` tool to find information in the document before generating an answer.

This final notebook showcases the power and flexibility of LangChain in building sophisticated, context-aware AI systems.

---

## Acknowledgements

These notebooks are inspired by and adapted from the official [LangChain Conceptual Documentation](https://python.langchain.com/docs/get_started/introduction). They serve as a practical implementation of those concepts using a fully local, open-source stack.

