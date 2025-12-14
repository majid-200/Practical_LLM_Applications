# Practical LLM Applications

This repository is a collection of projects demonstrating practical applications with Large Language Models (LLMs). The key focus is on runimg open-source models locally, enabling development without relying on proprietary APIs.

Each project is contained within its own folder and includes a dedicated README with detailed setup and conceptual explanations.

---

## 🚀 Projects

| Project                                                     | Description                                                                                                   | Key Concepts Covered                        |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **LangChain Conceptual Tutorials** | A series of notebooks adapted from official LangChain documentation, progressing from basics to advanced agents. | RAG, Agents, Chatbots, Structured Output, Vector Stores |
| **Analyze Financial Data with AI** | **"FinVault"**: A Streamlit application featuring a hierarchical multi-agent team. A "Supervisor" agent delegates tasks to specialized analysts to research stocks and SEC filings. | Multi-Agent Systems, Supervisor-Worker Pattern, Streamlit UI, Tool Calling, Financial Analysis |
| *(More projects will be added here...)*                     |                                                                                                               |                                             |


---

## 🛠️ Core Technologies

This repository primarily uses the following technologies:

- **[LangChain]:** The core framework for orchestrating LLM workflows.
- **[Ollama]:** For running and managing local LLMs and embedding models.
- **[LangGraph]:** For building stateful, multi-actor applications.
- **[Streamlit]:** For building interactive web interfaces for the AI applications.
- **[Pydantic]:** For defining data schemas to enable reliable, structured output.

## 🔧 General Setup

1.  **Clone the Repository:**
   
    ```bash
    git clone https://github.com/majid-200/Practical_LLM_Applications.git
    cd Practical_LLM_Applications
    ```

2.  **Create a Virtual Environment:**
   
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
   
    Install dependencies based on the project you want to run. 
