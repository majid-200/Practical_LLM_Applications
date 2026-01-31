import mlflow
from setup import model

mlflow.set_experiment("LangChain Local LLM Quickstart")
mlflow.langchain.autolog()

model.invoke(
    "Why is the McLaren F1 engine bay lined with gold? Explain in one sentence."
)