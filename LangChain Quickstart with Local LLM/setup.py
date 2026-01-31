from langchain.chat_models import BaseChatModel, init_chat_model

DEEPSEEK_MODEL = "deepseek-r1:8b"
QWEN3_MODEL = "qwen3:8b"

def create_model(model_name: str = QWEN3_MODEL) -> BaseChatModel:
    """Create a chat model instance based on the specified model name."""
    return init_chat_model(
        model_name,
        model_provider="ollama",
        reasoning=False,
        seed=42,
        )


model = create_model()