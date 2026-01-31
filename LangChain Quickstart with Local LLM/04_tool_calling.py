from pprint import pprint

from langchain_core.tools import tool
from setup import QWEN3_MODEL, create_model

@tool
def calculate_hp_per_liter(horsepower: int, displacement_liters: float) -> float:
    """Calculate the specific output (efficiency) of an engine."""
    return round(horsepower / displacement_liters, 2)

tools = {calculate_hp_per_liter.name: calculate_hp_per_liter}
model_with_tools = create_model(QWEN3_MODEL).bind_tools([calculate_hp_per_liter])

query = "The Gordon Murray T.50 has a 3.9L V12 making 654 HP. Calculate its specific output."
response = model_with_tools.invoke(query)
tool_call = response.tool_calls[0]
pprint(tool_call, indent=2)

print(f"Response: {tools[tool_call["name"]].invoke(tool_call["args"])}")