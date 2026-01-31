from langchain.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from setup import model

system_message = SystemMessage(
    "You are a Master Mechanic specializing in high-performance naturally aspirated engines. Keep answers technical within a sentence or two."
)

user_message = HumanMessagePromptTemplate.from_template(
    "Tell me about the engine in the {car_model} in one sentence."
)

prompt_template = ChatPromptTemplate.from_messages(
    [system_message, user_message]
)

prompt = prompt_template.format_messages(car_model="Honda S2000 (2004)")
print([m.model_dump() for m in prompt])

response = model.invoke(prompt)
print(response.content)