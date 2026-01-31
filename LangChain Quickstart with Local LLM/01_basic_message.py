from setup import model

response = model.invoke(
    "What is the firing order of a C63 W204 M156 V8 engine? Reply with just the numbers."
)

print(response.content)