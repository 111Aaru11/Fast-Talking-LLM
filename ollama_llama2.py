from ollama import chat

response = chat(model = "llama2", messages = [{"role" : "user", "content": "write poem on moon"}])

print(response['message']['content'])
