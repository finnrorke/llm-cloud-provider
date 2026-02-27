#!/usr/bin/python3

from openai import OpenAI

client = OpenAI(
        base_url = "http://localhost:8000/v1",
        api_key="EMPTY"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello to LLM"}]
)

print(response.choices[0].message.content)
