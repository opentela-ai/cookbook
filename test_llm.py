import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OTELA_API"),
    base_url="https://api.opentela.ai/v1/service/llm/v1",
)

resp = client.chat.completions.create(
    model="moonshotai/Kimi-K3",
    messages=[{"role": "user", "content": "Who is Alan Turing?"}],
    stream=True
)
for chunk in resp:
    if chunk.choices[0].delta.reasoning_content is not None:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)