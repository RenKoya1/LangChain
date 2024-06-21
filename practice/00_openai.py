import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "What is the meaning of life?",
        }
    ],
    model="gpt-3.5-turbo",
)
print(response.choices[0].message.content)