from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)
output = chat.predict_messages([HumanMessage(content="What is the meaning of life?")])

print(output)