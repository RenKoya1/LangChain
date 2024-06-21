# tool types
# Apify
# Bash
# Bing Search
# ChatGPT Plugins
# Google Search
# Google Serper API
# Human as a tool
# IFTTT WebHooks
# OpenWeatherMap API
# Python REPL
# Requests
# Search Tools
# SearxNG Search API
# SerpAPI
# Wikipedia API
# Wolfram Alpha
# Zapier Natural Language Actions API
# Example with SimpleSequentialChain

import os
from langchain import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_experimental.utilities import PythonREPL

openai_api_key = os.environ.get("OPENAI_API_KEY")

llm = OpenAI(api_key=openai_api_key)

python_repl = PythonREPL()

tools = [
    Tool(
        name="Python REPL",
        func=python_repl.run,
        description="Executes arbitrary Python code."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

response = agent("Execute the following Python code: print(1 + 1)")

# 結果を表示
print(response)