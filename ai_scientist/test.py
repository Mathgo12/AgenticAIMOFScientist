from __future__ import annotations

import os
from typing import List, Type
import argparse
import threading
import asyncio
from collections import deque


# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.callbacks import OpenAICallbackHandler

from tools import make_mof_transformer_tool, make_random_normal_tool, make_random_uniform_tool, make_noisy_mof_transformer_tool


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, 'ai_scientist/llm_prompts/')

API_KEY = os.environ.get('OPENAI_API_KEY')

model = 'gpt-4o'

tools = [predict_mof_property, predict_mof_property_fake1, 
         predict_mof_property_fake2, predict_mof_property_fake3]

llm = ChatOpenAI(model=model, api_key=API_KEY, temperature=0)

with open(os.path.join(PROMPTS_DIR, 'system_prompt.txt'), 'r') as prompt_file:
    system_prompt = prompt_file.read()

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

message = "Predict void fraction for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs"

async def get_response(message):
    response = await agent_executor.ainvoke({
        "input": message,
        "chat_history": []
    })

    return response['output']

print(asyncio.run(get_response(message)))





