from __future__ import annotations

import os
from typing import List, Type
import asyncio


# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

from academy.agent import Agent, action, loop
from academy.handle import Handle

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, 'ai_scientist/llm_prompts/')

class ReflectionAgent(Agent):
    def __init__(self, api_key: str, model: str = "gpt-4o", prompts_dir = PROMPTS_DIR) -> None:
        super().__init__()
        if not api_key:
            raise ValueError("API key cannot be empty.")
    
        self.api_key = api_key
        self.model_name = model
    
        with open(os.path.join(prompts_dir, 'reflection_prompt.txt'), 'r') as f:
            self.reflection_prompt = f.read()
    
        self.reflection_agent_executor = self._create_reflection_agent()
    
        self.max_reflections = 5


    def _create_reflection_agent(self):
        print('Launching Reflection Agent')
        """Creates the agent responsible for reflecting on the initial output."""
        llm = ChatOpenAI(model=self.model_name, api_key=self.api_key, temperature=0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.reflection_prompt),
            MessagesPlaceholder(variable_name="reflection_input"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
       
        reflector_agent = create_openai_tools_agent(llm = llm, tools = [], prompt = prompt)
        return AgentExecutor(agent=reflector_agent, tools=[], verbose=True)


    @action
    async def reflect(self, reflection_input: list):
        """
            input: reflection_input (list) - a list of messages to the reflection agent executor.
                   ex: 
                    [HumanMessage(content=f"Original user prompt: {message}"),
                     AIMessage(content=f"AI Scientist output: {response['output']}"), 
                     HumanMessage(content=f"The following tools were used: {used_tools}")]
                     
            return: (1) reflection feedback; includes "INSUFFICIENT" if insufficient or inadequate tools are 
                    available
                    (2) total tokens used
                    
        """
        
        reflection_output = await self.reflection_agent_executor.ainvoke({"reflection_input": reflection_input})
  
        return reflection_output['output']


