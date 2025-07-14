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

from tools import predict_mof_property, predict_mof_property_fake1, predict_mof_property_fake2, predict_mof_property_fake3
from tools import MOFTransformerToolAgent, MofPredictionInput

from academy.agent import Agent, action, loop
from concurrent.futures import ThreadPoolExecutor
from academy.exchange.local import LocalExchangeFactory
from academy.manager import Manager
from academy.handle import Handle

from langchain.tools import tool

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, 'ai_scientist/llm_prompts/')


class ToolCallerAgent(Agent):
    def __init__(self, api_key: str,  
                 reflection_agent: Handle[ReflectionAgent],
                 moftrans_agent: Handle[MOFTransformerToolAgent],
                 model: str = "gpt-4o",
                 prompts_dir = PROMPTS_DIR):

        super().__init__()
        
        if not api_key:
            raise ValueError("API key cannot be empty.")
        

        self.api_key = api_key
        self.model_name = model
        
        self.tools = [predict_mof_property_agentic, predict_mof_property_fake1, predict_mof_property_fake2, predict_mof_property_fake3]

        with open(os.path.join(prompts_dir, 'reflection_prompt.txt'), 'r') as f:
            self.reflection_prompt = f.read()
        
        self.agent_executor = self._create_agent(self.tools)

        self.max_reflections = 5
        self.chat_history = []

        self.reflection_agent = reflection_agent
        

        print("AI Scientist (LangChain MOF Agent) Initializing...")
        print(f"Model selected: {self.model_name}")

    def _create_agent(self, tools: List):
        llm = ChatOpenAI(model=self.model_name, api_key=self.api_key, temperature=0)

        with open(os.path.join(PROMPTS_DIR, 'system_prompt.txt'), 'r') as prompt_file:
            system_prompt = prompt_file.read()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_openai_tools_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

    @action
    async def query_scientist(self, message: str) -> str:
        if not message:
            return "Received an empty query. Please provide a prompt."

        reflection_iter = 1
        
        reflection_chat_history = self.chat_history

        final_response = ""
        success_code = 0 # 0 = success, 1 = early exit by tool-calling agent, 2 = exit after reflection
        
        while reflection_iter <= self.max_reflections:
            print(f"\n[Invoking agent with prompt: '{message}']")

            response = self.agent_executor.invoke({
                "input": message,
                "chat_history": reflection_chat_history
            })

            used_tools = []
            if "intermediate_steps" in response:
                for action, observation in response["intermediate_steps"]:
                    used_tools.append(action.tool)

            print(f"USED TOOLS: {used_tools}")

            reflection_inpt = [HumanMessage(content=f"Original user prompt: {message}"),
                               AIMessage(content=f"AI Scientist output: {response['output']}"), 
                               HumanMessage(content=f"The following tools were used: {used_tools}")]

            refl_future = await self.reflection_agent.reflect(reflection_inpt)
            reflection_output = await refl_future
            
            reflection_chat_history.append(AIMessage(content=response['output']))

            if "INSUFFICIENT" in response['output']:
                final_response = 'There are insufficient or inadequate tools to respond to the user request.'
                success_code = 1
                break

            if "INSUFFICIENT" in reflection_output:
                final_response = 'There are insufficient or inadequate tools to respond to the user request.'
                success_code = 2
                break

            if "PROCEED" in reflection_output:
                reflection_chat_history.append(HumanMessage(content=f"Reflection feedback: {reflection_output}."))
                final_response = response['output']
                break

            print("Reflection: Re-running the agent based on feedback.")
            reflection_msg = f"Reflection feedback: {reflection_output}. Please generate a new " \
            "response by taking the reflection agent output into consideration."
            
            reflection_chat_history.append(HumanMessage(content=reflection_msg))
            
            reflection_iter += 1

        # summary/explanation of tool use
        if success_code==0 and len(used_tools) > 0:
            interpretability_prompt = f"""
            You are now tasks to summarize the tools you have used, how you have used them, and why you decided to use them in that way. 
            Here is the original user's request: {message}. You used the following tools in your latest response: {used_tools}. Consider 
            carefully the input arguments to each tool and the functionality of each tool. Also, mention the tools by their original names 
            and NOT by the Python function name. Please be as concise as possible to communicate the necessary information in as few words
            as possible while maintaining clarity of thought.
    
            """
            interp_response = self.agent_executor.invoke({
                    "input": [interpretability_prompt],
                    "chat_history": reflection_chat_history
                })

            final_response = final_response + "\n" + interp_response['output']

        history = f'''Input: {message};\nAI Scientist Result: {final_response}\n'''
        self.chat_history.append(history)
            
        with open(os.path.join(PROMPTS_DIR, 'history.txt'), 'a') as history_file:
            history_file.write(history)
            
        return success_code, final_response

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
        
        reflection_output = self.reflection_agent_executor.invoke({"reflection_input": reflection_input})['output']
  
        return reflection_output


class UserAgent(Agent):
    def __init__(self, tool_calling_agent: Handle[ToolCallerAgent]) -> None:
        super().__init__()
        self.tool_calling_agent = tool_calling_agent
        self.user_queries = deque()


    @action
    async def add_user_query(self, query: str) -> None:
        self.user_queries.append(query)

    @loop
    async def call_ai_scientist(self, shutdown: asyncio.Event) -> None:
        while not shutdown.is_set():
            if len(self.user_queries) > 0:
                query = self.user_queries.popleft()
                future = await self.tool_calling_agent.query_scientist(query)
                _, response = await future
                print(f'AI Scientist response: {response}')

            await asyncio.sleep(5)
            


async def ainput(prompt: str = ""):
    return await asyncio.to_thread(input, prompt)

async def main() -> None:
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:

        parser = argparse.ArgumentParser()
        parser.add_argument('--clear-history', action='store_true')
    
        args = parser.parse_args()
        API_KEY = os.environ.get('OPENAI_API_KEY')
    
        reflection_agent_handle = await manager.launch(ReflectionAgent, args=(API_KEY,), kwargs={"model":"gpt-4o", "prompts_dir":PROMPTS_DIR})

        mof_transformer_handle = await manager.launch(MOFTransformerToolAgent)

        tool_calling_agent_handle = await manager.launch(ToolCallerAgent, args = (API_KEY, reflection_agent_handle, mof_transformer_handle, ))

        user_agent_handle = await manager.launch(UserAgent, args = (tool_calling_agent_handle,))

        prompt = input("\nYou: ")
        print(f'USER PROMPT: {prompt}')
            
        await user_agent_handle.add_user_query(prompt)

        #await manager.shutdown(reflection_agent_handle, blocking=True)
        #await manager.shutdown(tool_calling_agent_handle, blocking=True)
        #await manager.shutdown(user_agent_handle, blocking=True)
        
        # Clear history
    if args.clear_history:
        open(os.path.join(PROMPTS_DIR, 'history.txt'), 'w').close()
    
if __name__ == '__main__':
    asyncio.run(main())






















