from __future__ import annotations

import os
from typing import List, Type
import asyncio
from collections import deque
from functools import partial

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.tools import Tool

#from tools import predict_mof_property, predict_mof_property_fake1, predict_mof_property_fake2, predict_mof_property_fake3
from tools import make_mof_transformer_tool, make_random_normal_tool, make_random_uniform_tool, make_noisy_mof_transformer_tool, make_mof_db_tool

from reflector import ReflectionAgent
from tool_agents import MOFTransformerAgent, NormalRandomVariableAgent, UniformRandomVariableAgent, NoisyMOFTransformerAgent, MOFDatabaseAgent

from academy.agent import Agent, action, loop
from academy.handle import Handle

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, 'ai_scientist/llm_prompts/')

class ToolCallerAgent(Agent):
    def __init__(self, api_key: str,  
                 reflection_agent: Handle[ReflectionAgent],
                 mof_transformer_agent: Handle[MOFTransformerToolAgent], 
                 random_normal_agent: Handle[NormalRandomVariableAgent], 
                 random_uniform_agent: Handle[UniformRandomVariableAgent],
                 noisy_mof_transformer_agent: Handle[NoisyMOFTransformerAgent],
                 mof_db_agent: Handle[MOFDatabaseAgent],
                 model: str = "gpt-4o",
                 prompts_dir = PROMPTS_DIR):

        super().__init__()
        
        if not api_key:
            raise ValueError("API key cannot be empty.")
        

        self.api_key = api_key
        self.model_name = model

        self.reflection_agent = reflection_agent
        self.mof_transformer_agent = mof_transformer_agent
        self.random_normal_agent = random_normal_agent
        self.random_uniform_agent = random_uniform_agent
        self.noisy_mof_transformer_agent = noisy_mof_transformer_agent
        self.mof_db_agent = mof_db_agent
        
        self._tools = None
        self._executor = None
        
        with open(os.path.join(prompts_dir, 'reflection_prompt.txt'), 'r') as f:
            self.reflection_prompt = f.read()

        self.max_reflections = 5
        self.chat_history = []

        print("AI Scientist (LangChain MOF Agent) Initializing...")
        print(f"Model selected: {self.model_name}")

    async def agent_on_startup(self) -> None:
        self.tools = []
        self.tools.append(make_mof_transformer_tool(self.mof_transformer_agent))
        self.tools.append(make_random_normal_tool(self.random_normal_agent))
        self.tools.append(make_random_uniform_tool(self.random_uniform_agent))
        self.tools.append(make_noisy_mof_transformer_tool(self.noisy_mof_transformer_agent)) 
        self.tools.append(make_mof_db_tool(self.mof_db_agent))

        self.agent_executor = self._create_agent(self.tools)

    def _create_agent(self, tools: List):
        print('Launching Tool-Calling Agent')
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

            response = await self.agent_executor.ainvoke({
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




