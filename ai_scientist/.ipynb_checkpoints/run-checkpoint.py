from __future__ import annotations

import os
from typing import List, Type
import argparse
import asyncio

from concurrent.futures import ThreadPoolExecutor
from academy.exchange.local import LocalExchangeFactory
from academy.manager import Manager
from academy.handle import Handle

from academy.agent import Agent, action
from tools import make_random_normal_tool

from reflector import ReflectionAgent
from tool_caller import ToolCallerAgent
from user_agent import UserAgent

from tool_agents import MOFTransformerAgent, NormalRandomVariableAgent, UniformRandomVariableAgent, NoisyMOFTransformerAgent

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, 'ai_scientist/llm_prompts/')

class ExampleToolCaller(Agent):
    def __init__(self, handle:Handle[NormalRandomVariableAgent]):
        super().__init__()
        self.handle = handle
        self.tool = make_random_normal_tool(self.handle)
        print('Finished __init__()')

    @action
    async def call_tool(self):
        tool = make_random_normal_tool(self.handle)
        result = await self.tool.ainvoke({'cif_dir': 'path'})
        print('Finished Calling Tool')
        print(result)
        return

async def main() -> None:
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:
        
        API_KEY = os.environ.get('OPENAI_API_KEY')

        reflection_agent_handle = await manager.launch(ReflectionAgent, args=(API_KEY,), kwargs={"model":"gpt-4o", "prompts_dir":PROMPTS_DIR})

        mof_transformer_agent_handle = await manager.launch(MOFTransformerAgent)

        random_normal_agent_handle = await manager.launch(NormalRandomVariableAgent)

        random_uniform_agent_handle = await manager.launch(UniformRandomVariableAgent)

        noisy_mof_transformer_agent_handle = await manager.launch(NoisyMOFTransformerAgent)

        example_tool_caller_handle = await manager.launch(ExampleToolCaller, args = (random_normal_agent_handle, ))
        
        tool_calling_agent_handle = await manager.launch(ToolCallerAgent, 
                                                         args = (API_KEY, reflection_agent_handle, 
                                                                 mof_transformer_agent_handle, 
                                                                  random_normal_agent_handle, 
                                                                random_uniform_agent_handle, 
                                                                noisy_mof_transformer_agent_handle, 
                                                                ), 
                                                        kwargs={"model":"gpt-4o", "prompts_dir":PROMPTS_DIR})
            
        while True:
            prompt = input("\nYou: ")
            if prompt == 'EXIT':
                break
            print(f'USER PROMPT: {prompt}')
            future = await tool_calling_agent_handle.query_scientist(prompt)
            _, response = await future
            print(response)

        await asyncio.sleep(2)
        
        await manager.shutdown(reflection_agent_handle, blocking=True)
        await manager.shutdown(tool_calling_agent_handle, blocking=True)
        await manager.shutdown(mof_transformer_agent_handle, blocking=True)
        await manager.shutdown(random_normal_agent_handle, blocking=True)
        await manager.shutdown(random_uniform_agent_handle, blocking=True)
        await manager.shutdown(noisy_mof_transformer_agent_handle, blocking=True)
        
    
if __name__ == '__main__':
    asyncio.run(main())
