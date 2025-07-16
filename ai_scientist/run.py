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

from tool_agents import MOFTransformerAgent, NormalRandomVariableAgent, UniformRandomVariableAgent, NoisyMOFTransformerAgent, MOFDatabaseAgent

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, 'ai_scientist/llm_prompts/')

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

        mof_db_agent_handle = await manager.launch(MOFDatabaseAgent)
        
        tool_calling_agent_handle = await manager.launch(ToolCallerAgent, 
                                                         args = (API_KEY, reflection_agent_handle, 
                                                                 mof_transformer_agent_handle, 
                                                                  random_normal_agent_handle, 
                                                                random_uniform_agent_handle, 
                                                                noisy_mof_transformer_agent_handle, 
                                                                mof_db_agent_handle
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
        await manager.shutdown(mof_db_agent_handle, blocking=True)
        
    
if __name__ == '__main__':
    asyncio.run(main())
