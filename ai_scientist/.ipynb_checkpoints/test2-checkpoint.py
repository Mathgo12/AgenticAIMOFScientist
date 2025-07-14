from __future__ import annotations

import os
from typing import List, Type
import argparse
import asyncio

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from concurrent.futures import ThreadPoolExecutor
from academy.exchange.local import LocalExchangeFactory
from academy.manager import Manager
from academy.handle import Handle

from academy.agent import Agent, action, loop

from reflector import ReflectionAgent
from tool_caller import ToolCallerAgent
from user_agent import UserAgent

from tools import make_mof_transformer_tool, make_random_normal_tool, make_random_uniform_tool, make_noisy_mof_transformer_tool
from tools import FakePredictor1
from tool_agents import MOFTransformerAgent, NormalRandomVariableAgent, UniformRandomVariableAgent, NoisyMOFTransformerAgent

class ExampleTool(Agent):
    @action
    async def test_action(self):
        print('Example Action Executed')
        return 1

def example_tool_maker(handle: Handle[ExampleTool]):
    @tool
    async def example_tool():
        """
            Prints 'Example Action Executed' and returns 1
        """
        future = await handle.test_action()
        result = await future
        return result
    return example_tool

class ExampleToolCaller(Agent):
    def __init__(self, handle:Handle[ExampleTool]):
        super().__init__()
        self.handle = handle
        self._tool = None

    @property
    def tool(self):
        if self._tool is None:
            self._tool = example_tool_maker(self.handle)
        return self._tool

    @action
    async def call_tool(self):
        #tool = example_tool_maker(self.handle)
        result = await self.tool.ainvoke('')
        print('Finished Calling Tool')
        return result
        

async def main() -> None:
    async with await Manager.from_exchange_factory(factory=LocalExchangeFactory(), executors=ThreadPoolExecutor(),) as manager:
        tool_handle = await manager.launch(ExampleTool)
        handle = await manager.launch(ExampleToolCaller, args=(tool_handle, ))
        future = await handle.call_tool()
        result = await future
        print(result)
        
        await asyncio.sleep(2)
        await manager.shutdown(handle, blocking=True)

asyncio.run(main())













        






