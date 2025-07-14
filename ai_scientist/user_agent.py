from __future__ import annotations

import os
from typing import List, Type
import asyncio
from collections import deque

# LangChain Imports
from tool_caller import ToolCallerAgent

from academy.agent import Agent, action, loop
from academy.handle import Handle

class UserAgent(Agent):
    def __init__(self, tool_calling_agent: Handle[ToolCallerAgent]) -> None:
        super().__init__()
        self.tool_calling_agent = tool_calling_agent
        self.user_queries = deque()


    @action
    async def add_user_query(self, query: str) -> None:
        print('Added user query')
        self.user_queries.append(query)

    @loop
    async def call_ai_scientist(self, shutdown: asyncio.Event) -> None:
        while not shutdown.is_set():
            if len(self.user_queries) > 0:
                query = self.user_queries.popleft()
                print('Awaiting query_scientist')
                future = await self.tool_calling_agent.query_scientist(query)
                _, response = await future
                print(f'AI Scientist response: {response}')

            await asyncio.sleep(2)


            
