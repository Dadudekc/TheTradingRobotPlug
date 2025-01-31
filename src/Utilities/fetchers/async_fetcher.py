# -------------------------------------------------------------------
# File: async_fetcher.py
# Location: src/Utilities/fetchers
# Description: Handles asynchronous fetching of data.
# -------------------------------------------------------------------

import aiohttp
import asyncio
from typing import List, Awaitable, Any

class AsyncFetcher:
    def __init__(self, logger):
        self.logger = logger

    async def create_session(self) -> aiohttp.ClientSession:
        """
        Creates and returns an aiohttp ClientSession.
        """
        self.logger.debug("Creating new aiohttp ClientSession.")
        return aiohttp.ClientSession()

    async def fetch_multiple(self, tasks: List[Awaitable[Any]]) -> List[Any]:
        """
        Executes multiple asynchronous tasks concurrently.

        Args:
            tasks (List[Awaitable[Any]]): List of async tasks to execute.

        Returns:
            List[Any]: Results of the executed tasks.
        """
        self.logger.debug(f"Executing {len(tasks)} asynchronous tasks.")
        return await asyncio.gather(*tasks, return_exceptions=True)
