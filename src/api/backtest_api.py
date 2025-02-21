"""
File: backtest_api.py
Location: src/api

Description:
    FastAPI-based API for triggering backtests asynchronously.
    Provides:
      - Asynchronous execution with background task management.
      - Dynamic strategy loading from the STRATEGY_REGISTRY.
      - Performance tracking with detailed backtest logs.
      - Extended error handling for robustness.
"""
import sys
from pathlib import Path

# Define project root dynamically and add it to sys.path
project_root = Path(__file__).resolve().parents[2]  # Adjust based on depth
sys.path.append(str(project_root))

import logging
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

# Internal Imports
from src.Utilities.strategies.backtester import BacktestRunner
from src.Utilities.strategies.registry import STRATEGY_REGISTRY

# --------------------------------------------------------------------------------
# FastAPI Initialization
# --------------------------------------------------------------------------------
app = FastAPI(
    title="Backtest API",
    version="2.0",
    description="A robust API to trigger and manage backtests asynchronously.",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# --------------------------------------------------------------------------------
# Logging Setup
# --------------------------------------------------------------------------------
logger = logging.getLogger("BacktestAPI")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --------------------------------------------------------------------------------
# Global Dictionary for Running Tasks
# --------------------------------------------------------------------------------
BACKTEST_TASKS = {}

# --------------------------------------------------------------------------------
# API Models
# --------------------------------------------------------------------------------
class BacktestRequest(BaseModel):
    """
    Schema for backtest request payload.
    """
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    interval: str
    initial_balance: Optional[float] = 10000.0
    apply_unifier: Optional[bool] = False


class BacktestStatusResponse(BaseModel):
    """
    Schema for tracking ongoing backtests.
    """
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------------
# Background Task Execution
# --------------------------------------------------------------------------------
async def execute_backtest(task_id: str, request: BacktestRequest):
    """
    Asynchronous function to execute the backtest.
    Stores results in the BACKTEST_TASKS dictionary.
    """
    try:
        logger.info(f"[TASK-{task_id}] Starting backtest for {request.symbol} using {request.strategy}...")

        # Validate strategy
        if request.strategy not in STRATEGY_REGISTRY:
            error_msg = f"Strategy '{request.strategy}' not found."
            logger.error(f"[TASK-{task_id}] {error_msg}")
            BACKTEST_TASKS[task_id] = {"status": "failed", "error": error_msg}
            return

        # Initialize BacktestRunner
        runner = BacktestRunner(
            strategy_name=request.strategy,
            initial_balance=request.initial_balance,
            apply_unifier=request.apply_unifier
        )

        # Execute backtest
        metrics = await asyncio.to_thread(
            runner.run, request.symbol, request.start_date, request.end_date, request.interval
        )

        # Store result
        BACKTEST_TASKS[task_id] = {"status": "completed", "result": metrics}
        logger.info(f"[TASK-{task_id}] Backtest completed successfully for {request.symbol}.")

    except Exception as e:
        error_msg = f"Unexpected error during backtest: {e}"
        logger.error(f"[TASK-{task_id}] {error_msg}", exc_info=True)
        BACKTEST_TASKS[task_id] = {"status": "failed", "error": error_msg}


# --------------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------------
@app.post("/run_backtest", response_model=BacktestStatusResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Triggers a backtest asynchronously and returns a task ID for tracking.

    Args:
        request (BacktestRequest): The backtest parameters.
        background_tasks (BackgroundTasks): FastAPI's background task manager.

    Returns:
        BacktestStatusResponse: Contains the task ID and initial status.
    """
    task_id = f"{request.strategy}-{request.symbol}-{datetime.utcnow().timestamp()}"
    BACKTEST_TASKS[task_id] = {"status": "running"}
    
    # Run the backtest asynchronously
    background_tasks.add_task(execute_backtest, task_id, request)

    logger.info(f"Backtest task started: {task_id}")
    return BacktestStatusResponse(task_id=task_id, status="running")


@app.get("/backtest_status/{task_id}", response_model=BacktestStatusResponse)
def get_backtest_status(task_id: str):
    """
    Retrieves the status of an ongoing or completed backtest.

    Args:
        task_id (str): The ID of the backtest task.

    Returns:
        BacktestStatusResponse: The current status and result if available.
    """
    if task_id not in BACKTEST_TASKS:
        raise HTTPException(status_code=404, detail=f"Task ID '{task_id}' not found.")

    task_info = BACKTEST_TASKS[task_id]
    return BacktestStatusResponse(task_id=task_id, **task_info)


@app.get("/list_strategies", response_model=Dict[str, List[str]])
def list_strategies():
    """
    Lists all available trading strategies.

    Returns:
        Dict[str, List[str]]: A dictionary containing available strategies.
    """
    strategies = list(STRATEGY_REGISTRY.keys())
    logger.info("Available strategies retrieved.")
    return {"strategies": strategies}


@app.get("/health", response_model=Dict[str, str])
def health_check():
    """
    Health-check endpoint to verify API availability.

    Returns:
        Dict[str, str]: API status response.
    """
    return {"status": "OK"}


# --------------------------------------------------------------------------------
# Extended Logging & Scalability Features
# --------------------------------------------------------------------------------
@app.delete("/clear_completed_tasks")
def clear_completed_tasks():
    """
    Clears completed or failed tasks from memory to maintain efficiency.

    Returns:
        Dict[str, str]: Response indicating cleanup status.
    """
    completed_tasks = [task_id for task_id, data in BACKTEST_TASKS.items() if data["status"] in ["completed", "failed"]]
    for task_id in completed_tasks:
        del BACKTEST_TASKS[task_id]

    logger.info(f"Cleared {len(completed_tasks)} completed tasks.")
    return {"message": f"Cleared {len(completed_tasks)} completed tasks."}
