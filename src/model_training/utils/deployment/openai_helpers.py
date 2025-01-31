# -------------------------------------------------------------------
# File Path: C:/TheTradingRobotPlug/Scripts/model_training/utils/openai_helpers.py
# Description: Advanced OpenAI helper module for feature suggestions, 
#              hyperparameter tuning, performance reports, and risk assessments.
#              Includes asynchronous operations, caching, rate limiting, 
#              secure environment management, and robust error handling.
# -------------------------------------------------------------------

import os
import logging
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter
from async_lru import alru_cache
import json

# Load environment variables from .env file
load_dotenv()

# Constants for rate limiting (adjust based on OpenAI's API limits)
RATE_LIMIT = 60  # number of requests
RATE_PERIOD = 60  # per seconds

# Initialize the rate limiter
limiter = AsyncLimiter(max_rate=RATE_LIMIT, time_period=RATE_PERIOD)

class OpenAIUtils:
    def __init__(
        self,
        model: str = "gpt-4",
        logger: Optional[logging.Logger] = None,
        max_retries: int = 5,
        backoff_factor: float = 0.5,
        environment: str = "development"
    ):
        """
        Initialize the OpenAI helper class with advanced configurations.

        Args:
            model (str): The OpenAI model to use (default: 'gpt-4').
            logger (logging.Logger): Logger instance for logging (optional).
            max_retries (int): Maximum retry attempts for API requests (default: 5).
            backoff_factor (float): Factor for exponential backoff (default: 0.5).
            environment (str): Deployment environment (default: 'development').
        """
        self.model = model
        self.logger = logger if logger else self._setup_default_logger(environment)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.api_key = self._initialize_api_key()

        if not self.api_key:
            self.logger.error("OpenAI API key is missing. Please set it in the .env file.")
            raise ValueError("Missing OpenAI API key.")

        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _setup_default_logger(self, environment: str) -> logging.Logger:
        """
        Set up a default logger with advanced configurations.

        Args:
            environment (str): Deployment environment.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger("OpenAIUtils")
        logger.setLevel(logging.DEBUG if environment == "development" else logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)

        return logger

    def _initialize_api_key(self) -> Optional[str]:
        """
        Load OpenAI API key from environment variables.

        Returns:
            Optional[str]: OpenAI API key if available, else None.
        """
        return os.getenv("OPENAI_API_KEY")

    async def _handle_openai_error(self, e: Exception, attempt: int) -> bool:
        """
        Handle OpenAI API errors with detailed logging and determine if a retry should be attempted.

        Args:
            e (Exception): The exception encountered.
            attempt (int): Current attempt number.

        Returns:
            bool: True if the request should be retried, False otherwise.
        """
        error_type = type(e).__name__
        if isinstance(e, aiohttp.ClientResponseError):
            status = e.status
            if status == 429:
                self.logger.warning(f"[{error_type}] Rate limit exceeded on attempt {attempt}.")
            elif status == 401:
                self.logger.error(f"[{error_type}] Authentication failed. Check your API key.")
                return False
            else:
                self.logger.error(f"[{error_type}] API error {status}: {e.message}")
        else:
            self.logger.error(f"[{error_type}] Unexpected error: {str(e)}")

        return attempt < self.max_retries

    @alru_cache(maxsize=128)
    async def _execute_with_retry(self, prompt: str, max_tokens: int) -> str:
        """
        Execute OpenAI API call with retry logic and exponential backoff.

        Args:
            prompt (str): The prompt to send to the OpenAI API.
            max_tokens (int): Maximum number of tokens in the response.

        Returns:
            str: The content of the response or an error message.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "n": 1,
            "stop": None,
            "temperature": 0.7
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                async with limiter:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            self.api_url,
                            headers=self.headers,
                            json=payload
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                content = data['choices'][0]['message']['content'].strip()
                                self.logger.debug(f"API response on attempt {attempt}: {content}")
                                return content
                            else:
                                error_text = await response.text()
                                raise aiohttp.ClientResponseError(
                                    status=response.status,
                                    message=error_text,
                                    request_info=response.request_info,
                                    history=response.history
                                )
            except Exception as e:
                should_retry = await self._handle_openai_error(e, attempt)
                if not should_retry:
                    break
                sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                self.logger.info(f"Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)

        return "Error: Unable to process the request after multiple attempts."

    def _generate_prompt(self, template: str, **kwargs) -> str:
        """
        Generate a prompt by formatting the template with provided keyword arguments.

        Args:
            template (str): The prompt template.
            **kwargs: Keyword arguments to format the template.

        Returns:
            str: The formatted prompt.
        """
        return template.format(**kwargs)

    async def suggest_new_features(
        self,
        data_columns: List[str],
        model_context: str = "trading model"
    ) -> str:
        """
        Suggest new derived features based on existing columns.

        Args:
            data_columns (List[str]): A list of current data columns used in the model.
            model_context (str): Additional context about the model.

        Returns:
            str: Suggested new features or transformations.
        """
        prompt_template = (
            "You are a machine learning expert specializing in {model_context}. "
            "Given these financial indicators: {data_columns}, suggest derived features or transformations "
            "that could improve model predictions for stock trading. Consider technical indicators, "
            "mathematical transformations, or domain-specific insights to enhance predictive power."
        )
        prompt = self._generate_prompt(
            prompt_template,
            data_columns=', '.join(data_columns),
            model_context=model_context
        )
        return await self._execute_with_retry(prompt, max_tokens=200)

    async def suggest_hyperparameters(
        self,
        model_name: str,
        mse: float,
        rmse: float,
        r2: float,
        model_goal: str = "predict stock prices"
    ) -> str:
        """
        Suggest optimal hyperparameters based on performance metrics.

        Args:
            model_name (str): The name of the model being used (e.g., LSTM, Random Forest).
            mse (float): Mean Squared Error of the model.
            rmse (float): Root Mean Squared Error of the model.
            r2 (float): R² score of the model.
            model_goal (str): The goal of the model.

        Returns:
            str: Suggested hyperparameters with reasoning.
        """
        prompt_template = (
            "You are assisting in tuning a {model_name} model to {model_goal}. "
            "The current performance metrics are: MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}. "
            "What hyperparameters should be adjusted to improve this model's performance, "
            "considering both speed and accuracy? Provide actionable tuning suggestions with reasoning."
        )
        prompt = self._generate_prompt(
            prompt_template,
            model_name=model_name,
            model_goal=model_goal,
            mse=mse,
            rmse=rmse,
            r2=r2
        )
        return await self._execute_with_retry(prompt, max_tokens=250)

    async def generate_model_report(
        self,
        mse: float,
        rmse: float,
        mae: float,
        r2: float,
        model_name: str = "LSTM",
        business_goal: str = "predicting stock prices"
    ) -> str:
        """
        Generate a comprehensive performance report with suggestions for improvement.

        Args:
            mse (float): Mean Squared Error of the model.
            rmse (float): Root Mean Squared Error of the model.
            mae (float): Mean Absolute Error of the model.
            r2 (float): R² score of the model.
            model_name (str): Name of the model.
            business_goal (str): The specific goal of the model in the business context.

        Returns:
            str: Generated report with suggestions for improvement.
        """
        prompt_template = (
            "As a machine learning expert, review the performance of a {model_name} model used for {business_goal}. "
            "The model's metrics are: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, and R²={r2:.2f}. "
            "Given these metrics, what specific steps should be taken to improve the model’s performance? "
            "Provide recommendations on tuning, data adjustments, or architecture modifications."
        )
        prompt = self._generate_prompt(
            prompt_template,
            model_name=model_name,
            business_goal=business_goal,
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2
        )
        return await self._execute_with_retry(prompt, max_tokens=300)

    async def summarize_results(
        self,
        results: Dict[str, Any],
        context: str = "trading strategy optimization"
    ) -> str:
        """
        Summarize key findings and insights based on provided results.

        Args:
            results (Dict[str, Any]): Dictionary of model results or performance metrics.
            context (str): Additional context for the summary.

        Returns:
            str: Summarized insights and recommendations.
        """
        prompt_template = (
            "As an AI specialized in {context}, please summarize the following results and offer insights for improvement: {results}. "
            "Provide actionable insights and suggestions to enhance performance."
        )
        results_json = json.dumps(results)
        prompt = self._generate_prompt(
            prompt_template,
            results=results_json,
            context=context
        )
        return await self._execute_with_retry(prompt, max_tokens=250)

    async def get_risk_assessment(
        self,
        model_performance: Dict[str, Any],
        financial_goal: str = "risk management in stock trading"
    ) -> str:
        """
        Generate a risk assessment based on model performance and financial goals.

        Args:
            model_performance (Dict[str, Any]): Dictionary of model performance metrics.
            financial_goal (str): Specific financial goal or risk context.

        Returns:
            str: Risk assessment and suggestions for mitigating risks.
        """
        prompt_template = (
            "As a financial AI advisor, assess the risks based on the following performance metrics: {model_performance}. "
            "Focus on {financial_goal} and suggest ways to mitigate potential risks."
        )
        performance_json = json.dumps(model_performance)
        prompt = self._generate_prompt(
            prompt_template,
            model_performance=performance_json,
            financial_goal=financial_goal
        )
        return await self._execute_with_retry(prompt, max_tokens=250)

    async def parallel_requests(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """
        Execute multiple OpenAI requests in parallel to improve efficiency.

        Args:
            tasks (List[Dict[str, Any]]): A list of dictionaries with 'prompt' and 'max_tokens' for each API call.

        Returns:
            List[str]: Responses from each request.
        """
        responses = []
        semaphore = asyncio.Semaphore(RATE_LIMIT)

        async def bound_execute(task):
            async with semaphore:
                return await self._execute_with_retry(task['prompt'], task['max_tokens'])

        tasks_coroutines = [bound_execute(task) for task in tasks]
        for coroutine in asyncio.as_completed(tasks_coroutines):
            response = await coroutine
            responses.append(response)
        return responses

    # Additional utility methods can be added here

# Example Usage:
# Ensure that you run the asynchronous methods within an event loop.
# Here's a simple example of how to use the OpenAIUtils class.

if __name__ == "__main__":
    import asyncio

    async def main():
        utils = OpenAIUtils(environment="development")

        # Example: Suggest new features
        data_columns = ["moving_average", "volume", "price_change"]
        features = await utils.suggest_new_features(data_columns)
        print("Suggested Features:", features)

        # Example: Suggest hyperparameters
        hyperparams = await utils.suggest_hyperparameters(
            model_name="Random Forest",
            mse=0.25,
            rmse=0.5,
            r2=0.85
        )
        print("Hyperparameter Suggestions:", hyperparams)

        # Example: Generate model report
        report = await utils.generate_model_report(
            mse=0.25,
            rmse=0.5,
            mae=0.4,
            r2=0.85
        )
        print("Model Report:", report)

    asyncio.run(main())