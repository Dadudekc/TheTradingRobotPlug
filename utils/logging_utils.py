"""
utils/logging_utils.py
----------------------
Provides a decorator for handling exceptions and logging errors.
"""

import functools
import logging
import traceback

def handle_exceptions(logger_function):
    """
    Decorator that wraps a function to catch exceptions and log detailed error messages.
    
    Args:
        logger_function (callable): Function to call with the error message.
    
    Returns:
        callable: The wrapped function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_message = f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
                logger_function(error_message)
                raise
        return wrapper
    return decorator
