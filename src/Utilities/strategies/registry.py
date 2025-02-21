"""
File: registry.py
Location: src/Utilities/strategies

Description:
    A registry for trading strategies.
    Strategies can register themselves via the provided decorator.
"""

# Global registry dictionary to store strategy name -> strategy class mappings.
STRATEGY_REGISTRY = {}

def register_strategy(name: str):
    """
    Decorator to register a strategy class with a unique name.

    Args:
        name (str): The unique identifier for the strategy.
    
    Returns:
        The original class after registration.
    """
    def decorator(cls):
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator
