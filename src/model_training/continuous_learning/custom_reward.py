def custom_reward(trade_data):
    """
    Custom reward function for RL agent
    Args:
        trade_data (dict): Contains information like balance, portfolio value, etc.
    
    Returns:
        reward (float): The reward for the current step.
    """
    profit = trade_data['profit']
    drawdown = trade_data['drawdown']

    # Example: reward for profit with a penalty for drawdowns
    reward = profit - 0.5 * drawdown
    return reward
