"""
external/cloud_storage.py
-------------------------
Provides a minimal implementation for cloud upload.
Since Docker/cloud integration is not required, this module simply logs the action.
"""

import logging

def upload_model_to_cloud(file_path):
    """
    Dummy cloud upload function.
    
    Args:
        file_path (str): Path of the model file.
    
    Returns:
        None
    """
    logging.info(f"Cloud upload disabled. Model at {file_path} was not uploaded.")
    return None
