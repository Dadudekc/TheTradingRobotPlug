# Scripts/tests/test_data_fetch_core.py

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, date
import re
from aioresponses import aioresponses  
import os
from Scripts.Utilities.data_fetch_utils import DataFetchUtils
from pandas.errors import ParserError

