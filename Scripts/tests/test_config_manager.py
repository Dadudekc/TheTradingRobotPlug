# Scripts/tests/test_config_manager.py

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from Scripts.Utilities.config_manager import ConfigManager

# Sample configuration data
YAML_CONFIG = """
database:
  user: test_user
  password: test_pass
  host: test_host
  port: 5432
  dbname: test_db
features:
  feature_list: open,high,low,close,volume
  target: close
debug_mode: true
"""

JSON_CONFIG = """
{
    "database": {
        "user": "json_user",
        "password": "json_pass",
        "host": "json_host",
        "port": 3306,
        "dbname": "json_db"
    },
    "features": {
        "feature_list": "open,high,low,close,volume",
        "target": "close"
    },
    "debug_mode": false
}
"""

TOML_CONFIG = """
[database]
user = "toml_user"
password = "toml_pass"
host = "toml_host"
port = 27017
dbname = "toml_db"

[features]
feature_list = "open,high,low,close,volume"
target = "close"

debug_mode = true
"""

ENV_CONTENT = """
DATABASE_USER=env_user
DATABASE_PASSWORD=env_pass
DATABASE_HOST=env_host
DATABASE_PORT=6543
DATABASE_DBNAME=env_db
FEATURES_TARGET=close
DEBUG_MODE=true
"""

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def temp_config_files():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        yaml_file = tmpdir / "config.yaml"
        json_file = tmpdir / "config.json"
        toml_file = tmpdir / "config.toml"
        env_file = tmpdir / ".env"

        yaml_file.write_text(YAML_CONFIG)
        json_file.write_text(JSON_CONFIG)
        toml_file.write_text(TOML_CONFIG)
        env_file.write_text(ENV_CONTENT)

        yield {
            "yaml": yaml_file,
            "json": json_file,
            "toml": toml_file,
            "env": env_file
        }

@pytest.fixture
def project_root():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)

def test_load_yaml_config(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["yaml"]]
    config_manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        required_keys=[],
        logger=mock_logger,
        project_root=project_root
    )

    assert config_manager.get("database.user") == "env_user"  # Env variable overrides config file
    assert config_manager.get("database.password") == "env_pass"
    assert config_manager.get("database.host") == "env_host"
    assert config_manager.get("database.port", value_type=int) == 6543
    assert config_manager.get("database.dbname") == "env_db"
    assert config_manager.get("features.feature_list", value_type=list) == ["open", "high", "low", "close", "volume"]
    assert config_manager.get("features.target") == "close"
    assert config_manager.get("debug_mode", value_type=bool) is True

def test_load_json_config(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["json"]]
    config_manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        required_keys=[],
        logger=mock_logger,
        project_root=project_root
    )

    assert config_manager.get("database.user") == "env_user"
    assert config_manager.get("database.password") == "env_pass"
    assert config_manager.get("database.host") == "env_host"
    assert config_manager.get("database.port", value_type=int) == 6543
    assert config_manager.get("database.dbname") == "env_db"
    assert config_manager.get("features.feature_list", value_type=list) == ["open", "high", "low", "close", "volume"]
    assert config_manager.get("features.target") == "close"
    assert config_manager.get("debug_mode", value_type=bool) is True

def test_load_toml_config(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["toml"]]
    config_manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        required_keys=[],
        logger=mock_logger,
        project_root=project_root
    )

    assert config_manager.get("database.user") == "env_user"
    assert config_manager.get("database.password") == "env_pass"
    assert config_manager.get("database.host") == "env_host"
    assert config_manager.get("database.port", value_type=int) == 6543
    assert config_manager.get("database.dbname") == "env_db"
    assert config_manager.get("features.feature_list", value_type=list) == ["open", "high", "low", "close", "volume"]
    assert config_manager.get("features.target") == "close"
    assert config_manager.get("debug_mode", value_type=bool) is True

def test_type_casting(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["yaml"]]
    config_manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        required_keys=[],
        logger=mock_logger,
        project_root=project_root
    )

    # Boolean casting
    assert config_manager.get("debug_mode", value_type=bool) is True

    # List casting
    features = config_manager.get("features.feature_list", value_type=list)
    assert features == ["open", "high", "low", "close", "volume"]

    # Integer casting
    assert config_manager.get("database.port", value_type=int) == 6543

def test_required_keys(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["yaml"]]
    required_keys = [
        "database.user",
        "database.password",
        "database.host",
        "database.port",
        "database.dbname",
        "features.target"
    ]

    config_manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        required_keys=required_keys,
        logger=mock_logger,
        project_root=project_root
    )

    # All required keys are present, no exception should be raised
    assert config_manager.get("database.user") == "env_user"

    # Now test with a missing key
    required_keys_missing = ["database.user", "database.password", "missing.key"]
    with pytest.raises(KeyError) as exc_info:
        ConfigManager(
            config_files=config_files,
            env_file=temp_config_files["env"],
            required_keys=required_keys_missing,
            logger=mock_logger,
            project_root=project_root
        )
    # Update the assertion to match the uppercase error message
    assert "MISSING.KEY" in str(exc_info.value)

def test_reload_configurations(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["yaml"]]
    config_manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        required_keys=[],
        logger=mock_logger,
        project_root=project_root
    )

    # Initial value
    assert config_manager.get("database.user") == "env_user"

    # Modify the .env file to change DATABASE_USER
    new_env_content = """
    DATABASE_USER=new_env_user
    DATABASE_PASSWORD=new_env_pass
    DATABASE_HOST=new_env_host
    DATABASE_PORT=7654
    DATABASE_DBNAME=new_env_db
    FEATURES_TARGET=new_close
    DEBUG_MODE=false
    """
    temp_config_files["env"].write_text(new_env_content)

    # Reload configurations
    config_manager.reload_configurations()

    # Check updated values
    assert config_manager.get("database.user") == "new_env_user"
    assert config_manager.get("database.password") == "new_env_pass"
    assert config_manager.get("database.host") == "new_env_host"
    assert config_manager.get("database.port", value_type=int) == 7654
    assert config_manager.get("database.dbname") == "new_env_db"
    assert config_manager.get("features.target") == "new_close"
    assert config_manager.get("debug_mode", value_type=bool) is False

def test_list_configurations(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["yaml"]]
    config_manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        required_keys=[],
        logger=mock_logger,
        project_root=project_root
    )

    # Pass mask_keys in lowercase to match ConfigManager's storage
    all_configs = config_manager.list_configurations(mask_keys=["database.password", "debug_mode"])

    # Access configurations using lowercase keys
    assert all_configs["database.user"] == "env_user"
    assert all_configs["database.password"] == "*****"
    assert all_configs["debug_mode"] == "*****"

def test_get_db_urls(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["yaml"]]
    required_keys = [
        "database.user",
        "database.password",
        "database.host",
        "database.port",
        "database.dbname"
    ]
    config_manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        required_keys=required_keys,
        logger=mock_logger,
        project_root=project_root
    )

    db_url = config_manager.get_db_url()
    expected_db_url = "postgresql://env_user:env_pass@env_host:6543/env_db"
    assert db_url == expected_db_url

    # Similarly, define and test get_async_db_url if implemented
