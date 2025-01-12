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
    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
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
    with tempfile.TemporaryDirectory() as tmpdir_name:
        yield Path(tmpdir_name)

def test_load_yaml_config(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["yaml"]]
    manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        required_keys=[],
        logger=mock_logger,
        project_root=project_root
    )

    assert manager.get("database.user") == "env_user"
    assert manager.get("database.password") == "env_pass"
    assert manager.get("database.host") == "env_host"
    assert manager.get("database.port", value_type=int) == 6543
    assert manager.get("database.dbname") == "env_db"
    assert manager.get("features.feature_list", value_type=list) == ["open", "high", "low", "close", "volume"]
    assert manager.get("features.target") == "close"
    assert manager.get("debug_mode", value_type=bool) is True

def test_load_json_config(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["json"]]
    manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        logger=mock_logger,
        project_root=project_root
    )

    assert manager.get("database.user") == "env_user"
    assert manager.get("database.password") == "env_pass"
    assert manager.get("database.host") == "env_host"
    assert manager.get("database.port", value_type=int) == 6543
    assert manager.get("database.dbname") == "env_db"
    assert manager.get("features.feature_list", value_type=list) == ["open", "high", "low", "close", "volume"]
    assert manager.get("features.target") == "close"
    assert manager.get("debug_mode", value_type=bool) is True

def test_load_toml_config(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["toml"]]
    manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        logger=mock_logger,
        project_root=project_root
    )

    assert manager.get("database.user") == "env_user"
    assert manager.get("database.password") == "env_pass"
    assert manager.get("database.host") == "env_host"
    assert manager.get("database.port", value_type=int) == 6543
    assert manager.get("database.dbname") == "env_db"
    assert manager.get("features.feature_list", value_type=list) == ["open", "high", "low", "close", "volume"]
    assert manager.get("features.target") == "close"
    assert manager.get("debug_mode", value_type=bool) is True

def test_type_casting(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["yaml"]]
    manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        logger=mock_logger,
        project_root=project_root
    )

    assert manager.get("debug_mode", value_type=bool) is True
    features = manager.get("features.feature_list", value_type=list)
    assert features == ["open", "high", "low", "close", "volume"]
    assert manager.get("database.port", value_type=int) == 6543

def test_required_keys(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["yaml"]]
    required = ["database.user", "database.password", "database.host", "database.port", "database.dbname", "features.target"]
    manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        required_keys=required,
        logger=mock_logger,
        project_root=project_root
    )

    # Should pass with no exception
    assert manager.get("database.user") == "env_user"

    # Check missing keys
    required_missing = ["database.user", "database.password", "missing.key"]
    with pytest.raises(KeyError) as exc_info:
        ConfigManager(
            config_files=config_files,
            env_file=temp_config_files["env"],
            required_keys=required_missing,
            logger=mock_logger,
            project_root=project_root
        )
    assert "MISSING.KEY" in str(exc_info.value)

def test_reload_configurations(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["yaml"]]
    manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        logger=mock_logger,
        project_root=project_root
    )

    assert manager.get("database.user") == "env_user"

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

    # Reload
    manager.reload_configurations()

    assert manager.get("database.user") == "new_env_user"
    assert manager.get("database.password") == "new_env_pass"
    assert manager.get("database.host") == "new_env_host"
    assert manager.get("database.port", value_type=int) == 7654
    assert manager.get("database.dbname") == "new_env_db"
    assert manager.get("features.target") == "new_close"
    assert manager.get("debug_mode", value_type=bool) is False

def test_list_configurations(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["yaml"]]
    manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        logger=mock_logger,
        project_root=project_root
    )

    # Request masking
    all_configs = manager.list_configurations(mask_keys=["database.password", "debug_mode"])
    assert all_configs["database.user"] == "env_user"
    # Password should be masked
    assert all_configs["database.password"] == "*****"
    # debug_mode should be masked
    assert all_configs["debug_mode"] == "*****"

def test_get_db_urls(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["yaml"]]
    required = ["database.user", "database.password", "database.host", "database.port", "database.dbname"]
    manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        required_keys=required,
        logger=mock_logger,
        project_root=project_root
    )

    db_url = manager.get_db_url()
    expected_db_url = "postgresql://env_user:env_pass@env_host:6543/env_db"
    assert db_url == expected_db_url

    async_db_url = manager.get_async_db_url()
    expected_async_url = "postgresql+asyncpg://env_user:env_pass@env_host:6543/env_db"
    assert async_db_url == expected_async_url
