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
    manager.reload_configurations(env_file=temp_config_files["env"])
    assert manager.get("database.user") == "new_env_user"

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

    # Validate masked keys
    assert all_configs["database.password"] == "*****"
    assert all_configs["debug_mode"] == "*****"

    # Validate other keys - `.env` should override the YAML file
    assert all_configs["database.user"] == "env_user"

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

def test_list_configurations_no_mask(mock_logger, temp_config_files, project_root):
    manager = ConfigManager(
        config_files=[temp_config_files["yaml"]],
        env_file=temp_config_files["env"],
        logger=mock_logger,
        project_root=project_root
    )

    all_configs = manager.list_configurations()
    assert all_configs["database.user"] == "env_user"
    assert all_configs["database.password"] == "env_pass"

def test_unsupported_file_format(mock_logger, temp_config_files, project_root):
    unsupported_file = temp_config_files["yaml"].with_suffix(".txt")
    unsupported_file.write_text("Unsupported format content")

    manager = ConfigManager(
        config_files=[unsupported_file],
        logger=mock_logger,
        project_root=project_root
    )

    mock_logger.warning.assert_called_with(f"Unsupported config file format: {unsupported_file}")

def test_malformed_yaml(mock_logger, temp_config_files, project_root):
    malformed_yaml = temp_config_files["yaml"]
    malformed_yaml.write_text("database: {user: unbalanced_braces")

    ConfigManager(
        config_files=[malformed_yaml],
        logger=mock_logger,
        project_root=project_root
    )

    mock_logger.error.assert_called_once()
    assert "Error loading" in mock_logger.error.call_args[0][0]
    assert "while parsing a flow mapping" in mock_logger.error.call_args[0][0]


# ----------------------------------------------------------------------------
#  Overwrite .env so that DATABASE_USER is truly missing
# ----------------------------------------------------------------------------
def test_missing_env_variable(mock_logger, temp_config_files, project_root):
    # Remove DATABASE_USER from OS environment
    os.environ.pop("DATABASE_USER", None)

    # Overwrite the .env to remove DATABASE_USER entirely
    no_user_env_content = """
    DATABASE_PASSWORD=env_pass
    DATABASE_HOST=env_host
    DATABASE_PORT=6543
    DATABASE_DBNAME=env_db
    FEATURES_TARGET=close
    DEBUG_MODE=true
    """
    temp_config_files["env"].write_text(no_user_env_content)

    # Overwrite the YAML file to remove database.user
    no_user_yaml_content = """
    database:
      password: test_pass
      host: test_host
      port: 5432
      dbname: test_db
    features:
      feature_list: open,high,low,close,volume
      target: close
    debug_mode: true
    """
    temp_config_files["yaml"].write_text(no_user_yaml_content)

    # Reinitialize ConfigManager to ensure changes take effect
    manager = ConfigManager(
        config_files=[temp_config_files["yaml"]],
        env_file=temp_config_files["env"],
        logger=mock_logger,
        project_root=project_root,
        strict_keys=True
    )

    # Now there's no "DATABASE_USER" anywhere, so it should raise:
    with pytest.raises(ValueError, match="Configuration for 'DATABASE_USER' is required but not provided."):
        manager.get("database.user", required=True)

# ----------------------------------------------------------------------------
#  Overwrite .env so that DATABASE_HOST is truly missing
# ----------------------------------------------------------------------------
def test_get_db_urls_missing_config(mock_logger, temp_config_files, project_root):
    # Remove DATABASE_HOST from the OS environment
    os.environ.pop("DATABASE_HOST", None)

    # Overwrite the .env to remove DATABASE_HOST entirely
    no_host_env_content = """
    DATABASE_USER=env_user
    DATABASE_PASSWORD=env_pass
    DATABASE_PORT=6543
    DATABASE_DBNAME=env_db
    FEATURES_TARGET=close
    DEBUG_MODE=true
    """
    temp_config_files["env"].write_text(no_host_env_content)

    # Overwrite the YAML file to remove database.host
    no_host_yaml_content = """
    database:
      user: test_user
      password: test_pass
      port: 5432
      dbname: test_db
    features:
      feature_list: open,high,low,close,volume
      target: close
    debug_mode: true
    """
    temp_config_files["yaml"].write_text(no_host_yaml_content)

    # Reload the environment to ensure DATABASE_HOST is gone
    os.environ.pop("DATABASE_HOST", None)  # Remove from the current environment
    from dotenv import load_dotenv  # Reimport to refresh
    load_dotenv(dotenv_path=temp_config_files["env"], override=True)

    # Initialize ConfigManager
    config_files = [temp_config_files["yaml"]]
    manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        logger=mock_logger,
        project_root=project_root,
        strict_keys=True
    )

    # Clear cache and print configurations for debugging
    manager.cache.clear()
    print("Loaded configurations:", manager.list_configurations())

    # Ensure 'database.host' is truly missing
    assert "database.host" not in manager.config

    # Now no "DATABASE_HOST" anywhere, so it should raise:
    with pytest.raises(ValueError, match="Configuration for 'DATABASE.HOST' is required but not provided."):
        manager.get_db_url()
