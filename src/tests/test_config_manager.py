import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from collections import defaultdict  # <-- Import for test_config_manager_no_files
from dotenv import load_dotenv  # For environment reload if needed

from src.Utilities.config_manager import ConfigManager

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


# -----------------------------------------------------------------------------
# Basic loading tests (YAML, JSON, TOML)
# -----------------------------------------------------------------------------
def test_load_yaml_config(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["yaml"]]
    manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        required_keys=[],
        logger=mock_logger,
        project_root=project_root
    )
    # .env overrides YAML => env_user
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
    # .env overrides JSON => env_user
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
    # .env overrides TOML => env_user
    assert manager.get("database.user") == "env_user"
    assert manager.get("database.password") == "env_pass"
    assert manager.get("database.host") == "env_host"
    assert manager.get("database.port", value_type=int) == 6543
    assert manager.get("database.dbname") == "env_db"
    assert manager.get("features.feature_list", value_type=list) == ["open", "high", "low", "close", "volume"]
    assert manager.get("features.target") == "close"
    assert manager.get("debug_mode", value_type=bool) is True


# -----------------------------------------------------------------------------
# Type casting, required keys, reloading
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# list_configurations, unsupported file, malformed YAML
# -----------------------------------------------------------------------------
def test_list_configurations(mock_logger, temp_config_files, project_root):
    config_files = [temp_config_files["yaml"]]
    manager = ConfigManager(
        config_files=config_files,
        env_file=temp_config_files["env"],
        logger=mock_logger,
        project_root=project_root
    )
    all_configs = manager.list_configurations(mask_keys=["database.password", "debug_mode"])
    assert all_configs["database.password"] == "*****"
    assert all_configs["debug_mode"] == "*****"
    assert all_configs["database.user"] == "env_user"

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

def test_malformed_yaml_or_toml(mock_logger, temp_config_files, project_root):
    """
    Combined test for malformed YAML or TOML to reduce redundancy.
    """
    malformed_yaml = temp_config_files["yaml"]
    malformed_yaml.write_text("database: {user: unbalanced_braces")

    ConfigManager(
        config_files=[malformed_yaml],
        logger=mock_logger,
        project_root=project_root
    )
    mock_logger.error.assert_called_once()
    assert "Error loading" in mock_logger.error.call_args[0][0]


# -----------------------------------------------------------------------------
# Tests specifically checking environment removal or missing env keys
# -----------------------------------------------------------------------------
def test_missing_env_variable(mock_logger, temp_config_files, project_root):
    os.environ.pop("DATABASE_USER", None)
    no_user_env_content = """
    DATABASE_PASSWORD=env_pass
    DATABASE_HOST=env_host
    DATABASE_PORT=6543
    DATABASE_DBNAME=env_db
    FEATURES_TARGET=close
    DEBUG_MODE=true
    """
    temp_config_files["env"].write_text(no_user_env_content)
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
    manager = ConfigManager(
        config_files=[temp_config_files["yaml"]],
        env_file=temp_config_files["env"],
        logger=mock_logger,
        project_root=project_root,
        strict_keys=True
    )
    with pytest.raises(ValueError, match="Configuration for 'DATABASE_USER' is required but not provided."):
        manager.get("database.user", required=True)

def test_get_db_urls_missing_config(mock_logger, temp_config_files, project_root):
    os.environ.pop("DATABASE_HOST", None)
    no_host_env_content = """
    DATABASE_USER=env_user
    DATABASE_PASSWORD=env_pass
    DATABASE_PORT=6543
    DATABASE_DBNAME=env_db
    FEATURES_TARGET=close
    DEBUG_MODE=true
    """
    temp_config_files["env"].write_text(no_host_env_content)
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
    load_dotenv(dotenv_path=temp_config_files["env"], override=True)

    manager = ConfigManager(
        config_files=[temp_config_files["yaml"]],
        env_file=temp_config_files["env"],
        logger=mock_logger,
        project_root=project_root,
        strict_keys=True
    )
    manager.cache.clear()
    # Confirm 'database.host' is missing
    assert "database.host" not in manager.config
    with pytest.raises(ValueError, match="Configuration for 'DATABASE.HOST' is required but not provided."):
        manager.get_db_url()


# -----------------------------------------------------------------------------
# Additional tests for type casting, invalid env, strict keys, flatten dict
# -----------------------------------------------------------------------------
def test_get_with_invalid_type(mock_logger, temp_config_files, project_root):
    manager = ConfigManager(
        config_files=[temp_config_files["yaml"]],
        logger=mock_logger,
        project_root=project_root
    )
    with pytest.raises(TypeError, match="Failed to cast config key 'DATABASE.USER' to <class 'dict'>"):
        manager.get("database.user", value_type=dict)

def test_str_to_bool(mock_logger, project_root):
    manager = ConfigManager(logger=mock_logger, project_root=project_root)
    assert manager._str_to_bool("true") is True
    assert manager._str_to_bool("false") is False
    assert manager._str_to_bool(1) is True
    assert manager._str_to_bool(0) is False
    with pytest.raises(ValueError, match="Cannot convert maybe to bool"):
        manager._str_to_bool("maybe")

def test_invalid_env_file(mock_logger, project_root):
    invalid_env_path = project_root / "non_existent.env"
    manager = ConfigManager(env_file=invalid_env_path, logger=mock_logger, project_root=project_root)
    mock_logger.warning.assert_called_with(
        f"No .env file found at {invalid_env_path}. Environment variables will be used as-is."
    )

def test_get_missing_key_with_strict_keys(mock_logger, temp_config_files, project_root):
    manager = ConfigManager(
        config_files=[temp_config_files["yaml"]],
        logger=mock_logger,
        project_root=project_root,
        strict_keys=True
    )
    with pytest.raises(ValueError, match="Configuration for 'DATABASE.MISSING_KEY' is required but not provided."):
        manager.get("database.missing_key", required=True)

def test_get_with_fallback_value(mock_logger, temp_config_files, project_root):
    manager = ConfigManager(
        config_files=[temp_config_files["yaml"]],
        logger=mock_logger,
        project_root=project_root
    )
    value = manager.get("non.existent.key", default="fallback_value")
    assert value == "fallback_value"

def test_flatten_dict(mock_logger, project_root):
    nested_dict = {
        "level1": {
            "level2": {
                "level3": {
                    "key": "value"
                }
            },
            "another_key": "value2"
        }
    }
    manager = ConfigManager(logger=mock_logger, project_root=project_root)
    flattened = manager._flatten_dict(nested_dict)
    assert flattened == {
        "level1.level2.level3.key": "value",
        "level1.another_key": "value2"
    }

def test_check_missing_keys(mock_logger, temp_config_files, project_root):
    required_keys = ["database.user", "database.password", "database.missing_key"]
    manager = ConfigManager(
        config_files=[temp_config_files["yaml"]],
        logger=mock_logger,
        project_root=project_root
    )
    with pytest.raises(KeyError, match="Missing required configuration keys: DATABASE.MISSING_KEY"):
        manager.check_missing_keys(required_keys)


# -----------------------------------------------------------------------------
#  test_loading_multiple_files => set config_overrides_env=True
# -----------------------------------------------------------------------------
def test_loading_multiple_files(mock_logger, temp_config_files, project_root):
    """
    Test configuration precedence with multiple files.
    Using config_overrides_env=True so that YAML can override ENV.
    """
    config_files = [temp_config_files["json"], temp_config_files["yaml"]]
    manager = ConfigManager(
        config_files=config_files,
        logger=mock_logger,
        project_root=project_root,
        config_overrides_env=True  # <-- So YAML overrides env
    )
    # Now YAML can override JSON & env => test_user
    assert manager.get("database.user") == "test_user"


# -----------------------------------------------------------------------------
#  test_get_db_url => set config_overrides_env=True or remove .env user
# -----------------------------------------------------------------------------
def test_get_db_url(mock_logger, temp_config_files, project_root):
    """
    Test construction of DB URL with config_overrides_env=True so YAML wins.
    """
    manager = ConfigManager(
        config_files=[temp_config_files["yaml"]],
        env_file=temp_config_files["env"],
        logger=mock_logger,
        project_root=project_root,
        config_overrides_env=True  # <-- So YAML overrides .env
    )
    db_url = manager.get_db_url()
    expected_url = "postgresql://test_user:test_pass@test_host:5432/test_db"
    assert db_url == expected_url


# -----------------------------------------------------------------------------
# test_config_manager_no_files => import defaultdict OR revise assertion
# -----------------------------------------------------------------------------
def test_config_manager_no_files(mock_logger, project_root):
    """Test ConfigManager initialization with no config files."""
    manager = ConfigManager(logger=mock_logger, project_root=project_root)
    
    # Ensure config is a defaultdict(dict) and empty
    assert isinstance(manager.config, defaultdict)
    assert len(manager.config) == 0  # Config should be empty
    
    # Check if the warning is logged
    warnings = [call.args[0] for call in mock_logger.warning.call_args_list]
    assert any("No .env file found at" in warning for warning in warnings)

def test_malformed_toml(mock_logger, temp_config_files, project_root):
    malformed_toml = temp_config_files["toml"]
    malformed_toml.write_text("database = { user = 'missing_quote }")
    manager = ConfigManager(config_files=[malformed_toml], logger=mock_logger, project_root=project_root)
    
    mock_logger.error.assert_called_once()
    assert "Error loading" in mock_logger.error.call_args[0][0]

def test_config_overrides_env(mock_logger, temp_config_files, project_root):
    manager = ConfigManager(
        config_files=[temp_config_files["yaml"]],
        env_file=temp_config_files["env"],
        logger=mock_logger,
        project_root=project_root,
        config_overrides_env=True
    )
    assert manager.get("database.user") == "test_user"

def test_missing_keys_with_strict(mock_logger, temp_config_files, project_root):
    # Remove DATABASE_HOST from .env
    no_host_env_content = """
    DATABASE_USER=env_user
    DATABASE_PASSWORD=env_pass
    DATABASE_PORT=6543
    DATABASE_DBNAME=env_db
    FEATURES_TARGET=close
    DEBUG_MODE=true
    """
    temp_config_files["env"].write_text(no_host_env_content)

    # Remove DATABASE_HOST from YAML
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

    # Remove DATABASE_HOST from the OS environment
    os.environ.pop("DATABASE_HOST", None)

    # Reload environment to ensure DATABASE_HOST is gone
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=temp_config_files["env"], override=True)

    manager = ConfigManager(
        config_files=[temp_config_files["yaml"]],
        env_file=temp_config_files["env"],
        logger=mock_logger,
        project_root=project_root,
        strict_keys=True
    )

    # Assert ValueError is raised for missing 'DATABASE.HOST'
    with pytest.raises(ValueError, match="Configuration for 'DATABASE.HOST' is required but not provided."):
        manager.get("database.host", required=True)
