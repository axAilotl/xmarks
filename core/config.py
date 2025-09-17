"""
Configuration management for XMarks
Simple configuration without external dependencies
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import os
import logging

logger = logging.getLogger(__name__)


def load_env_file(env_path: str = '.env'):
    """Load environment variables from .env file if it exists"""
    env_file = Path(env_path)
    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse KEY=VALUE format
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        
                        # Set environment variable if not already set
                        if key and not os.getenv(key):
                            os.environ[key] = value
            
            logger.debug(f"Loaded environment variables from {env_path}")
        except Exception as e:
            logger.warning(f"Could not load .env file {env_path}: {e}")
    else:
        logger.debug(f"No .env file found at {env_path}")


# Load environment variables from .env file on module import
load_env_file()


class Config:
    """Simple configuration manager"""
    
    def __init__(self):
        # All defaults should live in config.json. Keep in-memory store minimal
        # and populate from file, applying compatibility aliases afterwards.
        self.data = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        data = self.data
        
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        
        data[keys[-1]] = value
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                self._merge_config(self.data, file_config)
                # Backward compatibility: expose selected paths at top-level
                # so existing calls like config.get('vault_dir') keep working.
                self._apply_legacy_path_aliases()
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
    
    def _merge_config(self, base: Dict, override: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _apply_legacy_path_aliases(self):
        """Lift values from paths.* to top-level keys for backward compatibility."""
        paths = self.get('paths', {})
        if isinstance(paths, dict):
            for key in ['bookmarks_file', 'cookies_file', 'cache_dir', 'media_dir', 'vault_dir']:
                if key not in self.data and key in paths:
                    self.data[key] = paths[key]
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check required files exist
        required_files = ['bookmarks_file', 'cookies_file']
        for file_key in required_files:
            value = self.get(file_key) or self.get(f'paths.{file_key}')
            if not value:
                errors.append(f"Required path not configured: {file_key} (set 'paths.{file_key}' in config.json)")
                continue
            file_path = Path(value)
            if not file_path.exists():
                errors.append(f"Required file missing: {file_path} (config key: {file_key})")
        
        # Check directories can be created
        required_dirs = ['cache_dir', 'vault_dir', 'media_dir']
        for dir_key in required_dirs:
            value = self.get(dir_key) or self.get(f'paths.{dir_key}')
            if not value:
                errors.append(f"Required directory path not configured: {dir_key} (set 'paths.{dir_key}' in config.json)")
                continue
            dir_path = Path(value)
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                if not dir_path.exists():
                    errors.append(f"Cannot create directory: {dir_path} (config key: {dir_key})")
            except Exception as e:
                errors.append(f"Cannot create directory {dir_path}: {e} (config key: {dir_key})")
        
        # Validate rate limiting configuration
        rate_limit_config = self.get('rate_limit')
        if isinstance(rate_limit_config, dict):
            if rate_limit_config.get('requests_per_window', 0) <= 0:
                errors.append("rate_limit.requests_per_window must be positive")
            if rate_limit_config.get('window_duration', 0) <= 0:
                errors.append("rate_limit.window_duration must be positive")
        
        # Check environment variables for LLM providers if enabled
        llm_config = self.get('llm', {})
        env_var_mapping = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API',  # Using actual .env variable name
            'openrouter': 'OPEN_ROUTER_API_KEY'  # Using actual .env variable name
        }
        
        for provider in ['openai', 'anthropic', 'openrouter']:
            provider_config = llm_config.get(provider, {})
            if provider_config.get('enabled', False):
                env_var = env_var_mapping[provider]
                env_value = os.getenv(env_var)
                if not env_value or env_value.strip() == '':
                    errors.append(f"LLM provider {provider} is enabled but {env_var} environment variable is not set or empty")
        
        # Check YouTube API key if YouTube features are enabled
        youtube_config = self.get('youtube', {})
        if youtube_config.get('enable_embeddings', False) or youtube_config.get('enable_transcripts', False):
            youtube_api_key = os.getenv('YOUTUBE_API_KEY')
            if not youtube_api_key or youtube_api_key.strip() == '':
                errors.append("YouTube features are enabled but YOUTUBE_API_KEY environment variable is not set or empty")
        
        return errors
    
    def is_pipeline_stage_enabled(self, stage_path: str) -> bool:
        """Check if a pipeline stage is enabled using dot notation (e.g., 'documents.arxiv_papers')"""
        return self.get(f'pipeline.stages.{stage_path}', True)
    
    def get_processing_threshold(self, threshold_name: str, default: Any = None) -> Any:
        """Get processing threshold values (e.g., 'summary_min_chars', 'alt_text_delay_seconds')"""
        return self.get(f'processing.{threshold_name}', default)
    
    def get_download_setting(self, setting_name: str, default: Any = None) -> Any:
        """Get download configuration values (e.g., 'timeout_seconds', 'retry_attempts')"""
        return self.get(f'downloads.{setting_name}', default)
    
    def get_naming_pattern(self, pattern_type: str) -> str:
        """Get file naming pattern for a specific type (e.g., 'tweet', 'thread', 'media')"""
        return self.get(f'files.naming_patterns.{pattern_type}', '')
    
    def validate_and_warn(self) -> bool:
        """Validate configuration and log warnings for any issues. Returns True if valid."""
        errors = self.validate()
        
        if errors:
            logger.warning(f"Configuration validation found {len(errors)} issues:")
            for error in errors:
                logger.warning(f"  - {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True


# Global config instance
config = Config()

# Try to load from config file if it exists
config.load_from_file('config.json')