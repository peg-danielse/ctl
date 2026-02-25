import glob
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

import yaml
import util.square as square

from config import PATH

CONFIGURATION_BASE_PATH = os.path.join(PATH, "base_configuration")

class ConfigManager:
    instance = None

    @staticmethod
    def get_instance(label: str):
        if ConfigManager.instance is None:
            ConfigManager.instance = ConfigManager(label)
        return ConfigManager.instance

    def __init__(self, label: str):
        # Each entry is a list of (applied_at, config). applied_at is None for configs loaded from file.
        self.configs: Dict[str, List[Tuple[Optional[datetime], Any]]] = {}
        self.client = square.get_k8s_api_client()
        self.label = label

        # Use per-experiment config directory if it already has YAMLs; otherwise
        # seed it from the shared base_configuration directory.
        experiment_config_dir = os.path.join(PATH, "output", self.label, "config")
        os.makedirs(experiment_config_dir, exist_ok=True)

        base_config_files = glob.glob(os.path.join(experiment_config_dir, "*.yaml"))
        if os.path.isdir(CONFIGURATION_BASE_PATH):
            if not base_config_files:
                logger.info("Seeding experiment configs from %s", CONFIGURATION_BASE_PATH)
            for config_file in glob.glob(os.path.join(CONFIGURATION_BASE_PATH, "*.yaml")):
                dest = os.path.join(experiment_config_dir, os.path.basename(config_file))
                if not os.path.exists(dest):
                    shutil.copy(config_file, dest)
                    logger.info("Added base config %s", os.path.basename(config_file))
            base_config_files = glob.glob(os.path.join(experiment_config_dir, "*.yaml"))

        if not base_config_files:
            logger.warning(
                "ConfigManager for label '%s' found no YAML configs in %s; "
                "service configs will be empty until set_service_config is called.",
                self.label,
                experiment_config_dir,
            )

        for config_file in base_config_files:
            service_name = os.path.basename(config_file).split(".")[0]
            logger.info("Loading base config for service '%s' from %s", service_name, config_file)
            self.load_config(config_file, service_name)

        # Only configs for these services may be applied later (no LLM-suggested new resources)
        self.base_service_names = set(self.configs.keys())
        # Map manifest metadata.name (e.g. memcached-rate) -> our base key (e.g. memcached-rate-deployment)
        self._metadata_name_to_base_key: Dict[str, str] = {}
        for base_key, config_list in self.configs.items():
            if config_list:
                meta_name = (config_list[0][1].get("metadata") or {}).get("name")
                if meta_name:
                    self._metadata_name_to_base_key[meta_name] = base_key

    def get_service_config(self, service_name: str) -> Any:
        """
        Return the latest stored config for a service.
        Falls back from manifest metadata.name to our base key, and returns an
        empty dict if no configuration is known instead of raising IndexError.
        """
        # Direct lookup by our internal key (e.g. memcached-rate-deployment)
        configs = self.configs.get(service_name)
        if configs:
            return configs[-1][1]

        # Fallback: lookup by manifest metadata.name (e.g. memcached-rate)
        canonical = self._metadata_name_to_base_key.get(service_name)
        if canonical:
            configs = self.configs.get(canonical)
            if configs:
                return configs[-1][1]

        logger.warning(
            "No configuration found for service '%s'; returning empty config for prompt generation.",
            service_name,
        )
        return {}

    def set_service_config(self, service_name: str, config: Any, applied_at: Optional[datetime] = None) -> bool:
        """Apply and store a configuration. Returns True if applied, False if skipped (e.g. not in base config).
        service_name may be our base key (e.g. memcached-rate-deployment) or the manifest metadata.name (e.g. memcached-rate).
        """
        canonical = (
            service_name if service_name in self.base_service_names
            else self._metadata_name_to_base_key.get(service_name)
        )
        if canonical is None:
            logger.warning(
                "Refusing to apply configuration for '%s': not in base configuration (base: %s)",
                service_name, sorted(self.base_service_names),
            )
            return False
        if canonical not in self.configs:
            self.configs[canonical] = []
        try:
            square.apply_yaml_configuration(config, self.client)
        except Exception as e:
            print(f"Error applying configuration for {canonical}: {e}")

        if applied_at is None:
            applied_at = datetime.now(timezone.utc)
        self.configs[canonical].append((applied_at, config))
        return True

    def load_config(self, filepath: str, service_name: str):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        # Loaded configs have no applied_at; use None so we don't write a timestamp for them
        if service_name not in self.configs:
            self.configs[service_name] = []
        try:
            square.apply_yaml_configuration(data, self.client)
        except Exception as e:
            print(f"Error applying configuration for {service_name}: {e}")
        self.configs[service_name].append((None, data))

    def reset_config(self):
        square.reset_k8s(self.client, CONFIGURATION_BASE_PATH)

    def save_all_configs(self):
        for service_name in self.configs:
            for i, (applied_at, config) in enumerate(self.configs[service_name]):
                config_path = os.path.join(PATH, "output", self.label, "config", f"{service_name}_{i}.yaml")
                with open(config_path, 'w') as f:
                    if applied_at is not None:
                        f.write(f"# applied_at: {applied_at.isoformat()}\n")
                    yaml.dump(config, f)
                if applied_at is not None:
                    meta_path = os.path.join(PATH, "output", self.label, "config", f"{service_name}_{i}.meta.json")
                    with open(meta_path, 'w') as f:
                        json.dump({"applied_at": applied_at.isoformat()}, f, indent=2)


# Legacy functions for backward compatibility
def load_yaml_as_string(filepath):
    """Load a YAML file and return as string."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return yaml.dump(data)


def load_yaml_as_dict(filepath):
    """Load a YAML file and return as dictionary."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data

def setup_experiment_directory(label):
    """Ensure output/<label>/config contains a base set of YAML configs.

    If configs already exist there, we leave them alone. Otherwise, we copy
    from CONFIGURATION_BASE_PATH if it exists.
    """
    experiment_config_dir = os.path.join(PATH, "output", label, "config")
    os.makedirs(experiment_config_dir, exist_ok=True)

    if glob.glob(os.path.join(experiment_config_dir, "*.yaml")):
        print("Base configuration files already exist... skipping")
        return

    if os.path.isdir(CONFIGURATION_BASE_PATH):
        shutil.copytree(CONFIGURATION_BASE_PATH, experiment_config_dir, dirs_exist_ok=True)
        print("Base configuration files copied")
    else:
        print(f"No base configuration directory at {CONFIGURATION_BASE_PATH}; not copying.")
