import glob
import os
import shutil
from typing import Dict, List, Any

import yaml
import util.square as square

from config import PATH

CONFIGURATION_BASE_PATH = PATH + "/base_config"

class ConfigManager:
    instance = None

    @staticmethod
    def get_instance(label: str):
        if ConfigManager.instance is None:
            ConfigManager.instance = ConfigManager(label)
        return ConfigManager.instance
    
    def __init__(self, label: str):
        self.configs: Dict[str, List[Any]] = {}
        self.client = square.get_k8s_api_client()
        self.label = label
        
        os.makedirs(PATH + f"/output/{self.label}/config", exist_ok=True)

        base_config_files = glob.glob(CONFIGURATION_BASE_PATH + "/*.yaml")

        for config_file in base_config_files:
            shutil.copy(config_file, PATH + f"/output/{self.label}/config/{config_file.split('/')[-1]}")
            self.load_config(config_file, config_file.split("/")[-1].split(".")[0])

    def get_service_config(self, service_name: str) -> Any:
        return self.configs.get(service_name, [])[-1]

    def set_service_config(self, service_name: str, config: Any):
        if service_name not in self.configs:
            self.configs[service_name] = []
        try:
            square.apply_yaml_configuration(config, self.client)
        except Exception as e:
            print(f"Error applying configuration for {service_name}: {e}")

        self.configs[service_name].append(config)

    def load_config(self, filepath: str, service_name: str):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        self.set_service_config(service_name, data)

    def reset_config(self):
        square.reset_k8s(self.client, CONFIGURATION_BASE_PATH)

    def save_all_configs(self):
        for service_name in self.configs:
            for i, config in enumerate(self.configs[service_name]):
                with open(PATH + f"/output/{self.label}/config/{service_name}_{i}.yaml", 'w') as f:
                    yaml.dump(config, f)


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
    os.makedirs(PATH + f"/output/{label}", exist_ok=True)
    
    if not glob.glob(PATH + f"/output/{label}/config/*.yaml"):
        shutil.copytree(PATH + "/base_config", PATH + f"/output/{label}/config", dirs_exist_ok=True)
        print("Base configuration files copied")
    else:
        print("Base configuration files already exist... skipping")
