"""
Configuration management utilities for Knative services and autoscaler settings.
Handles reading, writing, and manipulating service configurations.
"""

import yaml
import os
import glob
import shutil
from config import PATH


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


def get_knative_knobs(service_config, auto_config=None):
    """
    Extract Knative autoscaling annotations and requested CPU from a service config.
    
    Args:
        service_config: Service configuration dictionary
        auto_config: Autoscaler configuration dictionary (optional)
        
    Returns:
        dict: Knative configuration knobs
    """
    knobs = {}
    
    # Extract autoscaling annotations
    try:
        annotations = service_config['spec']['template']['metadata']['annotations']
        for k, v in annotations.items():
            if k.startswith('autoscaling.knative.dev/'):
                knobs[k] = v
    except Exception:
        pass

    # Extract requested CPU
    try:
        containers = service_config['spec']['template']['spec']['containers']
        if containers and 'resources' in containers[0] and 'requests' in containers[0]['resources']:
            cpu = containers[0]['resources']['requests'].get('cpu')
            if cpu:
                knobs['requested_cpu'] = cpu
    except Exception:
        pass

    return knobs


def set_knative_knobs(file_path, knob_values):
    """
    Update Knative autoscaling annotations and requested CPU in a service YAML file.
    
    Args:
        file_path: Path to the service YAML file
        knob_values: Dictionary of {knob_name: value}
    """
    with open(file_path, 'r') as f:
        docs = list(yaml.safe_load_all(f))
    
    # Assume first doc is the service config
    service = docs[0]
    annotations = service['spec']['template']['metadata'].setdefault('annotations', {})
    
    for k, v in knob_values.items():
        if k.startswith('autoscaling.knative.dev/'):
            annotations[k] = v
        elif k == 'requested_cpu':
            containers = service['spec']['template']['spec']['containers']
            if containers:
                containers[0].setdefault('resources', {}).setdefault('requests', {})['cpu'] = v
    
    docs[0] = service
    with open(file_path, 'w') as f:
        yaml.dump_all(docs, f)


def get_vscaling_knobs(service_config, auto_config):
    """
    Extract all relevant vscaling/autoscaler keys from the config-autoscaler.yaml.
    
    Args:
        service_config: Service configuration dictionary
        auto_config: Autoscaler configuration dictionary
        
    Returns:
        dict: VScaling configuration knobs
    """
    knobs = {}
    try:
        if 'data' in auto_config:
            for k, v in auto_config['data'].items():
                knobs[k] = v
    except Exception:
        pass
    return knobs


def set_vscaling_knobs(file_path, knob_values):
    """
    Update vscaling/autoscaler keys in the config file.
    
    Args:
        file_path: Path to the autoscaler config file
        knob_values: Dictionary of {knob_name: value}
    """
    with open(file_path, 'r') as f:
        docs = list(yaml.safe_load_all(f))
    
    # Assume first doc is the configmap
    config = docs[0]
    data = config.setdefault('data', {})
    
    for k, v in knob_values.items():
        data[k] = v
    
    docs[0] = config
    with open(file_path, 'w') as f:
        yaml.dump_all(docs, f)


def setup_experiment_directory(label):
    """
    Set up the experiment directory structure and copy base configurations.
    
    Args:
        label: Experiment label
    """
    os.makedirs(PATH + f"/output/{label}", exist_ok=True)
    
    if not glob.glob(PATH + f"/output/{label}/config/*.yaml"):
        shutil.copytree(PATH + "/base_config", PATH + f"/output/{label}/config", dirs_exist_ok=True)
        print("Base configuration files copied")
    else:
        print("Base configuration files already exist... skipping")
