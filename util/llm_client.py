"""
LLM client utilities for configuration generation and analysis.
Handles communication with the LLM API for generating configuration changes.
"""

import re
import requests
import yaml

from config import GEN_API_URL, OPENAI_API_KEY, OPENAI_MODEL, PATH
from prompt import GENERATE_PROMPT
from util.config_manager import load_yaml_as_string

def generate_prompt(service_name, trace_df, metric_dfs, anomaly_index, 
                   anomalies, knobs, label):
    """
    Generate the prompt for configuration generation.
    
    Args:
        service_name: Name of the service with the anomaly
        trace_df: Trace data DataFrame
        metric_dfs: Metrics data dictionary
        anomaly_index: Index of the anomaly in trace data
        anomalies: List of all anomalies
        knobs: Current configuration knobs
        label: Experiment label
        
    Returns:
        tuple: (timestamp, duration, snapshot, prompt)
    """
    from util.analysis import metric_snapshot
    
    # Create metric snapshot for the anomaly
    timestamp, duration, snapshot = metric_snapshot(
        service_name, trace_df, metric_dfs, anomaly_index, anomalies, knobs,
        phase="adaptation", subphase="configuration_application"
    )

    # Load service and autoscaler configurations
    service_config = load_yaml_as_string(PATH + f"/output/{label}/config/{service_name}.yaml")
    auto_config = load_yaml_as_string(PATH + f"/output/{label}/config/config-autoscaler.yaml")

    # Prepare prompt for configuration generation
    prompt = GENERATE_PROMPT.format(
        knowledge_yaml=load_yaml_as_string(PATH + '/knowledge/knative_autoscaling_knowledge2.yaml'),
        service_name=service_name,
        revision_name=service_name,
        anomaly_type="latency spike",
        timestamp=timestamp,
        duration=duration,
        snapshot=snapshot,
        service_config=service_config,
        auto_config=auto_config
    )
    
    return prompt


def call_llm(prompt, llm_type="default"):
    if llm_type == "openai":
        return _call_openai(prompt)
    else:
        return _call_default_llm(prompt)


def _call_openai(prompt):
    """
    Call OpenAI API with the given prompt.
    
    Args:
        prompt: The prompt to send to OpenAI
        
    Returns:
        str: OpenAI response content
    """
    print(f"Sending prompt to OpenAI {OPENAI_MODEL} for configuration generation...")

    # Prepare OpenAI API request
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_completion_tokens": 2000,
        "temperature": 0.1,  # Low temperature for consistent configuration generation
        "top_p": 0.9
    }

    # Send request to OpenAI API
    print(f"Sending request to OpenAI with payload size: {len(str(payload))} characters")
    print(f"Prompt length: {len(prompt)} characters")
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120  # Increase timeout to 2 minutes
    )
    
    print(f"Response status: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    
    if response.status_code != 200:
        print(f"OpenAI API error: {response.status_code} - {response.text}")
        return None
    
    data = response.json()
    print(f"Response data keys: {list(data.keys())}")
    print(f"Choices count: {len(data.get('choices', []))}")
    
    if not data.get("choices") or len(data["choices"]) == 0:
        print("❌ No choices in response")
        return None
        
    choice = data["choices"][0]
    print(f"Choice keys: {list(choice.keys())}")
    print(f"Message keys: {list(choice.get('message', {}).keys())}")
    
    return choice["message"]["content"]


def _call_default_llm(prompt):
    message = [["user", prompt]]
    payload = {
        "messages": message,
        "max_new_tokens": 2000
    }

    response = requests.post(GEN_API_URL, json=payload)
    data = response.json()["response"]

    return data


def read_prompt(response: str):
    """
    Extract YAML configuration from LLM response.
    
    Args:
        response: LLM API response object
        
    Returns:
        str: Extracted YAML configuration
    """
    print(f"Extracting YAML from response of length: {len(response)}")
        
    # Try to find YAML blocks in the response
    yaml_patterns = [
        r"```yaml\n([\s\S]*?)\n```",
        r"```\n([\s\S]*?)\n```",
        r"<yaml>\n([\s\S]*?)\n</yaml>",
        r"```([\s\S]*?)```"
    ]
    
    for pattern in yaml_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)

        for match in matches:
            yaml_str = match.strip()

            try:
                yaml.safe_load(yaml_str)
                return yaml_str

            except yaml.YAMLError as e:
                continue

    return None


def generate_configuration_with_openai(service_name, trace_df, metric_dfs, anomaly_index, 
                                     anomalies, knobs, label):
    """
    Generate configuration changes for a specific anomaly using OpenAI GPT-4o-mini.
    
    Args:
        service_name: Name of the service with the anomaly
        trace_df: Trace data DataFrame
        metric_dfs: Metrics data dictionary
        anomaly_index: Index of the anomaly in trace data
        anomalies: List of all anomalies
        knobs: Current configuration knobs
        label: Experiment label
        
    Returns:
        str: LLM response content
    """
    try:
        # Generate the prompt
        timestamp, duration, snapshot, prompt = generate_prompt(
            service_name, trace_df, metric_dfs, anomaly_index, anomalies, knobs, label
        )

        # Call OpenAI LLM
        llm_response = call_llm(prompt, llm_type="openai", label=label, anomaly_index=anomaly_index)
        
        if llm_response is None:
            return None

        return llm_response
        
    except Exception as e:
        print(f"Error generating configuration with OpenAI for {service_name}: {e}")
        return None

def generate_configuration_for_anomaly(service_name, trace_df, metric_dfs, anomaly_index, 
                                     anomalies, knobs, label):
    """
    Generate configuration changes for a specific anomaly using LLM.
    
    Args:
        service_name: Name of the service with the anomaly
        trace_df: Trace data DataFrame
        metric_dfs: Metrics data dictionary
        anomaly_index: Index of the anomaly in trace data
        anomalies: List of all anomalies
        knobs: Current configuration knobs
        label: Experiment label
        
    Returns:
        tuple: (success: bool, configuration_update: str)
    """
    try:
        # Generate the prompt
        timestamp, duration, snapshot, prompt = generate_prompt(
            service_name, trace_df, metric_dfs, anomaly_index, anomalies, knobs, label
        )

        # Call default LLM
        llm_response = call_llm(prompt, llm_type="default", label=label, anomaly_index=anomaly_index)
        
        if llm_response is None:
            return False, None

        # Extract configuration update
        configuration_update = read_prompt(llm_response)

        if configuration_update:
            print("Configuration update received")
            return True, configuration_update
        else:
            print("No valid configuration update received")
            return False, None

    except Exception as e:
        print(f"Error generating configuration for {service_name}: {e}")
        return False, None


def generate_configuration(prompt, llm_type="default"):
    try:
        # Call the specified LLM
        llm_response = call_llm(prompt, llm_type=llm_type)
        
        if llm_response is None:
            return False, None

        # For OpenAI, return the raw response
        if llm_type == "openai":
            return True, llm_response
        
        # For default LLM, extract configuration update
        configuration_update = read_prompt(llm_response)
        
        if configuration_update:
            print("Configuration update received")
            return True, configuration_update
        else:
            print("No valid configuration update received")
            return False, None

    except Exception as e:
        print(f"Error generating configuration for {service_name}: {e}")
        return False, None
