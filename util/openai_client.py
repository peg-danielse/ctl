"""
OpenAI client utilities for GPT-4o-mini integration.
Handles communication with OpenAI API for configuration generation and analysis.
"""

import re
import yaml
import requests
from typing import List, Tuple, Optional
from config import OPENAI_API_KEY, OPENAI_MODEL, PATH
from prompt import GENERATE_PROMPT
from util.config_manager import load_yaml_as_string
from util.sequence import append_generation


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
        tuple: (success: bool, configuration_update: str)
    """
    from util.analysis import metric_snapshot
    
    try:
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
            "max_completion_tokens": 2000
        }
        
        # Add temperature and top_p for GPT-4o-mini
        payload["temperature"] = 0.1  # Low temperature for consistent configuration generation
        payload["top_p"] = 0.9

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
            return False, None
        
        data = response.json()
        print(f"Response data keys: {list(data.keys())}")
        print(f"Choices count: {len(data.get('choices', []))}")
        
        if not data.get("choices") or len(data["choices"]) == 0:
            print("❌ No choices in response")
            return False, None
            
        choice = data["choices"][0]
        print(f"Choice keys: {list(choice.keys())}")
        print(f"Message keys: {list(choice.get('message', {}).keys())}")
        
        llm_response = choice["message"]["content"]

        print("OpenAI response received")  
        print(f"Response length: {len(llm_response)} characters")

        print(f"Response preview: {llm_response[:200]}...")
        # Save the prompt and response
        append_generation(PATH + f"/output/{label}/{label}_openai_prompts.json", anomaly_index, llm_response)

        # Extract configuration update
        configuration_update = extract_yaml_from_response(llm_response)

        if configuration_update:
            print("Configuration update received from OpenAI")
            print(f"Extracted YAML length: {len(configuration_update)} characters")
            return True, configuration_update
        else:
            print("No valid configuration update received from OpenAI")
            print("YAML extraction failed - checking response content...")
            return False, None

    except Exception as e:
        print(f"Error generating configuration with OpenAI for {service_name}: {e}")
        return False, None


def extract_yaml_from_response(response_text):
    """
    Extract YAML configuration from OpenAI response.
    
    Args:
        response_text: Raw response from OpenAI API
        
    Returns:
        str: Extracted YAML configuration or None
    """
    try:
        print(f"Extracting YAML from response of length: {len(response_text)}")
        
        # Try to find YAML blocks in the response
        yaml_patterns = [
            r"```yaml\n([\s\S]*?)\n```",
            r"```\n([\s\S]*?)\n```",
            r"<yaml>\n([\s\S]*?)\n</yaml>",
            r"```([\s\S]*?)```"
        ]
        
        for i, pattern in enumerate(yaml_patterns):
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            print(f"Pattern {i+1} found {len(matches)} matches")
            for match in matches:
                yaml_str = match.strip()
                print(f"Trying to parse YAML of length: {len(yaml_str)}")
                # Validate that it's actually YAML
                try:
                    yaml.safe_load(yaml_str)
                    print("✅ Valid YAML found!")
                    return yaml_str
                except yaml.YAMLError as e:
                    print(f"❌ YAML parsing failed: {e}")
                    continue
        
        # If no YAML blocks found, try to extract the entire response as YAML
        print("Trying to parse entire response as YAML...")
        try:
            yaml.safe_load(response_text)
            print("✅ Entire response is valid YAML!")
            return response_text
        except yaml.YAMLError as e:
            print(f"❌ Entire response is not valid YAML: {e}")
            pass
            
        print("❌ No valid YAML found in response")
        return None
        
    except Exception as e:
        print(f"Error extracting YAML from response: {e}")
        return None


def test_openai_connection():
    """
    Test OpenAI API connection and model availability.
    
    Returns:
        bool: True if connection is successful
    """
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! Please respond with 'OpenAI connection successful'."
                }
            ],
            "max_completion_tokens": 50
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"✅ OpenAI connection successful: {content}")
            return True
        else:
            print(f"❌ OpenAI API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False


def compare_llm_responses(local_response, openai_response):
    """
    Compare responses from local LLM and OpenAI for analysis.
    
    Args:
        local_response: Response from local LLM
        openai_response: Response from OpenAI
        
    Returns:
        dict: Comparison analysis
    """
    try:
        local_yaml = extract_yaml_from_response(local_response) if local_response else None
        openai_yaml = extract_yaml_from_response(openai_response) if openai_response else None
        
        comparison = {
            "local_llm": {
                "has_response": local_response is not None,
                "has_yaml": local_yaml is not None,
                "response_length": len(local_response) if local_response else 0
            },
            "openai": {
                "has_response": openai_response is not None,
                "has_yaml": openai_yaml is not None,
                "response_length": len(openai_response) if openai_response else 0
            }
        }
        
        if local_yaml and openai_yaml:
            try:
                local_config = yaml.safe_load(local_yaml)
                openai_config = yaml.safe_load(openai_yaml)
                comparison["config_comparison"] = {
                    "local_keys": list(local_config.keys()) if isinstance(local_config, dict) else [],
                    "openai_keys": list(openai_config.keys()) if isinstance(openai_config, dict) else []
                }
            except yaml.YAMLError:
                comparison["config_comparison"] = {"error": "Failed to parse YAML"}
        
        return comparison
        
    except Exception as e:
        return {"error": str(e)}
