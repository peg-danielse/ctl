import datetime
import os
import random
import re
import threading
import time
import pandas as pd
import requests
import yaml
import logging
from typing import Any

from config import GEN_API_URL, OPENAI_API_KEY, OPENAI_MODEL, GEMINI_API_KEY, GEMINI_MODEL, PATH
from prompt import GENERATE_PROMPT, GOAL, RESULT_PROMPT
from util.config_manager import ConfigManager, load_yaml_as_string
from util.data_retrieval import DataCollector

# Configure logging
logger = logging.getLogger(__name__)
data_collector = DataCollector.get_instance()

def generate_prompt(service_name, trace_df, metric_dfs, 
                   anomalies, label):
    
    # Create metric snapshot for the anomaly
    snapshot = data_collector.log_all_data_to_snapshot(start_time=anomalies[0]['timestamp']- pd.to_timedelta(3, unit='m'), 
                                                        end_time=anomalies[0]['timestamp'] + pd.to_timedelta(anomalies[0]['duration_seconds'] + 3, unit='s'), 
                                                        phase="adaptation", 
                                                        subphase="configuration_application")

    # Load service and autoscaler configurations
    logger.info(f"loading service config for {service_name}")

    config_manager = ConfigManager.get_instance(label)
    
    service_config = yaml.dump(config_manager.get_service_config(service_name))
    auto_config = yaml.dump(config_manager.get_service_config("config-autoscaler"))

    # Prepare prompt for configuration generation
    prompt = GENERATE_PROMPT.format(
        service_name=service_name,
        revision_name=service_name,
        anomaly_type="latency spike",
        timestamp=anomalies[0]['timestamp'],
        duration=anomalies[0]['duration_seconds'],
        snapshot=snapshot,
        service_config=service_config,
        auto_config=auto_config
    )
    
    return prompt


def call_llm(prompt, llm_type="self-hosted"):
    match (llm_type):
        case "openai":
            return _call_openai(prompt)
        case "gemini":
            return _call_gemini(prompt)
        case "self-hosted":
            return _call_default_llm(prompt)
        case _:
            raise ValueError(f"Invalid LLM type: {llm_type}")

def _call_openai(messages):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Debug: Log the structure of incoming messages
    logger.debug(f"OpenAI API called with {len(messages)} messages")
    for i, msg in enumerate(messages):
        logger.debug(f"Message {i}: type={type(msg)}, content={str(msg)[:100]}...")
    
    # Convert the chat format to OpenAI format
    openai_messages = []
    for i, message in enumerate(messages):
        if isinstance(message, list) and len(message) == 2:
            role, content = message
            # Ensure content is a string
            if not isinstance(content, str):
                logger.warning(f"Message {i} content is not a string, converting: {type(content)} -> {str(content)}")
                content = str(content)
            openai_messages.append({"role": role, "content": content})
        else:
            # Handle single prompt case
            logger.warning(f"Message {i} is not in expected format [role, content], got: {type(message)}")
            openai_messages.append({"role": "user", "content": str(message)})
    
    payload = {
        "model": OPENAI_MODEL,
        "messages": openai_messages,
        "max_completion_tokens": 2000,
        "temperature": 0.1,  # Low temperature for consistent configuration generation
        "top_p": 0.9
    }
    
    # Debug: Validate the payload structure
    logger.debug(f"OpenAI payload has {len(openai_messages)} messages")
    for i, msg in enumerate(openai_messages):
        if not isinstance(msg.get('content'), str):
            logger.error(f"Invalid message {i}: content is {type(msg.get('content'))}, not string")
            logger.error(f"Message content: {msg.get('content')}")
        logger.debug(f"Message {i}: role='{msg.get('role')}', content_type={type(msg.get('content'))}")

    # Send request to OpenAI API
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120  # Increase timeout to 2 minutes
    )
    
    if response.status_code != 200:
        logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
        return None
    
    data = response.json()        
    choice = data["choices"][0]
    
    return choice["message"]["content"]


def _call_default_llm(messages):
    # Convert the chat format to the expected format for the default LLM
    formatted_messages = []
    for message in messages:
        if isinstance(message, list) and len(message) == 2:
            role, content = message
            # Skip empty or None content
            if content is not None and str(content).strip():
                formatted_messages.append([role, str(content)])
        else:
            # Handle single prompt case
            if message is not None and str(message).strip():
                formatted_messages.append(["user", str(message)])
    
    # Safety check: ensure we have at least one message
    if not formatted_messages:
        logger.warning("No valid messages found, using default message")
        formatted_messages = [["user", "Please provide a response."]]
    
    payload = {
        "messages": formatted_messages,
        "max_new_tokens": 4096
    }

    response = requests.post(GEN_API_URL, json=payload)
    data = response.json()["response"]

    return data


def _call_gemini(messages):
    """
    Call Google Gemini API with the provided messages.
    """
    # Validate API key
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set. Please set the environment variable.")
        return None
    
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    
    # Convert the chat format to Gemini format
    # Gemini expects a single text input, so we'll concatenate the conversation
    conversation_text = ""
    for message in messages:
        if isinstance(message, list) and len(message) == 2:
            role, content = message
            # Ensure content is a string
            if not isinstance(content, str):
                content = str(content)
            conversation_text += f"{role.upper()}: {content}\n\n"
        else:
            # Handle single prompt case
            conversation_text += f"USER: {str(message)}\n\n"
    
    # Remove trailing newlines
    conversation_text = conversation_text.strip()
    
    payload = {
        "contents": [{
            "parts": [{
                "text": conversation_text
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": 4096,
            "temperature": 0.1,  # Low temperature for consistent configuration generation
            "topP": 0.9
        }
    }
    
    # Debug: Log the request structure
    logger.debug(f"Gemini API called with conversation length: {len(conversation_text)}")
    logger.debug(f"Using model: {GEMINI_MODEL}")
    logger.debug(f"API key present: {bool(GEMINI_API_KEY)}")
    
    # Send request to Gemini API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    logger.debug(f"Making request to: {url}")
    
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120  # 2 minutes timeout
    )
    
    if response.status_code != 200:
        logger.error(f"Gemini API error: {response.status_code} - {response.text}")
        return None
    
    data = response.json()
    
    # Extract the generated content (be robust to missing parts)
    if "candidates" in data and len(data["candidates"]) > 0:
        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", []) if isinstance(candidate.get("content", {}), dict) else []
        if parts and isinstance(parts[0], dict) and "text" in parts[0]:
            return parts[0]["text"]
        logger.warning(f"Gemini candidate missing text parts; finishReason={candidate.get('finishReason')}")
        return ""
    
    logger.error(f"Unexpected Gemini API response format: {data}")
    return ""


def read_prompt(response: str):
    yaml_patterns = [
        r"```yaml\n([\s\S]*?)\n```",
        r"```\n([\s\S]*?)\n```",
        r"<yaml>\n([\s\S]*?)\n</yaml>",
        r"```([\s\S]*?)```"
    ]
    
    for pattern in yaml_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)

        yaml_str = ""
        for i, match in enumerate(matches):
            if i > 0:
                yaml_str += "---\n"
            yaml_str += match.strip() + "\n"

        try:
            yaml.safe_load_all(yaml_str)
            return yaml_str

        except yaml.YAMLError as e:
            continue

    return None

# TODO: filter the previous configurations to only include the ones that are the highest performing ones.
class Chat():
    MAX_EXAMPLES = 5
    
    def __init__(self, service_name: str, knowledge: str, goal: str):
        self.previous_configurations = []
        self.service_name = service_name
        self.knowledge = knowledge
        self.goal = goal

    def add_example(self, configuration: Any, result: Any):
        self.previous_configurations.append(["assistant", yaml.dump(configuration)])
        self.previous_configurations.append(["user", yaml.dump(result)])

        if len(self.previous_configurations) > self.MAX_EXAMPLES:
            self.previous_configurations.pop(0)

    def get_chat_log(self, prompt) -> list:
        # Ensure prompt is not None or empty
        if prompt is None:
            prompt = ""
        prompt = str(prompt).strip()
        
        # If prompt is empty, use a default message
        if not prompt:
            prompt = "Please provide a configuration recommendation."
            
        return [["user", self.knowledge]] + self.previous_configurations + [["user", self.goal]] + [["user", prompt]] 

class ChatManager():
    lock = threading.Lock()
    instance = None

    @staticmethod
    def get_instance():
        if ChatManager.instance is None:
            ChatManager.instance = ChatManager()
        return ChatManager.instance

    def __init__(self):
        self.chats: list[Chat] = []
        self.knowledge = load_yaml_as_string(PATH + '/knowledge/knative_autoscaling_knowledge2.yaml')
        self.goal = GOAL

    def add_chat(self, service_name):
        self.chats.append(Chat(service_name, self.knowledge, self.goal))

    def get_chat(self, service_name) -> Chat:
        return next((item for item in self.chats if item.service_name == service_name), None)

    def add_example(self, service_name, configuration, result):
        selected = self.get_chat(service_name)
        selected.add_example(configuration, RESULT_PROMPT.format(result=result))

    # keep chat in memory, if the service_name is the same, update the chat
    def generate_configuration(self, prompt, service_name, llm_type="openai") -> str:
            selected = self.get_chat(service_name)

            if selected is None:
                selected = Chat(service_name, self.knowledge, self.goal)
                self.chats.append(selected)  # Actually add it to the chats list

            chat = selected.get_chat_log(prompt)
            llm_response = call_llm(chat, llm_type=llm_type)

            os.makedirs(f"./output/tmp/data", exist_ok=True)
            with open(f"./output/tmp/data/llm_responses{random.randint(0, 100)}.txt", "a") as f:
                f.write("--------------------------------\n")
                f.write(f"{service_name}:\n")
                f.write(f"{prompt}\n")
                f.write(f"{llm_response}\n")

            configuration = read_prompt(llm_response)

            return configuration

    def save_all_chats(self, label):
        import os
        for chat in self.chats:
            # Create directory if it doesn't exist
            chat_dir = PATH + f"/output/{label}/data/chats"
            os.makedirs(chat_dir, exist_ok=True)
            
            with open(f"{chat_dir}/{chat.service_name}.yaml", "w") as f:
                f.write(f"{chat.goal}\n")
                f.write(f"{chat.knowledge}\n")
                for message in chat.previous_configurations:
                    f.write(f"{message[0]}\n")
                    f.write(f"{message[1]}\n")


def task_score_configuration(measurement_time, configuration, service_name, label):
    start_time = datetime.datetime.now().astimezone(datetime.timezone.utc)
    time.sleep(measurement_time * 60)
    end_time = start_time + datetime.timedelta(seconds=measurement_time)
    
    dc= DataCollector.get_instance()
    score = dc.log_all_data_to_snapshot(start_time, end_time, phase="evaluation", subphase="configuration_evaluation")
    
    chat_manager = ChatManager.get_instance()
    
    try:
        chat_manager.lock.acquire()
        chat_manager.add_example(service_name, configuration, score)
    finally:
        chat_manager.lock.release()
