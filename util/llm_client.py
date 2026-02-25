import datetime
import os
import re
import threading
import time
import pandas as pd
import requests
import yaml
import logging
from typing import Any, Optional

from config import GEN_API_URL, OPENAI_API_KEY, OPENAI_MODEL, GEMINI_API_KEY, GEMINI_MODEL, PATH
from prompt import GENERATE_PROMPT, GOAL, RESULT_PROMPT
from util.config_manager import ConfigManager, load_yaml_as_string
from util.data_retrieval import DataCollector

KNOWLEDGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge", "knative_autoscaling_knowledge2.yaml")


def _format_snapshot_for_prompt(data: dict) -> str:
    """
    Format a metric snapshot or result dict as readable YAML for the LLM prompt.
    Handles non-JSON-serializable types (datetime, numpy, etc.).
    """
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        if isinstance(obj, (int, float)) and (obj != obj or abs(obj) == float("inf")):  # NaN, Inf
            return str(obj)
        return obj

    try:
        sanitized = _sanitize(data)
        return yaml.dump(sanitized, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except Exception as e:
        logger.warning(f"Could not format snapshot as YAML: {e}")
        return str(data)


# Configure logging
logger = logging.getLogger(__name__)
data_collector = DataCollector.get_instance()


def _build_constraints_text() -> str:
    """Load knowledge file and format constraints for the prompt."""
    try:
        with open(KNOWLEDGE_PATH) as f:
            knowledge = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load knowledge file: {e}")
        return "Policy: respect existing configuration keys and valid value ranges."
    lines = []
    # Fixed-replica services (NEVER change replicas for these)
    fixed = knowledge.get("fixed_replica_services") or []
    if fixed:
        lines.append("FORBIDDEN: Do NOT change replicas for these services (must stay at 1):")
        lines.append(f"  {', '.join(fixed)}")
        lines.append("")
    # Policy constraints (rules the LLM must follow)
    policy = knowledge.get("policy_constraints") or []
    if policy:
        lines.append("Policy rules:")
        for i, rule in enumerate(policy, 1):
            lines.append(f"  {i}. {rule}")
        lines.append("")
    # Parameter constraints (value ranges)
    param = knowledge.get("parameter_constraints") or {}
    if param:
        lines.append("Parameter value ranges (strict bounds):")
        for name, spec in param.items():
            key = spec.get("key", name)
            r = spec.get("range")
            mn = spec.get("min")
            note = spec.get("note", "")
            if r:
                lines.append(f"  - {key}: allowed range {r}")
            elif mn is not None:
                lines.append(f"  - {key}: minimum {mn}")
            if note:
                lines.append(f"    ({note})")
        lines.append("")
    return "\n".join(lines) if lines else "Respect valid Knative autoscaling parameter values."


def get_fixed_replica_services() -> set:
    """Return set of service names that must always have 1 replica."""
    try:
        with open(KNOWLEDGE_PATH) as f:
            knowledge = yaml.safe_load(f)
        return set(knowledge.get("fixed_replica_services") or [])
    except Exception:
        return set()


def enforce_fixed_replica_policy(service_name: str, config: dict) -> dict:
    """
    If the service is in the fixed-replica list, sanitize config to enforce replicas=1.
    Returns the (possibly modified) config. Does not mutate the input.
    """
    fixed = get_fixed_replica_services()
    if service_name not in fixed:
        return config
    kind = config.get("kind", "")
    if kind == "Deployment":
        spec = config.get("spec") or {}
        if isinstance(spec, dict) and spec.get("replicas") != 1:
            config = dict(config)
            config["spec"] = dict(spec)
            config["spec"]["replicas"] = 1
            logger.info(f"Policy: enforced replicas=1 for fixed-replica service {service_name}")
    elif kind == "Service":
        # Knative Service: enforce min-scale=max-scale=1
        annotations = (config.get("metadata") or {}).get("annotations") or {}
        if isinstance(annotations, dict):
            min_ok = annotations.get("autoscaling.knative.dev/min-scale") == "1"
            max_ok = annotations.get("autoscaling.knative.dev/max-scale") == "1"
            if not min_ok or not max_ok:
                config = dict(config)
                meta = dict(config.get("metadata") or {})
                ann = dict(meta.get("annotations") or {})
                ann["autoscaling.knative.dev/min-scale"] = "1"
                ann["autoscaling.knative.dev/max-scale"] = "1"
                meta["annotations"] = ann
                config["metadata"] = meta
                logger.info(f"Policy: enforced min-scale=max-scale=1 for fixed-replica service {service_name}")
    return config


def _build_node_placement_text(snapshot: dict) -> str:
    """Format node-to-service placement for the prompt."""
    nodes = snapshot.get("nodes") or {}
    if not nodes:
        return "No node placement data available."
    lines = []
    # Per-node: services and pod counts
    for node_name, info in nodes.items():
        svcs = (info or {}).get("services") or {}
        if not svcs:
            lines.append(f"  {node_name}: (no services)")
            continue
        parts = [f"{svc}={pods:.1f}" for svc, pods in sorted(svcs.items())]
        lines.append(f"  {node_name}: {', '.join(parts)}")
    return "\n".join(lines) if lines else "No node placement data available."


def _build_root_cause_text(snapshot: dict) -> str:
    """Format pod health (restarts, OOM) so the LLM can fix the right service (e.g. srv-rate OOM, not memcached-rate)."""
    health = snapshot.get("pod_health") or {}
    if not health:
        return "No pod health data (restarts/OOM) in this window."
    lines = []
    for svc, info in sorted(health.items()):
        restarts = info.get("restarts", 0)
        oom = info.get("oom_killed", False)
        if restarts > 0 or oom:
            parts = []
            if restarts > 0:
                parts.append(f"restarts={restarts}")
            if oom:
                parts.append("OOM_KILLED=True")
            lines.append(f"  {svc}: {', '.join(parts)}")
    if not lines:
        return "No restarts or OOM in this window."
    return "Services with restarts or OOM (fix these first; they are likely root cause):\n" + "\n".join(lines)


# Map pod_health service names to ConfigManager keys (from base_config filenames)
_ROOT_CAUSE_SERVICE_TO_CONFIG_KEY = {
    "memcached-reserve": "memcached-reservation-deployment",
    "memcached-rate": "memcached-rate-deployment",
    "memcached-profile": "memcached-profile-deployment",
    "frontend": "frontend-deployment",
    "mongodb-rate": "mongodb-rate-deployment",
    "mongodb-user": "mongodb-user-deployment",
    "mongodb-reservation": "mongodb-reservation-deployment",
    "mongodb-profile": "mongodb-profile-deployment",
    "mongodb-geo": "mongodb-geo-deployment",
}


def _build_root_cause_configs(snapshot: dict, config_manager: Any) -> str:
    """Current YAML config for each service in pod_health (restarts/OOM) so the LLM can suggest fixes for them."""
    health = snapshot.get("pod_health") or {}
    root_cause_services = [svc for svc, info in health.items() if info.get("restarts", 0) > 0 or info.get("oom_killed", False)]
    if not root_cause_services:
        return ""
    lines = []
    for svc in sorted(root_cause_services):
        config_key = _ROOT_CAUSE_SERVICE_TO_CONFIG_KEY.get(svc, svc)
        try:
            config = config_manager.get_service_config(config_key)
            if config:
                lines.append(f"# Current configuration for {svc} (root cause; suggest changes to fix restarts/OOM):")
                lines.append(yaml.dump(config, default_flow_style=False, sort_keys=False))
                lines.append("")
        except (IndexError, KeyError, TypeError):
            pass
    return "\n".join(lines).strip() if lines else ""


def generate_prompt(service_name, trace_df, metric_dfs, 
                   anomalies, label):
    
    # Create metric snapshot for the anomaly
    snapshot = data_collector.log_all_data_to_snapshot(start_time=anomalies[0]['timestamp']- pd.to_timedelta(3, unit='m'), 
                                                        end_time=anomalies[0]['timestamp'] + pd.to_timedelta(anomalies[0]['duration_seconds'] + 3, unit='s'), 
                                                        phase="adaptation", 
                                                        subphase="configuration_application")

    # Load service and autoscaler configurations (must exist in base_configuration)
    logger.info(f"loading service config for {service_name}")

    config_manager = ConfigManager.get_instance(label)
    raw_service_config = config_manager.get_service_config(service_name)
    if not raw_service_config:
        logger.error(
            "Service '%s' has no configuration in base_configuration. "
            "Add %s.yaml to the base_configuration directory and re-run.",
            service_name,
            service_name,
        )
        return None

    service_config = yaml.dump(raw_service_config)
    auto_config = yaml.dump(config_manager.get_service_config("config-autoscaler"))

    # Build constraints, node placement, root-cause, and current configs for root-cause services
    constraints = _build_constraints_text()
    node_placement = _build_node_placement_text(snapshot)
    root_cause = _build_root_cause_text(snapshot)
    root_cause_configs = _build_root_cause_configs(snapshot, config_manager)
    if root_cause_configs:
        root_cause_configs = "\n\nCurrent configurations for root-cause services above (use these to suggest fixes):\n```yaml\n" + root_cause_configs + "\n```"
    else:
        root_cause_configs = ""

    # Prepare prompt for configuration generation
    prompt = GENERATE_PROMPT.format(
        service_name=service_name,
        revision_name=service_name,
        anomaly_type="latency spike",
        timestamp=anomalies[0]['timestamp'],
        duration=anomalies[0]['duration_seconds'],
        constraints=constraints,
        node_placement=node_placement,
        root_cause=root_cause,
        root_cause_configs=root_cause_configs,
        snapshot=_format_snapshot_for_prompt(snapshot),
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
            "maxOutputTokens": 8192,
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
    
    # First, try to extract YAML blocks from common wrapper formats
    for pattern in yaml_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)

        yaml_str = ""
        for i, match in enumerate(matches):
            if i > 0:
                yaml_str += "---\n"
            yaml_str += match.strip() + "\n"

        if not yaml_str.strip():
            continue

        try:
            # Validate that the extracted string is valid YAML (may contain multiple docs)
            list(yaml.safe_load_all(yaml_str))
            return yaml_str
        except yaml.YAMLError:
            continue

    # Fallback: try to interpret the entire response as YAML (e.g., plain YAML with no fences)
    try:
        list(yaml.safe_load_all(response))
        return response
    except yaml.YAMLError:
        return None

# TODO: filter the previous configurations to only include the ones that are the highest performing ones.
class Chat():
    MAX_EXAMPLES = 5
    
    def __init__(self, service_name: str, knowledge: str, goal: str):
        self.previous_configurations = []
        # Full LLM chat exchanges for this service, used only for persistence/analysis.
        # Each item is a dict with keys: service, label, llm_type, messages, response.
        self.llm_exchanges: list[dict[str, Any]] = []
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

    def add_llm_exchange(
        self,
        messages: list[Any],
        llm_response: Any,
        llm_type: Optional[str] = None,
        label: Optional[str] = None,
    ) -> None:
        """
        Record a full LLM exchange (chat messages + raw response) for later saving.
        This does NOT affect the messages sent back to the LLM on subsequent calls.
        """
        if llm_response is None or not str(llm_response).strip():
            return

        sanitized_messages: list[dict[str, str]] = []
        try:
            for msg in messages or []:
                if isinstance(msg, list) and len(msg) == 2:
                    role, content = msg
                    sanitized_messages.append(
                        {
                            "role": str(role),
                            "content": "" if content is None else str(content),
                        }
                    )
                else:
                    sanitized_messages.append(
                        {
                            "role": "user",
                            "content": "" if msg is None else str(msg),
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to sanitize chat messages for LLM exchange on {self.service_name}: {e}")
            sanitized_messages = []

        exchange: dict[str, Any] = {
            "service": self.service_name,
            "label": label,
            "llm_type": llm_type,
            "messages": sanitized_messages,
            "response": str(llm_response),
        }
        self.llm_exchanges.append(exchange)

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
        formatted_result = _format_snapshot_for_prompt(result)
        selected.add_example(configuration, RESULT_PROMPT.format(result=formatted_result))

    # keep chat in memory, if the service_name is the same, update the chat
    def generate_configuration(self, prompt, service_name, llm_type: str = "openai", label: Optional[str] = None) -> Optional[str]:
        """
        Ask the configured LLM for a new configuration for the given service.

        - Returns a YAML string with one or more documents on success.
        - Returns None (and logs) if the LLM did not produce a usable configuration.
        - Also prints the configuration and writes it to the experiment's config
          folder (if label is provided) or to tmp for debugging.
        """
        selected = self.get_chat(service_name)

        if selected is None:
            selected = Chat(service_name, self.knowledge, self.goal)
            self.chats.append(selected)  # Actually add it to the chats list

        print(prompt)
        chat = selected.get_chat_log(prompt)
        llm_response = call_llm(chat, llm_type=llm_type)

        # Persist the full chat + raw LLM response on the Chat instance for later saving.
        try:
            ChatManager.lock.acquire()
            selected.add_llm_exchange(chat, llm_response, llm_type=llm_type, label=label)
        finally:
            ChatManager.lock.release()

        # Persist a lightweight text log of prompt + response for debugging/analysis.
        # If an experiment label is provided, save under that run's data folder;
        # otherwise, fall back to the shared tmp directory.
        try:
            if label:
                data_dir = os.path.join(PATH, "output", label, "data")
            else:
                data_dir = os.path.join(PATH, "output", "tmp", "data")
            os.makedirs(data_dir, exist_ok=True)
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(
                data_dir,
                f"llm_response_{service_name}_{llm_type}.txt",
            )
            with open(log_path, "a") as f:
                f.write("--------------------------------\n")
                f.write(f"Timestamp: {ts}\n")
                f.write(f"{service_name}:\n\n")
                f.write("--------------------------------\n")
                f.write(f"{prompt}\n")
                f.write("--------------------------------\n")
                f.write(f"{llm_response}\n")
        except Exception as e:
            logger.error("Failed to write LLM debug log for '%s': %s", service_name, e)

        if not llm_response or not str(llm_response).strip():
            logger.error("LLM returned an empty response for service '%s'; skipping configuration.", service_name)
            return None

        configuration = read_prompt(llm_response)
        if not configuration or not str(configuration).strip():
            logger.error("No configuration could be parsed from LLM response for service '%s'; skipping.", service_name)
            return None

        # Print generated configuration to stdout (applies to both OpenAI and Gemini)
        print(f"\n--- Generated configuration for {service_name} ({llm_type}) ---\n{configuration}\n---")

        # Persist generated configuration close to the LLM client.
        # If we know the experiment label, save under that run's config folder.
        try:
            if label:
                config_dir = os.path.join(PATH, "output", label, "config")
            else:
                config_dir = os.path.join(PATH, "output", "tmp", "config")
            os.makedirs(config_dir, exist_ok=True)
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe_service = service_name.replace("/", "_")
            generated_path = os.path.join(config_dir, f"generated_{safe_service}_{ts}.yaml")
            with open(generated_path, "w") as f:
                f.write(configuration)
            logger.info("Saved generated config for '%s' to %s", service_name, generated_path)
        except Exception as e:
            logger.error("Failed to persist generated configuration for '%s': %s", service_name, e)

        return configuration

    def save_all_chats(self, label):
        import os
        for chat in self.chats:
            # Create directory if it doesn't exist
            chat_dir = PATH + f"/output/{label}/data/chats"
            os.makedirs(chat_dir, exist_ok=True)
            
            with open(f"{chat_dir}/{chat.service_name}.yaml", "w") as f:
                # High-level goal and knowledge context
                f.write(f"{chat.goal}\n")
                f.write(f"{chat.knowledge}\n")

                # Previous configuration/result pairs that are used as in-context examples
                for message in chat.previous_configurations:
                    f.write(f"{message[0]}\n")
                    f.write(f"{message[1]}\n")

                # Full LLM exchanges (chat + raw response) for this service
                if chat.llm_exchanges:
                    f.write("\n# LLM exchanges (full chat + raw response)\n")
                    for idx, exchange in enumerate(chat.llm_exchanges, start=1):
                        f.write(f"\n## Exchange {idx}\n")
                        if exchange.get("label") is not None:
                            f.write(f"label: {exchange['label']}\n")
                        if exchange.get("llm_type") is not None:
                            f.write(f"llm_type: {exchange['llm_type']}\n")
                        f.write("chat_messages:\n")
                        for msg in exchange.get("messages", []):
                            role = msg.get("role", "user")
                            content = msg.get("content", "")
                            f.write(f"- role: {role}\n")
                            f.write("  content: |\n")
                            for line in str(content).splitlines() or [""]:
                                f.write(f"    {line}\n")
                        f.write("llm_response: |\n")
                        for line in str(exchange.get("response", "")).splitlines() or [""]:
                            f.write(f"  {line}\n")


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
