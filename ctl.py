import os, math, json, glob, re, requests, time, subprocess, heapq, yaml, shutil

import hashlib


from functools import wraps
from operator import itemgetter

# experimental code, remove security warning.
import warnings
warnings.filterwarnings('ignore', message="Unverified HTTPS request*")

import numpy as np
import pandas as pd

import shap
from sklearn.ensemble import IsolationForest
from kubernetes import client, config

from pprint import pprint

from util.analysis import *
from util.sequence import *
from util.square import *

from config import PATH, GEN_API_URL, SPAN_PROCESS_MAP
from prompt import GENERATE_PROMPT, CHECK_PROMPT, FILE_PROMPT, RESULT_PROMPT

import textwrap


LOAD_TEST_TIME = '10m'

def simple_hash(s):
    hash_object = hashlib.sha256(s.encode())
    hex_digest = hash_object.hexdigest()
    
    int_hash = int(hex_digest, 16)
    
    return int_hash % 10000

def report_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper


def locust_load_test(label, base_label, loop): # todo add all loadtest configurations
    if len(glob.glob(PATH + "/output/" + f"{base_label}/data/{label}")) != 0:
        print(f"loadtest using label: {label} already exists.", "skipping...")
        return

    # ("--w-shape", type=int,is_required=False, default=1)
    # ("--w-mean", type=int, is_required=False, default=150)
    # ("--w-user-min", type=int, is_required=False, default=100)
    # ("--w-user-max", type=int, is_required=False, default=1000)
    # ("--w-dt", type=int, is_required=False, default=20)
    # ("--w-ls-y", type=int, is_required=False, default=500)
    # ("--seed", type=int, is_required=False, default=42)

    cmd = [
        "/home/paul/ctl/venv/bin/locust",
        "--processes", "32",
        "-f", "./locust/hotel-reservations.py",
        "-H", "http://145.100.135.11:30505",
        "-t", LOAD_TEST_TIME,
        "--csv", label,
        "--headless",
        "--w-user-min", str(1000),
        "--w-user-max", str(6000),
        "--w-mean", str(3000),
        "--w-ls-y", str(10000),
        "--w-dt", str(40),
        "--seed", str(simple_hash(f"{base_label}_{loop}")) # str(simple_hash(f"fix_{loop}"))
    ]

    print(f"Test {label} started...\n")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1200  # 20 minutes
    )

    print(f"Test {label}, Finished")
    time.sleep(30)

    experiment_files = glob.glob(f"./{label}_*")

    os.makedirs(PATH + f"/output/{base_label}/data/{label}", exist_ok=True)
    for file in experiment_files:
        shutil.move(file, PATH + f"/output/{base_label}/data/{label}/")

    return


def read_data(label, loop):
    try:
        h_df = read_history(f"{label}_{loop}", label)
        r_df = read_response(f"{label}_{loop}", label)
        t_df = read_traces(f"{label}_{loop}", label)
        m_dfs = read_metrics(f"{label}_{loop}", label)

    except FileNotFoundError as fnfe:
        shutil.rmtree(PATH + f"/output/{label}/data/{label}_{loop}")
    
        print("reading the test data failed... retrying load test")
        locust_load_test(f"{label}_{loop}", label, loop)
        
        h_df, r_df, t_df, m_dfs = read_data(label, loop)

    return h_df, r_df, t_df, m_dfs

def parse_cpu_value(cpu_str):
    """
    Parse CPU value from Kubernetes format to cores.
    Supports formats like: "100m", "0.5", "1", "2.5"
    """
    if not cpu_str:
        return 0.0
    
    cpu_str = str(cpu_str).strip()
    
    try:
        if cpu_str.endswith('m'):
            # Millicores: "100m" -> 0.1 cores
            return int(cpu_str[:-1]) / 1000.0
        else:
            # Cores: "0.5", "1", "2.5" -> direct value
            return float(cpu_str)
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not parse CPU value '{cpu_str}': {e}")
        return 0.0

def metric_snapshot(service_name, trace_df, metric_dfs, anomaly_index, history_df, anomalies, knobs):
    # Get anomaly window
    start_time = trace_df["startTime"][anomaly_index]
    duration = pd.to_timedelta(trace_df["total"][anomaly_index], unit="us")
    start_plus_duration = start_time + duration
    start_minus_10s = start_time - pd.Timedelta(seconds=10)

    # Filter history_df for the window
    hist_window = history_df[(history_df["Timestamp"] >= start_minus_10s) & (history_df["Timestamp"] <= start_plus_duration)].copy()

    # Fill NaN values with the maximum value of each column
    if not hist_window.empty:
        for column in hist_window.columns:
            if column != "Timestamp":  # Skip timestamp column
                max_value = hist_window[column].max()
                if pd.notna(max_value):  # Only fill if max value is not NaN
                    hist_window[column] = hist_window[column].fillna(max_value)
                else:
                    # If all values in column are NaN, fill with 0
                    hist_window[column] = hist_window[column].fillna(0)
    else:
        # If hist_window is empty, fill with 0
        hist_window = hist_window.fillna(0)

    # Filter service metrics for the window
    m_df = metric_dfs.get(service_name, pd.DataFrame())
    if not m_df.empty:
        m_window = m_df[(m_df["index"] >= start_minus_10s) & (m_df["index"] <= start_plus_duration)].copy()
    else:
        m_window = pd.DataFrame()

    # Service max CPU usage in window
    service_max_cpu_usage = m_window[f"{service_name}_container_cpu_usage_seconds_total"].max() if not m_window.empty and f"{service_name}_container_cpu_usage_seconds_total" in m_window else None

    # Total max CPU usage in window
    total_max_cpu_usage = 0.0
    for k_service, k_df in metric_dfs.items():
        if not k_df.empty and f"{k_service}_container_cpu_usage_seconds_total" in k_df:
            k_window = k_df[(k_df["index"] >= start_minus_10s) & (k_df["index"] <= start_plus_duration)]
            if not k_window.empty:
                total_max_cpu_usage += k_window[f"{k_service}_container_cpu_usage_seconds_total"].max()

    # Service pod metrics in window
    actual_pods = m_window.get(f"{service_name}_autoscaler_actual_pods", pd.Series([None])).max()
    terminating_pods = m_window.get(f"{service_name}_autoscaler_terminating_pods", pd.Series([None])).max()
    requested_pods = m_window.get(f"{service_name}_autoscaler_requested_pods", pd.Series([None])).max()

    # Throughput and errors in window
    max_throughput = hist_window["Requests/s"].max() if not hist_window.empty else None
    total_errors = hist_window["Total Failure Count"].max() if not hist_window.empty else None

    # Latency percentiles in window
    p99 = hist_window["99%"].mean() if "99%" in hist_window else None
    p50 = hist_window["50%"].mean() if "50%" in hist_window else None

    # Anomaly counts in the window (SHAP-based)
    window_mask = (trace_df["startTime"] >= start_minus_10s) & (trace_df["startTime"] <= start_plus_duration)
    anomaly_df = trace_df[window_mask]

    # Count anomalies for this service in the window
    changed_service_anomaly_count = len([1 for a_s_name, ai, *_ in anomalies if service_name == a_s_name and ai in anomaly_df.index])
    total_anomaly_count = len([1 for _, ai, *_ in anomalies if ai in anomaly_df.index])

    pprint(knobs)

    # Calculate CPU utilization correctly
    try:
        cpu_request_str = knobs.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [{}])[0].get('resources', {}).get('requests', {}).get('cpu')
        cpu_request_cores = parse_cpu_value(cpu_request_str)
        total_requested_cpu = cpu_request_cores * actual_pods if actual_pods and cpu_request_cores else 0.0
        
        cpu_utilization_percent = 0.0
        if service_max_cpu_usage and total_requested_cpu > 0:
            cpu_utilization_percent = (service_max_cpu_usage / total_requested_cpu) * 100
    except (KeyError, IndexError, TypeError, ValueError) as e:
        print(f"Warning: Could not parse CPU request from configuration: {e}")
        cpu_utilization_percent = 0.0

    snapshot = pd.DataFrame([{
        "max_throughput": max_throughput,
        "total_errors": total_errors,
        "total_anomaly_count": total_anomaly_count,
        "changed_service_anomaly_count": changed_service_anomaly_count,
        "service_cpu_cores": service_max_cpu_usage,
        "service_cpu_utilization_of_requested": f"{cpu_utilization_percent:.0f}%",
        "system_cpu_cores_usage": total_max_cpu_usage,
        "system_cpu_cores_max": 14,
        "p99": p99,
        "p50": p50,
        "actual_pods": actual_pods,
        "terminating_pods": terminating_pods,
        "requested_pods": requested_pods,
    }])

    # Indent YAML for prompt
    snapshot_yaml = yaml.dump(snapshot.to_dict(orient='records')[0], default_flow_style=False, sort_keys=False)
    snapshot_yaml = textwrap.indent(snapshot_yaml, '  ')

    return start_time, duration, snapshot_yaml


def read_prompt(response):
    matches = re.findall(r"```yaml\n([\s\S]*?)\n```", response)

    yaml_str = ""
    for m in matches:
        yaml_str = yaml_str + m.strip() + "\n---\n"
    
    return yaml_str

def load_yaml_as_string(filepath):
    """
    Loads a YAML file and returns its string representation.
    Handles both regular YAML files and sequenced files.
    """
    data = load_yaml_as_dict(filepath)
    if data is None:
        return ""
    return yaml.dump(data)

def load_yaml_as_dict(filepath):
    """
    Loads a YAML file and returns its string representation for prompt insertion.
    Handles both regular YAML files and sequenced files (with # --- START: seq=X --- markers).
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if this is a sequenced file
    if "# --- START: seq=" in content:
        # This is a sequenced file, get the latest configuration
        return get_latest_config_from_sequenced_file(filepath)
    else:
        # This is a regular YAML file
        return yaml.safe_load(content)

def get_latest_config_from_sequenced_file(filepath):
    """
    Extracts the latest configuration from a sequenced file.
    Returns the configuration from the highest sequence number.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find all sequence blocks
    matches = re.findall(r'(?s)# --- START: seq=(\d+) ---\s*(.*?)\s*# --- END: seq=\1 ---', content)
    
    if not matches:
        return None
    
    # Get the highest sequence number
    latest_seq = max(int(seq) for seq, _ in matches)
    
    # Find the content for the latest sequence
    for seq, block in matches:
        if int(seq) == latest_seq:
            # Split the block to get just the configuration (before the --- separator)
            parts = block.strip().split('---', 1)
            if len(parts) >= 1:
                config_content = parts[0].strip()
                try:
                    return yaml.safe_load(config_content)
                except yaml.YAMLError as e:
                    print(f"Warning: Could not parse YAML from sequence {latest_seq} in {filepath}: {e}")
                    return None
    
    return None


# Find outliers in the traces using an IsolationForest classifier.
def trace_anomaly_detection(trace_df):
    features = trace_df.select_dtypes(include=["number"]).drop(columns=["total"])
    iso_forest = IsolationForest(contamination="auto", random_state=42)
    trace_df["anomaly"] = iso_forest.fit_predict(features)

    anomaly_indices = trace_df[(trace_df['anomaly'] == -1)].index.to_list()

    anom_features = features.iloc[anomaly_indices]
    shapes, names = shap_decisions(iso_forest, anom_features)
    
    # order the anomalies on their length.
    anomalies = []
    for s, ai in zip(shapes, anomaly_indices):
        duration = pd.to_timedelta(trace_df["total"][ai], unit="us")
        
        print(f"Anomaly {ai} lasted {duration}s")

        if duration < pd.Timedelta(seconds=2):
            print(f"\r -- Duration is less than 2s.")
            continue


        # Root Cause Analysis?
        value = heapq.nsmallest(1, enumerate(s), key=itemgetter(1))[0]
        
        service_name = SPAN_PROCESS_MAP[names[value[0]]]

        if service_name in ['mongo_rate']:
            continue
    
        anomalies.append((service_name, ai, duration))

    return sorted(anomalies, key=lambda x: -x[2])

@report_time
def generate_and_measure(label, loop):
    client = get_k8s_api_client()
    reset_k8s(client, PATH + f"/output/{label}/config" )
    
    history_df, responce_df, trace_df, metric_dfs = read_data(label, loop)
    anomalies = trace_anomaly_detection(trace_df)
    
    print(f"Generating and Measuring configuration changes from {len(anomalies)} anomalies")
    for i, anom in enumerate(anomalies):
        service_name, ai, _ = anom

        if(i > 10):
            print("Max iterations passed. starting next round... ")
            break

        # print(service_name)

        # Prepare data
        service_config = load_yaml_as_dict(PATH + f"/output/{label}/config/{service_name}.yaml")
        auto_config = load_yaml_as_dict(PATH + f"/output/{label}/config/config-autoscaler.yaml")
        
        if service_config is None:
            print(f"Warning: Could not load service configuration for {service_name}, skipping...")
            continue
            
        if auto_config is None:
            print(f"Warning: Could not load auto-configuration, skipping...")
            continue
        timestamp, duration, snapshot = metric_snapshot(service_name, trace_df, metric_dfs, ai, history_df, anomalies, service_config)

        prompt = GENERATE_PROMPT.format(
            knowledge_yaml=load_yaml_as_string(PATH + '/knowledge/knative_autoscaling_knowledge2.yaml'),
            service_name=service_name,
            revision_name=service_name,
            anomaly_type="latency spike",
            timestamp=timestamp,
            duration=duration,
            snapshot=snapshot,
            service_config=yaml.dump(service_config),
            auto_config=yaml.dump(auto_config)
        )

        message = [["system", "You are a Kubernetes expert. Please provide only a revised configuration that aims to resolvve the following anomaly"], ["user", prompt]]

        payload = {
            "messages": message,
            "max_new_tokens": 2000
        }

        response = requests.post(GEN_API_URL, json=payload)
        data = response.json()["response"]

        # save the prompt
        append_generation(PATH + f"/output/{label}/{label}_{loop}_prompts.json", ai, data)

        print(data)

        # read the configuration update
        configuration_update = read_prompt(data)
        
        print(configuration_update)

        configuration_update = list(yaml.safe_load_all(configuration_update))

        
        if configuration_update:
            error_count = 0
            # Apply the configuration update
            for config in configuration_update:
                print(config)
                try:
                    apply_yaml_configuration(config, client)
                except Exception as e:
                    print(f"Warning: Could not apply configuration: {e}")
                    error_count += 1
                    continue

            if error_count == len(configuration_update):
                print(f"Warning: All configurations failed to apply. Skipping load testing.")
                continue

            # Perform load testing using Locust
            post_update_label = f"{label}_{loop}_post_update_{ai}"
            locust_load_test(post_update_label, label, loop)

            # Collect performance data after configuration update
            try:
                # Get performance metrics for the updated configuration
                post_performance = get_kpi_list(post_update_label, label, service_name)
                
                # Log the performance data
                performance_file = PATH + f"/output/{label}/{label}_{loop}_performance.json"
                append_generation(performance_file, ai, json.dumps(post_performance))
                
                print(f"Performance after update: {post_performance}")
                
            except Exception as e:
                print(f"Warning: Could not collect performance data after update: {e}")
                post_performance = None
        
        reset_k8s(client, PATH + f"/output/{label}/config" )


def clean_float(value):
    try:
        val = float(value)
        return val if not math.isnan(val) else float(1000)
    except (ValueError, TypeError):
        return float('inf')


def evolve(label : str, loop: int):
    """
    Analyze updated configurations and their performance, then choose which configuration updates to accept.
    This function implements a multi-objective optimization approach to select the best configurations.
    """
    print(f"Starting evolution process for label: {label}, loop: {loop}")
    
    # Load generated configurations
    try:
        # Load the stored LLM responses and extract configurations the same way as generate_and_measure
        configs = []
        prompts_file = PATH + f"/output/{label}/{label}_{loop}_prompts.json"
        
        if os.path.exists(prompts_file):
            # Get all stored responses
            stored_responses = get_generation_content(prompts_file)
            
            for anomaly_id, response_data in stored_responses:
                try:
                    # Extract YAML from response using the same method as generate_and_measure
                    yaml_content = read_prompt(response_data)
                    if yaml_content:
                        # Parse YAML using the same method as generate_and_measure
                        parsed_configs = list(yaml.safe_load_all(yaml_content))
                        configs.append((anomaly_id, parsed_configs))
                except Exception as e:
                    print(f"Warning: Could not parse configuration for anomaly {anomaly_id}: {e}")
                    continue
        
        print(f"Loaded {len(configs)} generated configurations")
    except Exception as e:
        print(f"Error loading generated configurations: {e}")
        return {}

    # Load performance data
    try:
        trails = get_generation_content_perf(PATH + f"/output/{label}/{label}_{loop}_performance.json")
        perfs = [(i, json.loads(p)) for i, p in trails]
        perfs = [(i, p) for i, p in perfs if "Audit-Id" not in p]
        print(f"Loaded {len(perfs)} performance records")
    except Exception as e:
        print(f"Warning: Could not load performance data: {e}")
        perfs = []

    # Filter out configurations with poor throughput (below minimum threshold)
    min_throughput = 300
    perfs = [(i, p) for i, p in perfs if p.get('max_throughput', 0) >= min_throughput]
    print(f"After throughput filtering: {len(perfs)} configurations")

    perf_dict = dict(perfs)
    
    # Merge configurations with their performance data
    merged = [
        (i, config, perf_dict[i])
        for i, config in configs
        if i in perf_dict
    ]
    
    print(f"Successfully merged {len(merged)} configuration-performance pairs")

    if not merged:
        print("No valid configuration-performance pairs found. Skipping evolution.")
        return {}

    # Multi-objective optimization: Sort by multiple performance criteria
    # Lower values are better for all criteria
    sorted_data = sorted(
        merged,
        key=lambda item: (
            clean_float(item[2].get('total_anomaly_count', 0)),      # Fewer anomalies is better
            clean_float(item[2].get('changed_service_anomaly_count', 0)),  # Fewer service-specific anomalies is better
            clean_float(item[2].get('total_errors', 0)),             # Fewer errors is better
            clean_float(item[2].get('p99', 0)),                      # Lower p99 latency is better
            clean_float(item[2].get('p50', 0)),                      # Lower p50 latency is better
            clean_float(item[2].get('service_max_cpu_usage', 0)),    # Lower CPU usage is better
            clean_float(item[2].get('total_max_cpu_usage', 0)),      # Lower total CPU usage is better
            -clean_float(item[2].get('max_throughput', 0)),          # Higher throughput is better (negative for ascending sort)
        )
    )

    print(f"Sorted configurations by performance criteria")

    # Select the best configuration for each service
    best = {}
    for p in reversed(sorted_data):  # Reverse to get best first
        anomaly_id, config, performance = p
        
        service_names = []
        docs = []
        for d in config:            
            service_name = d.get('metadata', {}).get('name')
            if service_name:
                service_names.append(service_name)
                docs.append(d)
        
        # For each service in this configuration, check if it's the best we've seen
        for service_name in service_names:
            if service_name not in best or is_better_performance(performance, best[service_name][2]):
                best[service_name] = (anomaly_id, docs, performance)
                print(f"New best configuration for {service_name}: anomaly_id={anomaly_id}")

    # Apply the selected configurations
    print(f"\nApplying {len(best)} best configurations:")
    applied_count = 0
    for service_name, (anomaly_id, configs, performance) in best.items():
        fp = f"{PATH}/output/{label}/config/{service_name}.yaml"
        print(f"  Updating {service_name} (anomaly_id={anomaly_id})")
        print(f"    Performance: {performance}")
        print(f"    File: {fp}")
        
        # Find the configuration document for this specific service
        applied = False
        for d in configs:
            if service_name == d.get('metadata', {}).get('name'):
                try:
                    # Append the new configuration with performance data
                    next_seq = max(get_existing_seq(fp)) + 1
                    append_generation(fp, next_seq, yaml.dump(d) + "\n---\n" + str(performance))
                    print(f"    Applied configuration with sequence {next_seq}")
                    applied = True
                    applied_count += 1
                    break
                except Exception as e:
                    print(f"    Error applying configuration: {e}")
        
        if not applied:
            print(f"    Warning: No valid configuration found for {service_name}")

    print(f"Evolution completed. {applied_count} out of {len(best)} services updated successfully.")
    
    # Log evolution summary
    evolution_summary = {
        "label": label,
        "loop": loop,
        "total_configurations": len(configs),
        "valid_performance_records": len(perfs),
        "merged_pairs": len(merged),
        "services_updated": applied_count,
        "timestamp": time.time(),
        "updated_services": list(best.keys()) if best else []
    }
    
    summary_file = PATH + f"/output/{label}/{label}_{loop}_evolution_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(evolution_summary, f, indent=2)
    
    print(f"Evolution summary saved to: {summary_file}")
    
    # Print detailed summary
    print_evolution_summary(evolution_summary, best)
    
    return best

def print_evolution_summary(summary, best_configs):
    """
    Print a detailed summary of the evolution results.
    """
    print(f"\n{'='*60}")
    print(f"EVOLUTION SUMMARY - {summary['label']} (Loop {summary['loop']})")
    print(f"{'='*60}")
    print(f"Total configurations analyzed: {summary['total_configurations']}")
    print(f"Valid performance records: {summary['valid_performance_records']}")
    print(f"Configuration-performance pairs: {summary['merged_pairs']}")
    print(f"Services updated: {summary['services_updated']}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(summary['timestamp']))}")
    
    if best_configs:
        print(f"\nUpdated Services:")
        for service_name, (anomaly_id, configs, performance) in best_configs.items():
            print(f"  • {service_name} (anomaly_id: {anomaly_id})")
            print(f"    - Anomalies: {performance.get('total_anomaly_count', 'N/A')}")
            print(f"    - Errors: {performance.get('total_errors', 'N/A')}")
            print(f"    - P99 Latency: {performance.get('p99', 'N/A')}ms")
            print(f"    - Throughput: {performance.get('max_throughput', 'N/A')} req/s")
    else:
        print(f"\nNo services were updated.")
    
    print(f"{'='*60}")

def is_better_performance(new_perf, old_perf):
    """
    Compare two performance records and return True if new_perf is better than old_perf.
    Uses a weighted scoring system to evaluate overall performance.
    """
    # Define weights for different performance metrics (higher weight = more important)
    weights = {
        'total_anomaly_count': 10.0,
        'changed_service_anomaly_count': 8.0,
        'total_errors': 6.0,
        'p99': 5.0,
        'p50': 4.0,
        'max_throughput': 3.0,
        'service_max_cpu_usage': 2.0,
        'total_max_cpu_usage': 1.0,
    }
    
    def calculate_score(perf):
        score = 0.0
        for metric, weight in weights.items():
            value = clean_float(perf.get(metric, 0))
            if metric == 'max_throughput':
                # Higher throughput is better, so we add it positively
                score += value * weight
            else:
                # Lower values are better for other metrics, so we subtract
                score -= value * weight
        return score
    
    new_score = calculate_score(new_perf)
    old_score = calculate_score(old_perf)
    
    return new_score > old_score

@report_time
def main():
    LABEL ="naturally"
    
    os.makedirs(PATH + f"/output/{LABEL}", exist_ok=True)
    if(glob.glob(PATH + f"/output/{LABEL}/config/*.yaml") == []):
        shutil.copytree(PATH + "/base_config", PATH + f"/output/{LABEL}/config", dirs_exist_ok = True)
    else:
        print("base configuration files exist... skipping")

    print("Entering optimization loop:")
    for i in range(0, 1):
        print(f"########################### Loop: {i} ###########################")

        print("Load test to analyse present anomalies in the current configuration")
        client = get_k8s_api_client()
        reset_k8s(client, PATH + f"/output/{LABEL}/config")
        locust_load_test(label=f"{LABEL}_{i}", base_label=LABEL, loop=i)

        print("Generate possible changes to mitigate then Apply and measure the changes")
        generate_and_measure(LABEL, i)

        print("Select the configurations that have performed the best and evolve")
        evolve(LABEL, i)

if __name__=="__main__":
    main()
