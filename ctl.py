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
        "--seed", str(simple_hash(f"fix_{loop}")) #  str(simple_hash(f"{base_label}}_{loop}"))
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

def metric_snapshot(service_name, trace_df, metric_dfs, anomaly_index, history_df, anomalies, knobs):
    # Get anomaly window
    start_time = trace_df["startTime"][anomaly_index]
    duration = pd.to_timedelta(trace_df["total"][anomaly_index], unit="us")
    start_plus_duration = start_time + duration
    start_minus_10s = start_time - pd.Timedelta(seconds=10)

    # Filter history_df for the window
    hist_window = history_df[(history_df["Timestamp"] >= start_minus_10s) & (history_df["Timestamp"] <= start_plus_duration)].copy()

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

    snapshot = pd.DataFrame([{
        "max_throughput": max_throughput,
        "total_errors": total_errors,
        "total_anomaly_count": total_anomaly_count,
        "changed_service_anomaly_count": changed_service_anomaly_count,
        "service_cpu_cores": service_max_cpu_usage,
        "service_cpu_utilization_of_requested": ("%.0f" % (service_max_cpu_usage / ((int(knobs['requested_cpu'][:-1]) / 1000.0) * actual_pods) * 100)) + '%',
        "system_cpu_cores_usage": total_max_cpu_usage,
        "system_cpu_cores_max": 14,
        "p99": p99,
        "p50": p50,
        "actual_pods": actual_pods,
        "terminating_pods": terminating_pods,
        "requested_pods": requested_pods,
    }])

    # print(service_max_cpu_usage, int(knobs['requested_cpu'][:-1]), int(knobs['requested_cpu'][:-1]) / 1000, actual_pods, service_max_cpu_usage / ((int(knobs['requested_cpu'][:-1]) / 1000.0) * actual_pods))

    # Indent YAML for prompt
    snapshot_yaml = yaml.dump(snapshot.to_dict(orient='records')[0], default_flow_style=False, sort_keys=False)
    snapshot_yaml = textwrap.indent(snapshot_yaml, '  ')

    return start_time, duration, snapshot_yaml


def read_prompt(response):
    matches = re.findall(r"```yaml\n([\s\S]*?)\n```", response.json()["response"])

    yaml_str = ""
    for m in matches:
        yaml_str = m.strip()
    
    return yaml_str

def get_knative_knobs(service_config, auto_config=None):
    """
    Extracts Knative autoscaling annotations and requested CPU from a Knative service config.
    """
    knobs = {}
    # Traverse to annotations
    try:
        annotations = service_config['spec']['template']['metadata']['annotations']
        for k, v in annotations.items():
            if k.startswith('autoscaling.knative.dev/'):
                knobs[k] = v
    except Exception:
        pass

    # Get requested CPU
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
    Updates the knative autoscaling annotations and requested CPU in the given service YAML file.
    knob_values: dict of {knob_name: value}
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
    Extracts all relevant vscaling/autoscaler keys from the config-autoscaler.yaml.
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
    Updates the vscaling/autoscaler keys in the file.
    knob_values: dict of {knob_name: value}
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

def load_yaml_as_string(filepath):
    """
    Loads a YAML file.
    """
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return yaml.dump(data)

def load_yaml_as_dict(filepath):
    """
    Loads a YAML file and returns its string representation for prompt insertion.
    """
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data


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
    # client = get_k8s_api_client()
    # reset_k8s(client, PATH + f"/output/{label}/config" )
    
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

        kn_knobs = get_knative_knobs(service_config, auto_config)
        vscale_knobs = get_vscaling_knobs(service_config, auto_config)



        pprint(kn_knobs)
        pprint(vscale_knobs)

        timestamp, duration, snapshot = metric_snapshot(service_name, trace_df, metric_dfs, ai, history_df, anomalies, kn_knobs)

        # In the prompt construction, update kn_knobs and vscale_knobs to be indented YAML blocks
        kn_knobs_yaml = textwrap.indent(yaml.dump(kn_knobs, default_flow_style=False), '  ')
        vscale_knobs_yaml = textwrap.indent(yaml.dump(vscale_knobs, default_flow_style=False), '  ')

        prompt = GENERATE_PROMPT.format(
            knowledge_yaml=load_yaml_as_string(PATH + '/knowledge/knative_autoscaling_knowledge2.yaml'),
            service_name=service_name,
            revision_name=service_name,
            anomaly_type="latency spike",
            timestamp=timestamp,
            duration=duration,
            snapshot=snapshot,
            kn_knobs=yaml.dump(service_config),
            vscale_knobs=yaml.dump(auto_config)
        )

        print(prompt)

        message = [["user", prompt]]

        payload = {
            "messages": message,
            "max_new_tokens": 2000
        }

        response = requests.post(GEN_API_URL, json=payload)
        data = response.json()["response"]


        pprint(data)
        break

        # save the prompt
        append_generation(PATH + f"/output/{label}/{label}_{loop}_prompts.json", ai, data) 

        # read the configuration update
        configuration_update = read_prompt(data)

        print(configuration_update)

        reset_k8s(client, PATH + f"/output/{label}/config/*.yaml" )

        break # temporary


def clean_float(value):
    try:
        val = float(value)
        return val if not math.isnan(val) else float(1000)
    except (ValueError, TypeError):
        return float('inf')


def evolve(label : str, loop: int):
    configs = load_generated_configurations(label, loop)
    configs = [(i, yaml.safe_load_all(sol)) for i, sol in configs]

    trails = get_generation_content_perf(PATH + f"/output/{label}/{label}_{loop}_performance.json")
    perfs = [(i, yaml.safe_load(p)) for i, p in trails]
    perfs = [(i, p) for i, p in perfs if "Audit-Id" not in p]

    perfs = [(i, p) for i, p in perfs if 300 <= p.get('max_throughput', 0)]

    perf_dict = dict(perfs)
    # Step 4: Filter configs to include only those with a corresponding perf entry
    merged = [
        (i, config, perf_dict[i])
        for i, config in configs
        if i in perf_dict
    ]

    sorted_data = sorted(
        merged,
        key=lambda item: (
            clean_float(item[2].get('total_anomaly_count')),
            clean_float(item[2].get('changed_service_anomaly_count')),
            clean_float(item[2].get('total_errors')),
            clean_float(item[2].get('service_max_cpu_usage')),
            clean_float(item[2].get('total_max_cpu_usage')),
        )
    )

    best = {}
    for p in reversed(sorted_data):
        # print(p[0], p[2])

        name = []
        docs = []
        for d in p[1]:
            name.append(d.get('metadata', {}).get('name'))
            docs.append(d)
            # print(d.get('metadata', {}).get('name'))
        
        for n in name:
            best[n] = (p[0], docs, p[2])
    

    for service_name, (s, c, p) in best.items():
        fp = f"{PATH}/output/{label}/config/{service_name}.yaml"
        print(f"update configuration for {service_name} \n performance: {p} \n file : {fp}")
        for d in c:
            if service_name == d.get('metadata', {}).get('name'):
                append_generation(fp, max(get_existing_seq(fp)) + 1, yaml.dump(d) + "\n---\n" + str(p)) 

    return best

@report_time
def main():
    LABEL ="keep"
    
    os.makedirs(PATH + f"/output/{LABEL}", exist_ok=True)
    if(glob.glob(PATH + f"/output/{LABEL}/config/*.yaml") == []):
        shutil.copytree(PATH + "/base_config", PATH + f"/output/{LABEL}/config", dirs_exist_ok = True)
    else:
        print("base configuration files exist... skipping")

    print("Entering optimization loop:")
    for i in range(0, 1):
        print(f"########################### Loop: {i} ###########################")

        print("Load test to analyse present anomalies in the current configuration")
        # client = get_k8s_api_client()
        # reset_k8s(client, PATH + f"/output/{LABEL}/config")
        # locust_load_test(label=f"{LABEL}_{i}", base_label=LABEL, loop=i)

        print("Generate possible changes to mitigate then Apply and measure the changes")
        generate_and_measure(LABEL, i)

        # print("select the configurations that have performed the best and evolve")
        # evolve(LABEL, i)

if __name__=="__main__":
    main()

