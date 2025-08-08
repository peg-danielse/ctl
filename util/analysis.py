import sys, os, json, glob, re, requests, time, subprocess, heapq, yaml, shutil

import numpy as np
import pandas as pd

import shap
from sklearn.ensemble import IsolationForest

from typing import List, Tuple

from datetime import datetime
from functools import wraps
from operator import itemgetter

from config import PATH, SPAN_PROCESS_MAP

# read stat history
def read_history(label, base_label):
    history_df = pd.read_csv(PATH + f"/output/{base_label}/data/{label}/" + f'{label}_stats_history.csv')
    history_df["Timestamp"] = pd.to_datetime(history_df['Timestamp'], unit='s')

    return history_df

# read response time data
def read_response(label, base_label):
    resp_df = pd.read_csv(PATH + f"/output/{base_label}/data/{label}/" + f'{label}_responce_log.csv')
    
    return resp_df

# read Jaeger trace data
def read_traces(label, base_label):
    data = {}
    with open(PATH + f"/output/{base_label}/data/{label}/" + f'{label}_traces.json', 'r') as file:
        data = json.load(file)

    rows = []
    for trace in data["data"]:
        row = {"id": trace['traceID']}
        
        total = 0
        st = sys.maxsize
        for s in trace["spans"]:
            st = min(st, s["startTime"])
            total += s["duration"]
            row[s["operationName"]] =  s["duration"]
        
        row["startTime"] = st
        row["total"] = total
        rows.append(row)

    # fill NaN
    trace_df = pd.DataFrame(rows).fillna(0)
    
    # create trace patterns.
    span_cols = trace_df.columns.difference(['id'])
    trace_df["pattern"] = trace_df[span_cols].gt(0).astype(int).astype(str).agg("".join, axis=1)

    # convert times
    trace_df["startTime"] = pd.to_datetime(trace_df['startTime'], unit='us')

    return trace_df

def read_metrics(label, base_label):
    metrics = {}
    for n in glob.glob(PATH + f"/output/{base_label}/data/{label}/" + f'{label}_*_metrics.csv'):
        match = re.search(r"[^_/]+-[^_]+(?=_metrics\.csv)", n)
        name = "unknown"
        if match:
            name=match.group()
        else:
            print("unregognized file:", n, "skipping")
            continue

        metric_df = pd.read_csv(n, index_col=False).drop('Unnamed: 0', axis=1)
        metric_columns = metric_df.columns.drop('index')
        metric_df["index"] = pd.to_datetime(metric_df['index'], unit='s')

        metrics[name] = metric_df
    
    return metrics

def shap_decisions(iso_forest, features, mark = "_"):
    # try to explain a specific data points. as a short cut for RCA.
    shap_values = shap.TreeExplainer(iso_forest).shap_values(features)

    return shap_values, features.columns.tolist()

def get_kpi_list(label: str, base_label: str, service) -> List:
    history_df = read_history(label, base_label)
    responce_df = read_response(label, base_label)
    trace_df = read_traces(label, base_label)
    metric_dfs = read_metrics(label, base_label)

    for key, m_df in metric_dfs.items():
        m_df.fillna(0, inplace=True)

    # IMPROVEMENT: improve pipeline with the ELBD framework. will look good in the paper.
    # IMPROVEMENT: also perform outlier detection on the monitoring metrics.

    # Find outliers in the traces using an IsolationForest classifier.
    features = trace_df.select_dtypes(include=["number"]).drop(columns=["total"])
    iso_forest = IsolationForest(contamination="auto", random_state=42)
    trace_df["anomaly"] = iso_forest.fit_predict(features)

    # IMPROVEMENT: decision plot the shap values by clustering similar shap values...
    anomaly_indices = trace_df[(trace_df['anomaly'] == -1)].index.to_list()
    anom_features = features.iloc[anomaly_indices]
    shapes, names = shap_decisions(iso_forest, anom_features)

    service_anomaly_count = {}
    for s, ai in zip(shapes, anomaly_indices):
        duration = pd.to_timedelta(trace_df["total"][ai], unit="us")

        if duration < pd.Timedelta(seconds=2):
            continue

        values = heapq.nsmallest(2, enumerate(s), key=itemgetter(1))
        for v in values:
            if names[v[0]] in ['mongo_rate']:
                continue

            service_name = os.path.basename(SPAN_PROCESS_MAP[names[v[0]]])[:-5] 
            service_anomaly_count[service_name] = service_anomaly_count.get(service_name, 0) + 1

            break

    service_max_cpu_usage = max(metric_dfs.get(service, {}).get(f"{service}_container_cpu_usage_seconds_total", [-1]))
    
    total_max_cpu_usage = 0.0
    for k_service, m_df in metric_dfs.items():
        total_max_cpu_usage = total_max_cpu_usage + max(m_df.get(f"{k_service}_container_cpu_usage_seconds_total",[0]))

    kpi = {
        "max_throughput": max(history_df["Requests/s"]),
        "total_errors" : max(history_df["Total Failure Count"]),
        "total_anomaly_count": sum(service_anomaly_count.values()),
        "changed_service_anomaly_count": service_anomaly_count.get(service, 0),
        "service_max_cpu_usage": service_max_cpu_usage,
        "total_max_cpu_usage": total_max_cpu_usage,
        "p99": sum(history_df["99%"])/len(history_df),
        "p50": sum(history_df["50%"])/len(history_df),
    }

    return kpi