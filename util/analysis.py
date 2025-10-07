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
    
    # Remove the "Name" column which contains "Aggregated" string values
    if "Name" in history_df.columns:
        history_df = history_df.drop(columns=["Name"])

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
    """
    Get KPI list without using history_df. Uses trace data and metrics instead.
    """
    from util.monitoring import calculate_latency_metrics_from_traces
    
    responce_df = read_response(label, base_label)
    trace_df = read_traces(label, base_label)
    metric_dfs = read_metrics(label, base_label)

    # Fill NaN values in metrics
    for key, m_df in metric_dfs.items():
        m_df = m_df.fillna(0)

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

            service_name = os.path.basename(SPAN_PROCESS_MAP[names[v[0]]])
            service_anomaly_count[service_name] = service_anomaly_count.get(service_name, 0) + 1

            break

    # Helper function to safely get numeric max value
    def safe_numeric_max(series, default=0.0):
        """Safely get the maximum numeric value from a series, handling non-numeric data"""
        try:
            # Convert to numeric, coercing errors to NaN
            numeric_series = pd.to_numeric(series, errors='coerce')
            # Drop NaN values and get max
            max_val = numeric_series.dropna().max()
            return max_val if pd.notna(max_val) else default
        except Exception as e:
            print(f"Warning: Could not calculate max for series: {e}")
            return default

    # Helper function to safely get numeric mean value
    def safe_numeric_mean(series, default=0.0):
        """Safely get the mean numeric value from a series, handling non-numeric data"""
        try:
            # Convert to numeric, coercing errors to NaN
            numeric_series = pd.to_numeric(series, errors='coerce')
            # Drop NaN values and get mean
            mean_val = numeric_series.dropna().mean()
            return mean_val if pd.notna(mean_val) else default
        except Exception as e:
            print(f"Warning: Could not calculate mean for series: {e}")
            return default

    # Safely calculate service max CPU usage
    service_cpu_series = metric_dfs.get(service, {}).get(f"{service}_container_cpu_usage_seconds_total", pd.Series([0]))
    service_max_cpu_usage = safe_numeric_max(service_cpu_series, default=0.0)
    
    # Safely calculate total max CPU usage
    total_max_cpu_usage = 0.0
    for k_service, m_df in metric_dfs.items():
        cpu_series = m_df.get(f"{k_service}_container_cpu_usage_seconds_total", pd.Series([0]))
        total_max_cpu_usage += safe_numeric_max(cpu_series, default=0.0)

    print("changed_service_anomaly_count", service_anomaly_count.get(service, "not found"))
    print(service_anomaly_count)

    # Calculate latency metrics from trace data (replaces history_df)
    latency_metrics = calculate_latency_metrics_from_traces(trace_df)

    # Safely calculate KPI values using trace data instead of history_df
    kpi = {
        "max_throughput": latency_metrics["max_throughput"],
        "total_errors": latency_metrics["total_errors"],
        "total_anomaly_count": sum(service_anomaly_count.values()),
        "changed_service_anomaly_count": service_anomaly_count.get(service, 0),
        "service_max_cpu_usage": service_max_cpu_usage,
        "total_max_cpu_usage": total_max_cpu_usage,
        "p99": latency_metrics["p99"],
        "p50": latency_metrics["p50"],
    }

    return kpi


def metric_snapshot(service_name, trace_df, metric_dfs, phase=None, subphase=None):
    """
    Create a metric snapshot for a specific service and time period.
    
    Args:
        service_name: Name of the service
        trace_df: Trace data DataFrame
        metric_dfs: Dictionary of metric DataFrames
        anomaly_index: Index of the anomaly (optional)
        anomalies: List of anomalies (optional)
        knobs: Configuration knobs (optional)
        phase: Current phase (baseline, adaptation, etc.) (optional)
        subphase: Current subphase (configuration_application, stabilization, etc.) (optional)
        
    Returns:
        tuple: (timestamp, duration, snapshot_dict)
    """
    timestamp = datetime.utcnow()
    
    # Calculate basic metrics from trace data
    if not trace_df.empty:
        # Calculate response time metrics
        if 'total' in trace_df.columns:
            response_times = pd.to_numeric(trace_df['total'], errors='coerce')
            duration = response_times.mean() if not response_times.empty else 0
        else:
            duration = 0
        
        # Count requests and errors
        total_requests = len(trace_df)
        error_count = len(trace_df[trace_df.get('error', False) == True]) if 'error' in trace_df.columns else 0
        error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0
    else:
        duration = 0
        total_requests = 0
        error_count = 0
        
    # Get service-specific metrics
    service_metrics = {}
    if service_name in metric_dfs:
        service_data = metric_dfs[service_name]
        
        # Handle both DataFrame and dict formats
        if hasattr(service_data, 'columns'):
            # DataFrame format
            for col in service_data.columns:
                if 'cpu' in col.lower():
                    service_metrics['cpu_usage'] = service_data[col].mean() if not service_data[col].empty else 0
                elif 'memory' in col.lower():
                    service_metrics['memory_usage'] = service_data[col].mean() if not service_data[col].empty else 0
        elif isinstance(service_data, dict):
            # Dictionary format - extract available metrics
            for key, value in service_data.items():
                if 'cpu' in key.lower():
                    service_metrics['cpu_usage'] = value if isinstance(value, (int, float)) else 0
                elif 'memory' in key.lower():
                    service_metrics['memory_usage'] = value if isinstance(value, (int, float)) else 0
    
    # Calculate per-pattern deadline miss rates
    pattern_miss_rates = calculate_pattern_miss_rates(trace_df)
    
    # Perform SHAP anomaly detection
    shap_results = perform_shap_anomaly_detection(trace_df)
    
    # Calculate overall pattern stats using 65th percentile (same as individual patterns)
    overall_miss_rate = 0
    overall_miss_count = 0
    if not trace_df.empty and 'total' in trace_df.columns:
        overall_response_times = pd.to_numeric(trace_df['total'], errors='coerce')
        if not overall_response_times.empty:
            overall_p65_threshold = overall_response_times.quantile(0.65)
            overall_miss_count = len(overall_response_times[overall_response_times > overall_p65_threshold])
            overall_miss_rate = (overall_miss_count / len(overall_response_times) * 100)
    
    # Calculate additional metrics
    p90_response_time = 0
    if not trace_df.empty and 'total' in trace_df.columns:
        response_times = pd.to_numeric(trace_df['total'], errors='coerce')
        if not response_times.empty:
            p90_response_time = response_times.quantile(0.90)
    
    # Calculate overall CPU utilization from service metrics
    overall_cpu_utilization = 0
    cpu_values = []
    for service_name in metric_dfs.keys():
        if service_name in metric_dfs:
            service_data = metric_dfs[service_name]
            if hasattr(service_data, 'columns'):
                for col in service_data.columns:
                    if 'cpu' in col.lower():
                        cpu_val = service_data[col].mean() if not service_data[col].empty else 0
                        if cpu_val > 0:
                            cpu_values.append(cpu_val)
    
    if cpu_values:
        overall_cpu_utilization = sum(cpu_values) / len(cpu_values)
    
    # Calculate anomaly rate
    anomaly_rate = (shap_results['anomaly_count'] / total_requests * 100) if total_requests > 0 else 0
    
    # Restructure snapshot data according to the specified format
    snapshot = {
        'timestamp': timestamp,
        'total_requests': total_requests,
        'mean_response_time': duration,
        'p90_response_time': p90_response_time,
        'deadline_miss_rate': overall_miss_rate,
        'patterns': {},
        'cpu_utilization': overall_cpu_utilization,
        'anomaly_count': shap_results['anomaly_count'],
        'anomaly_rate': anomaly_rate,
        'services': {},
        'phase': phase,
        'subphase': subphase
    }
    
    # Add patterns with the specified structure
    for pattern, stats in pattern_miss_rates.items():
        # Calculate p90 for this pattern
        pattern_p90 = 0
        if not trace_df.empty:
            pattern_traces = trace_df[trace_df['pattern'] == pattern]
            if not pattern_traces.empty and 'total' in pattern_traces.columns:
                pattern_response_times = pd.to_numeric(pattern_traces['total'], errors='coerce')
                if not pattern_response_times.empty:
                    pattern_p90 = pattern_response_times.quantile(0.90)
        
        snapshot['patterns'][pattern] = {
            'deadline_miss_rate': stats['miss_rate'],
            'mean_response_time': stats['mean_response_time'],
            'p90_response_time': pattern_p90
        }
    
    # Add service-specific data with the specified structure
    for service_name in metric_dfs.keys():
        if service_name in metric_dfs:
            service_data = metric_dfs[service_name]
            
            # Get CPU utilization for this service
            service_cpu_utilization = 0
            if hasattr(service_data, 'columns'):
                for col in service_data.columns:
                    if 'cpu' in col.lower():
                        service_cpu_utilization = service_data[col].mean() if not service_data[col].empty else 0
                        break
            
            # Get anomaly information for this service from SHAP results
            service_anomalies = shap_results['anomalies_by_service'].get(service_name, [])
            service_anomaly_count = len(service_anomalies)
            service_anomaly_rate = (service_anomaly_count / total_requests * 100) if total_requests > 0 else 0
            
            snapshot['services'][service_name] = {
                'anomaly_count': service_anomaly_count,
                'anomaly_rate': service_anomaly_rate,
                'cpu_utilization': service_cpu_utilization
            }
    
    return timestamp, duration, snapshot


def calculate_pattern_miss_rates(trace_df):
    """
    Calculate deadline miss rates for each trace pattern using 65th percentile as threshold.
    
    Args:
        trace_df: DataFrame with trace data including 'pattern' and 'total' columns
        
    Returns:
        dict: Pattern miss rates with pattern as key and miss rate as value
    """
    if trace_df is None or trace_df.empty or 'pattern' not in trace_df.columns or 'total' not in trace_df.columns:
        return {}
    
    pattern_miss_rates = {}
    
    # Group by pattern and calculate miss rates
    for pattern in trace_df['pattern'].unique():
        if pd.isna(pattern) or pattern == '':
            continue
            
        pattern_traces = trace_df[trace_df['pattern'] == pattern]
        if pattern_traces.empty:
            continue
            
        response_times = pd.to_numeric(pattern_traces['total'], errors='coerce')
        if response_times.empty:
            continue
            
        # Skip patterns with fewer than 30 traces for reliable statistical analysis
        if len(response_times) < 30:
            continue
            
        # Calculate 65th percentile as deadline threshold for this pattern
        p65_threshold = response_times.quantile(0.65)
        
        # Count requests above the 65th percentile
        deadline_misses = len(response_times[response_times > p65_threshold])
        total_requests = len(response_times)
        miss_rate = (deadline_misses / total_requests * 100) if total_requests > 0 else 0
        
        pattern_miss_rates[pattern] = {
            'miss_rate': miss_rate,
            'miss_count': deadline_misses,
            'total_requests': total_requests,
            'mean_response_time': response_times.mean(),
            'max_response_time': response_times.max()
        }
    
    return pattern_miss_rates


def perform_shap_anomaly_detection(trace_df):
    """
    Perform SHAP-based anomaly detection on trace data and sort by service.
    Uses the same attribution method as get_kpi_list for consistency.
    
    Args:
        trace_df: DataFrame with trace data
        
    Returns:
        dict: SHAP anomaly detection results sorted by service
    """
    if trace_df is None or trace_df.empty:
        return {
            'anomaly_count': 0,
            'anomalies_by_service': {},
            'shap_contributions': {}
        }
    
    try:
        from sklearn.ensemble import IsolationForest
        import shap
        import heapq
        from operator import itemgetter
        import os
        from config import SPAN_PROCESS_MAP
        
        # Prepare features for anomaly detection (exclude non-numeric columns)
        feature_columns = trace_df.select_dtypes(include=["number"]).columns.tolist()
        # Remove 'total' column as it's the target, not a feature
        if 'total' in feature_columns:
            feature_columns.remove('total')
        
        if not feature_columns:
            return {
                'anomaly_count': 0,
                'anomalies_by_service': {},
                'shap_contributions': {}
            }
        
        features = trace_df[feature_columns]
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination="auto", random_state=42)
        anomaly_predictions = iso_forest.fit_predict(features)
        
        # Find anomalies
        anomaly_indices = trace_df[anomaly_predictions == -1].index.tolist()
        
        if not anomaly_indices:
            return {
                'anomaly_count': 0,
                'anomalies_by_service': {},
                'shap_contributions': {}
            }
        
        # Use the same SHAP calculation method as get_kpi_list
        anom_features = features.iloc[anomaly_indices]
        shapes, names = shap_decisions(iso_forest, anom_features)
        
        # Process anomalies using the same attribution logic as get_kpi_list
        service_anomaly_count = {}
        anomalies_by_service = {}
        shap_contributions = {}
        
        for s, ai in zip(shapes, anomaly_indices):
            # Get duration and skip anomalies with duration < 2 seconds (same as get_kpi_list)
            duration = pd.to_timedelta(trace_df["total"][ai], unit="us")
            
            # Get the 2 most negative SHAP values (same as get_kpi_list)
            values = heapq.nsmallest(2, enumerate(s), key=itemgetter(1))
            
            # Find the primary service using the same logic as get_kpi_list
            primary_service = None
            top_contributors = []
            
            for v in values:
                feature_name = names[v[0]]
                shap_value = v[1]
                
                # Skip certain features (same as get_kpi_list)
                if feature_name in ['mongo_rate']:
                    continue
                
                # Map feature to service using SPAN_PROCESS_MAP and os.path.basename (same as get_kpi_list)
                if feature_name in SPAN_PROCESS_MAP:
                    service_name = os.path.basename(SPAN_PROCESS_MAP[feature_name])
                    if primary_service is None:  # Use the first valid service as primary
                        primary_service = service_name
                    
                    top_contributors.append({'feature': feature_name, 'shap_value': float(shap_value)})
                    
                    # Count anomalies per service (same as get_kpi_list)
                    service_anomaly_count[service_name] = service_anomaly_count.get(service_name, 0) + 1
                
                break  # Only process the first valid contributor (same as get_kpi_list)
            
            if primary_service is None:
                continue
            
            # Store anomaly information
            anomaly_info = {
                'index': int(ai),
                'trace_id': trace_df.loc[ai, 'id'] if 'id' in trace_df.columns else None,
                'pattern': trace_df.loc[ai, 'pattern'] if 'pattern' in trace_df.columns else None,
                'response_time': trace_df.loc[ai, 'total'] if 'total' in trace_df.columns else None,
                'duration_seconds': duration.total_seconds(),
                'service': primary_service
            }
            
            # Group anomalies by service
            if primary_service not in anomalies_by_service:
                anomalies_by_service[primary_service] = []
            anomalies_by_service[primary_service].append(anomaly_info)
            
            # Aggregate contributions for overall analysis
            for contrib in top_contributors:
                feature_name = contrib['feature']
                shap_value = contrib['shap_value']
                if feature_name not in shap_contributions:
                    shap_contributions[feature_name] = {'count': 0, 'total_contribution': 0.0}
                shap_contributions[feature_name]['count'] += 1
                shap_contributions[feature_name]['total_contribution'] += abs(shap_value)
        
        # Calculate average contributions
        for feature_name in shap_contributions:
            contrib = shap_contributions[feature_name]
            contrib['average_contribution'] = contrib['total_contribution'] / contrib['count']
        
        # Sort anomalies within each service by response time (highest first)
        for service in anomalies_by_service:
            anomalies_by_service[service].sort(key=lambda x: x['response_time'], reverse=True)
        
        # Calculate total anomaly count (only counting anomalies > 2 seconds)
        total_anomaly_count = sum(service_anomaly_count.values())
        
        return {
            'anomaly_count': total_anomaly_count,
            'anomalies_by_service': anomalies_by_service,
            'shap_contributions': shap_contributions,
            'service_anomaly_counts': service_anomaly_count
        }
        
    except Exception as e:
        print(f"Error in SHAP anomaly detection: {e}")
        return {
            'anomaly_count': 0,
            'anomalies_by_service': {},
            'shap_contributions': {}
        }
