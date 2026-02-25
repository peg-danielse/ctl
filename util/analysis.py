import sys, json, glob, re, os
import logging
import traceback

import numpy as np
import pandas as pd

import shap
from sklearn.ensemble import IsolationForest

from datetime import datetime, timezone

from config import PATH

# Configure logging
logger = logging.getLogger(__name__)

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
def read_traces(path):
    data = {}
    with open(path, 'r') as file:
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
            # logger.warning(f"unrecognized file: {n}, skipping")
            continue

        metric_df = pd.read_csv(n, index_col=False).drop('Unnamed: 0', axis=1)
        metric_columns = metric_df.columns.drop('index')
        metric_df["index"] = pd.to_datetime(metric_df['index'], unit='s')

        metrics[name] = metric_df
    
    return metrics


def shap_decisions(iso_forest, features):
    # try to explain a specific data points. as a short cut for RCA.
    shap_values = shap.TreeExplainer(iso_forest).shap_values(features)

    return shap_values, features.columns.tolist()

def metric_snapshot(trace_df, metric_dfs, phase=None, subphase=None):
    # Create a metric snapshot for a specific service and time period.
    timestamp = datetime.now().astimezone(timezone.utc)
    
    try:
        response_times = pd.to_numeric(trace_df['total'], errors='coerce')
        duration = response_times.mean() if not response_times.empty else 0
        total_requests = len(trace_df)

    except Exception as e:
        logger.error(f"Error in snapshot: {e}")
        duration = 0
        total_requests = 0

        return {"error": str(e)}
    
    # Calculate per-pattern deadline miss rates
    pattern_miss_rates = calculate_pattern_miss_rates(trace_df)
    if pattern_miss_rates:
        overall_miss_rate = sum(e['miss_rate'] for e in pattern_miss_rates.values()) / len(pattern_miss_rates)
    else:
        overall_miss_rate = 0
    
    # Perform SHAP anomaly detection
    anomaly_results = perform_shap_anomaly_detection(trace_df)
    
    # Calculate overall pattern stats using 65th percentile (same as individual patterns)
    
    # Calculate additional metrics
    p90_response_time = 0
    if not trace_df.empty and 'total' in trace_df.columns:
        response_times = pd.to_numeric(trace_df['total'], errors='coerce')
        if not response_times.empty:
            p90_response_time = response_times.quantile(0.90)
    
    # Calculate overall CPU (cores) and memory (MB) utilization from service metrics
    overall_cpu_utilization_cores = 0
    overall_memory_utilization_mb = 0
    cpu_values = []
    memory_values = []
    
    for service_name in metric_dfs.keys():
        if service_name in metric_dfs:
            service_data = metric_dfs[service_name]
            if hasattr(service_data, 'columns'):
                for col in service_data.columns:
                    col_lower = col.lower()
                    if 'cpu' in col_lower:
                        cpu_val = service_data[col].mean() if not service_data[col].empty else 0
                        if cpu_val > 0:
                            cpu_values.append(cpu_val)
                    if 'memory' in col_lower:
                        mem_val = service_data[col].mean() if not service_data[col].empty else 0
                        if mem_val > 0:
                            memory_values.append(mem_val)
    
    if cpu_values:
        overall_cpu_utilization_cores = sum(cpu_values)
    if memory_values:
        overall_memory_utilization_mb = sum(memory_values) / (1024 * 1024)
    
    # Calculate anomaly rate
    anomaly_rate = (anomaly_results['anomaly_count'] / total_requests * 100) if total_requests > 0 else 0
    
    # Restructure snapshot data according to the specified format
    snapshot = {
        'timestamp': timestamp,
        'total_requests': total_requests,
        'mean_response_time': duration,
        'p90_response_time': p90_response_time,
        'deadline_miss_rate': overall_miss_rate,
        'patterns': {},
        'sum_cpu_utilization_cores': overall_cpu_utilization_cores,
        'sum_memory_utilization_mb': overall_memory_utilization_mb,
        'anomaly_count': anomaly_results['anomaly_count'],
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
            
            # Get CPU (cores) and memory (MB) utilization for this service
            service_cpu_utilization_cores = 0
            service_memory_utilization_mb = 0
            if hasattr(service_data, 'columns'):
                for col in service_data.columns:
                    col_lower = col.lower()
                    if 'cpu' in col_lower and service_cpu_utilization_cores == 0:
                        service_cpu_utilization_cores = service_data[col].mean() if not service_data[col].empty else 0
                    if 'memory' in col_lower and service_memory_utilization_mb == 0:
                        mem_val = service_data[col].mean() if not service_data[col].empty else 0
                        service_memory_utilization_mb = mem_val / (1024 * 1024)
            
            # Get anomaly information for this service from SHAP results
            service_anomalies = anomaly_results['anomalies_by_service'].get(service_name, [])
            service_anomaly_count = len(service_anomalies)
            service_anomaly_rate = (service_anomaly_count / total_requests * 100) if total_requests > 0 else 0
            
            snapshot['services'][service_name] = {
                'anomaly_count': service_anomaly_count,
                'anomaly_rate': service_anomaly_rate,
                'cpu_utilization_cores': service_cpu_utilization_cores,
                'memory_utilization_mb': service_memory_utilization_mb
            }
    
    return snapshot, anomaly_results

def calculate_pattern_miss_rates(trace_df):
    if trace_df is None or trace_df.empty or 'pattern' not in trace_df.columns or 'total' not in trace_df.columns:
        logger.error("Error in calculate_pattern_miss_rates: trace_df is None or empty or 'pattern' or 'total' not in trace_df.columns")
        return {}
    
    pattern_miss_rates = {}
    
    # Group by pattern and calculate miss rates
    for pattern in trace_df['pattern'].unique():
        pattern_traces = trace_df[trace_df['pattern'] == pattern]
        response_times = pd.to_numeric(pattern_traces['total'], errors='coerce')
            
        # Calculate threshold as mean + 1.5 * std for this pattern
        threshold = get_pattern_miss_rate_threshold(response_times)
        
        # Count requests above the threshold
        deadline_misses = len(response_times[response_times > threshold])
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

    trace_df.reset_index()
    if trace_df is None or trace_df.empty:
        return {
            'anomaly_count': 0,
            'anomalies_by_service': {},
            'shap_contributions': {}
        }
    
    try:
        import heapq
        from operator import itemgetter
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
        
        features = trace_df[feature_columns].copy()
        
        # Fit Isolation Forest
        iso_forest, training_features = get_trace_IsolationForest()
        
        # Ensure all training features are present and in the same order
        for col in training_features:
            if col not in features:
                features[col] = 0
        
        # Reorder features to match training order
        features = features[training_features]
        
        anomaly_predictions = iso_forest.predict(features)
        trace_df.reset_index()
        # Find anomalies
        anomaly_indices = trace_df[anomaly_predictions == -1].index.tolist()
        if not anomaly_indices:
            return {
                'anomaly_count': 0,
                'anomalies_by_service': {},
                'shap_contributions': {}
            }
        
        # Use the same SHAP calculation method as get_kpi_list
        # Use label-based indexing to avoid positional out-of-bounds errors
        # print(anomaly_indices)
        # print(features)
        anom_features = features.loc[anomaly_indices]
        shapes, names = shap_decisions(iso_forest, anom_features)
        
        # Process anomalies using the same attribution logic as get_kpi_list
        service_anomaly_count = {}
        anomalies_by_service = {}
        shap_contributions = {}
        
        for s, ai in zip(shapes, anomaly_indices):
            # Get duration and skip anomalies with duration < 2 seconds (same as get_kpi_list)
            duration = pd.to_timedelta(trace_df.loc[ai, "total"], unit="us")
            timestamp = trace_df.loc[ai, "startTime"]
            
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
                'timestamp': timestamp,
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
        # print(e)
        print(traceback.format_exc())
        logger.error(f"Error in SHAP anomaly detection: {e}")
        return {
            'anomaly_count': 0,
            'anomalies_by_service': {},
            'shap_contributions': {}
        }


iso_forest = None
iso_forest_features = None
def get_trace_IsolationForest() -> IsolationForest:
    global iso_forest, iso_forest_features
    if iso_forest is not None:
        return iso_forest, iso_forest_features
    
    # Prefer preprocessed CSV training traces if available, fall back to raw JSON.
    csv_path = os.path.join(PATH, "anomaly_detection", "training-set.csv")
    json_path = os.path.join(PATH, "anomaly_detection", "training_traces-2026-02-11.json")

    if os.path.exists(csv_path):
        logger.info(f"Loading training traces for IsolationForest from CSV: {csv_path}")
        trace_df = pd.read_csv(csv_path)
    else:
        logger.info(f"Training traces CSV not found, falling back to JSON: {json_path}")
        trace_df = read_traces(json_path)
    feature_columns = trace_df.select_dtypes(include=["number"]).columns.tolist()
    
    if 'total' in feature_columns:
        feature_columns.remove('total')
    
    iso_forest_features = feature_columns
    features = trace_df[feature_columns]
    _iso_forest = IsolationForest(contamination="auto", random_state=42)
    _iso_forest.fit(features)

    iso_forest = _iso_forest

    return iso_forest, iso_forest_features


def get_pattern_miss_rate_threshold(response_times):
    if response_times is None or response_times.empty:
        return 0
    mean_value = response_times.mean()
    if pd.isna(mean_value):
        return 0
    std_value = response_times.std()
    if pd.isna(std_value):
        return 0
    return mean_value + (1.5 * std_value)
