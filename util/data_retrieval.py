import os
import io
import time
import json
import re
import requests
import pandas as pd
import numpy as np
import sys
import argparse
import logging
from datetime import datetime, timedelta, timezone
from util.analysis import metric_snapshot

# Configure logging
logger = logging.getLogger(__name__)



# Constants for monitoring endpoints
JAEGER_ENDPOINT_FSTRING = "http://145.100.135.11:30550/api/traces?limit={limit}&lookback={lookback}&service={service}&start={start}"
PROMETHEUS_BASE_URL = "http://145.100.135.11:31207"


def convert_trace_data_to_dataframe(trace_data):
    rows = []
    for trace in trace_data["data"]:
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


class DataCollector:
    """
    Data collector for monitoring data from Prometheus and Jaeger.
    Tracks collection timing and appends data to CSV files.
    """

    instance = None
    
    def __init__(self, prometheus_base_url=PROMETHEUS_BASE_URL, jaeger_endpoint_fstring=JAEGER_ENDPOINT_FSTRING):
        self.prometheus_base_url = prometheus_base_url
        self.jaeger_endpoint_fstring = jaeger_endpoint_fstring
        self.csv_files_created = set()  # Track which CSV files have been created

    @staticmethod
    def get_instance():
        if DataCollector.instance is None:
            DataCollector.instance = DataCollector()
        return DataCollector.instance
    
    def collect_node_metrics(self, start_time, end_time, label):
        url = self.prometheus_base_url + '/api/v1/query_range'
        queries = ['100 * sum by (node) (rate(container_cpu_usage_seconds_total{container!="", pod!=""}[5m])) / sum by (node) (kube_node_status_allocatable{resource="cpu"})',
                   '100 * sum by (node) (container_memory_working_set_bytes{container!="", pod!=""}) / sum by (node) (kube_node_status_allocatable{resource="memory"})']
        
        metric_names = ['cpu_utilization', 'memory_utilization']

        node_metrics_df = None
        for query, metric_name in zip(queries, metric_names):
            params = {
                'query': query,
                'start': start_time.timestamp(),
                'end': end_time.timestamp(), 
                'step': '5s'
            }
            response = requests.get(url, params=params)
            result = response.json()

            if 'data' not in result or 'result' not in result['data'] or not result['data']['result']:
                logger.warning(f"No data returned for node metrics query: '{metric_name}' "
                             f"(query: {query[:100]}{'...' if len(query) > 100 else ''}) "
                             f"for time range {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                             f"- this may indicate the metric is not available or no data exists for this time period")
                continue
                    
            # Process the result data
            result_data = {"index": [], f"{metric_name}": []}
            
            for value_pair in result['data']['result'][0]["values"]:
                result_data['index'].append(int(value_pair[0]))
                result_data[f"{metric_name}"].append(float(value_pair[1]))
            
            result_df = pd.DataFrame(result_data)

            if node_metrics_df is None:
                node_metrics_df = result_df
            else:
                node_metrics_df = pd.merge(node_metrics_df, result_df, on='index', how='outer')

        return node_metrics_df


    def collect_prometheus_metrics(self, start_time, end_time, label, save=True):
        """
        Collect autoscaler metrics from Prometheus for all configurations.
        
        Args:
            start_time: datetime object for start of collection period
            end_time: datetime object for end of collection period  
            label: string label for output files
            
        Returns:
            tuple: (total_metrics_df, individual_metric_dfs)
        """
        
        # Get all configuration names
        url = self.prometheus_base_url + '/api/v1/label/configuration_name/values'
        response = requests.get(url)
        data = response.json()
        # logger.debug(f"Configuration names: {data}")
        
        total_m_df = None
        individual_dfs = {}
        
        logger.info(f"Collecting metrics for {len(data['data'])} configurations")

        for config_name in data['data']:     
            # Get revision names for this configuration
            url = self.prometheus_base_url + '/api/v1/label/revision_name/values'
            params = {'match[]': f'autoscaler_desired_pods{{namespace_name="default",configuration_name="{config_name}"}}'}
            response = requests.get(url, params=params)
            revision = response.json()
           
            if not revision['data']:
                logger.warning(f"No revisions found for {config_name}, skipping...")
                continue
                
            # Collect metrics for this configuration
            config_df = self._collect_config_metrics(config_name, revision['data'][-1], start_time, end_time)
            
            if config_df is not None and not config_df.empty:
                individual_dfs[config_name] = config_df
                csv_filename = f"{label}_{config_name}_metrics.csv"
                if save:
                    self._append_to_csv(config_df, csv_filename)
                
                if total_m_df is None:
                    total_m_df = config_df
                else:
                    total_m_df = pd.merge(total_m_df, config_df, on='index', how='outer')
        
        if total_m_df is not None:
            total_csv_filename = f"{label}_total_metrics.csv"
            if save:
                self._append_to_csv(total_m_df, total_csv_filename)
            
        return total_m_df, individual_dfs
    
    
    def _collect_config_metrics(self, config_name: str, revision_name: str, start_time: datetime, end_time: datetime):
        """
        Collect metrics for a specific configuration and revision.
        """
        url = self.prometheus_base_url + '/api/v1/query_range'
        
        # Define autoscaler metrics to collect
        queries = [
            'sum(autoscaler_requested_pods{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
            'sum(autoscaler_terminating_pods{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
            'sum(autoscaler_actual_pods{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
            'sum(activator_request_concurrency{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
            'sum(autoscaler_desired_pods{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
            'sum(autoscaler_stable_request_concurrency{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
            'sum(autoscaler_panic_request_concurrency{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
            'sum(autoscaler_target_concurrency_per_pod{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
            'sum(autoscaler_excess_burst_capacity{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
            'sum(rate(container_cpu_usage_seconds_total{{namespace="default", pod=~"{revision}.*", container != "POD", container != ""}}[1m])) by (container)'
        ]
        
        metric_df = None
        
        for query in queries:
            # Extract metric name from query
            match = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\{\{', query)
            metric_name = match.group(1) if match else query
            
            # Format query with config and revision
            formatted_query = query.format(config=config_name, revision=revision_name)
            params = {
                'query': formatted_query,
                'start': start_time.timestamp(),
                'end': end_time.timestamp(), 
                'step': '5s'
            }
            
            try:
                response = requests.get(url, params=params)
                result = response.json()

                if 'data' not in result or 'result' not in result['data'] or not result['data']['result']:
                    continue
                    logger.warning(f"No data returned for service metrics query: '{metric_name}' "
                                 f"for service '{config_name}' (revision: {revision_name}) "
                                 f"(query: {formatted_query[:100]}{'...' if len(formatted_query) > 100 else ''}) "
                                 f"for time range {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                                 f"- this may indicate the service is not running, metric is not available, or no data exists for this time period")
                    continue
                
                # Process the result data
                result_data = {"index": [], f"{config_name}_{metric_name}": []}
                
                for value_pair in result['data']['result'][0]["values"]:
                    result_data['index'].append(int(value_pair[0]))
                    result_data[f"{config_name}_{metric_name}"].append(float(value_pair[1]))
                
                result_df = pd.DataFrame(result_data)
                
                if metric_df is None:
                    metric_df = result_df
                else:
                    metric_df = pd.merge(metric_df, result_df, on='index', how='outer')
                    
            except Exception as e:
                logger.error(f"Error collecting metric {metric_name}: {e}")
                continue
        
        return metric_df
    
    def _append_to_csv(self, dataframe, filename):
        """
        Append data to CSV file, creating header only on first write.
        
        Args:
            dataframe: pandas DataFrame to append
            filename: CSV filename
        """
        if filename not in self.csv_files_created:
            # First time writing to this file, include header
            dataframe.to_csv(filename, mode='w', header=True, index=False)
            self.csv_files_created.add(filename)
            # logger.debug(f"Created new CSV file: {filename}")
        else:
            # Append to existing file without header
            dataframe.to_csv(filename, mode='a', header=False, index=False)
            # logger.debug(f"Appended data to CSV file: {filename}")
    
    def collect_jaeger_traces(self, start_time, end_time, label, service_name="frontend", save=True):
        """
        Collect traces from Jaeger for the frontend service using explicit start/end.
        
        Args:
            start_time: datetime for start of the window (UTC)
            end_time: datetime for end of the window (UTC)
            label: string label for output file
            service_name: Jaeger service name to query
            
        Returns:
            dict: parsed trace data
        """
        logger.info(f"Collecting Jaeger traces from {start_time} to {end_time} for service '{service_name}'")
        
        # Convert datetimes to microseconds since epoch (Jaeger expects microseconds)
        start_us = int(start_time.timestamp() * 1_000_000)
        end_us = int(end_time.timestamp() * 1_000_000)
        
        # Pad the window slightly to avoid boundary misses
        pad_us = 5 * 1_000_000
        start_us_padded = max(0, start_us - pad_us)
        end_us_padded = end_us + pad_us
        
        base_url = "http://145.100.135.11:30550/api"
        traces_url = base_url + "/traces"
        services_url = base_url + "/services"
        
        def save_and_return(traces, save=True):
            try:
                if save:
                    with io.open(f'./{label}_traces.json', 'w', encoding='utf8') as outfile:
                        json.dump(traces, outfile, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
                    # logger.info(f"Saved {len(traces.get('data', []))} traces to {label}_traces.json")
            except Exception as e:
                logger.error(f"Error writing traces file: {e}")
            return traces
        
        try:
            # Attempt 1: start/end with padded window for the requested service
            params = {
                'service': service_name,
                'limit': '5000',
                'start': start_us_padded,
                'end': end_us_padded,
            }
            resp = requests.get(traces_url, params=params)
            data = resp.json()
            if data.get('data'):
                return save_and_return(data, save)
            
            # Attempt 2: retry using lookback semantics if no data
            lookback_seconds = max(1, int((end_us - start_us) / 1_000_000))
            lookback_url = JAEGER_ENDPOINT_FSTRING.format(limit=str(5000), lookback=str(lookback_seconds), service=service_name, start=start_us)
            resp2 = requests.get(lookback_url)
            data2 = resp2.json()
            if data2.get('data'):
                return save_and_return(data2, save)
            
            # Nothing found
            logger.warning("No traces returned by Jaeger for any service in the given window.")
            return None
            
        except Exception as e:
            logger.error(f"Error collecting Jaeger traces: {e}")
            return save_and_return({'data': [], 'errors': None, 'total': 0}, save)

    def log_all_data_to_snapshot(self, start_time, end_time, phase=None, subphase=None, save=False):
        _, individual_metrics_dfs = self.collect_prometheus_metrics(start_time, end_time, "label", save=save)

        traces_data = self.collect_jaeger_traces(start_time, end_time, "label", save=save)

        if not traces_data.get('data'):
            print("No traces found")

        trace_df = convert_trace_data_to_dataframe(traces_data)
        snapshot, _ = metric_snapshot(trace_df, individual_metrics_dfs, phase, subphase)
        
        # Add collection metadata to snapshot
        snapshot['collection_start_time'] = start_time.isoformat()
        snapshot['collection_end_time'] = end_time.isoformat()

        return snapshot


    def log_all_data_to_csv(self, start_time, end_time, label, snapshot_json_path, phase=None, subphase=None):
        # Ensure output directory exists for the snapshot file
        snapshot_dir = os.path.dirname(snapshot_json_path) or "."
        try:
            os.makedirs(snapshot_dir, exist_ok=True)
        except Exception:
            pass

        # node_metrics_df = self.collect_node_metrics(start_time, end_time, label)

        total_metrics_df, individual_metrics_dfs = self.collect_prometheus_metrics(start_time, end_time, label)

        traces_data = self.collect_jaeger_traces(start_time, end_time, label)
        trace_df = convert_trace_data_to_dataframe(traces_data)
        traces_csv = f"{label}_traces.csv"
        if not trace_df.empty:
            self._append_to_csv(trace_df, traces_csv)

        snapshot, anomaly_results = metric_snapshot(trace_df, individual_metrics_dfs, phase, subphase)
        
        # Add collection metadata to snapshot
        snapshot['collection_label'] = label
        snapshot['collection_start_time'] = start_time.isoformat()
        snapshot['collection_end_time'] = end_time.isoformat()
        snapshot['traces_csv'] = traces_csv if not trace_df.empty else ""
        snapshot['total_metrics_csv'] = f"{label}_total_metrics.csv" if total_metrics_df is not None else ""
        snapshot['configurations'] = list(individual_metrics_dfs.keys())
        snapshot['trace_count'] = len(trace_df) if not trace_df.empty else 0
        snapshot['metrics_config_count'] = len(individual_metrics_dfs)

        # Append snapshot to JSON file
        self._append_snapshot_to_json(snapshot, snapshot_json_path)
        return snapshot, trace_df, individual_metrics_dfs, anomaly_results

    def _append_snapshot_to_json(self, snapshot, json_path):
        try:
            # Read existing snapshots if file exists
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    snapshots = json.load(f)
            else:
                snapshots = []
            
            # Add new snapshot
            snapshots.append(snapshot)
            
            # Write back to file
            with open(json_path, 'w') as f:
                json.dump(snapshots, f, indent=2, default=str)
            
            # logger.debug(f"Appended snapshot to JSON: {json_path} (total snapshots: {len(snapshots)})")
            
        except Exception as e:
            logger.error(f"Error appending snapshot to JSON: {e}")

