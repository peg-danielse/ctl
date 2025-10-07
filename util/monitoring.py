import os
import io
import glob
import math
import time
import json
import re
import datetime
import requests
import pandas as pd

# Constants for monitoring endpoints
JAEGER_ENDPOINT_FSTRING = "http://145.100.135.11:30550/api/traces?limit={limit}&lookback={lookback}&service={service}&start={start}"
PROMETHEUS_BASE_URL = "http://145.100.135.11:31207"

class MonitoringDataCollector:
    """
    Collects monitoring data from Prometheus and Jaeger during load tests.
    """
    
    def __init__(self, prometheus_base_url=PROMETHEUS_BASE_URL, jaeger_endpoint_fstring=JAEGER_ENDPOINT_FSTRING):
        self.prometheus_base_url = prometheus_base_url
        self.jaeger_endpoint_fstring = jaeger_endpoint_fstring
    
    def collect_prometheus_metrics(self, start_time, end_time, label):
        """
        Collect autoscaler metrics from Prometheus for all configurations.
        
        Args:
            start_time: datetime object for start of collection period
            end_time: datetime object for end of collection period  
            label: string label for output files
            
        Returns:
            tuple: (total_metrics_df, individual_metric_dfs)
        """
        print(f"Collecting Prometheus metrics from {start_time} to {end_time}")
        
        # Get all configuration names
        url = self.prometheus_base_url + '/api/v1/label/configuration_name/values'
        response = requests.get(url)
        data = response.json()
        print("Configuration names:", data)
        
        total_m_df = None
        individual_dfs = {}
        
        for config_name in data['data']:
            print(f"Processing configuration: {config_name}")
            
            # Get revision names for this configuration
            url = self.prometheus_base_url + '/api/v1/label/revision_name/values'
            params = {'match[]': f'autoscaler_desired_pods{{namespace_name="default",configuration_name="{config_name}"}}'}
            response = requests.get(url, params=params)
            revision = response.json()
            print("Revision name:", revision['data'])
            
            if not revision['data']:
                print(f"No revisions found for {config_name}, skipping...")
                continue
                
            # Collect metrics for this configuration
            config_df = self._collect_config_metrics(config_name, revision['data'][-1], start_time, end_time)
            
            if config_df is not None and not config_df.empty:
                individual_dfs[config_name] = config_df
                config_df.to_csv(f"{label}_{config_name}_metrics.csv")
                
                if total_m_df is None:
                    total_m_df = config_df
                else:
                    total_m_df = pd.merge(total_m_df, config_df, on='index', how='outer')
        
        if total_m_df is not None:
            total_m_df.to_csv(f"{label}_total_metrics.csv")
            
        return total_m_df, individual_dfs
    
    def _collect_config_metrics(self, config_name, revision_name, start_time, end_time):
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
                'start': start_time.isoformat() + 'Z',
                'end': end_time.isoformat() + 'Z',
                'step': '5s'
            }
            
            try:
                response = requests.get(url, params=params)
                result = response.json()
                
                if 'data' not in result or 'result' not in result['data'] or not result['data']['result']:
                    print(f"No data returned for query: {metric_name}")
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
                print(f"Error collecting metric {metric_name}: {e}")
                continue
        
        return metric_df
    
    def collect_jaeger_traces(self, run_time_seconds, label):
        """
        Collect traces from Jaeger for the frontend service.
        
        Args:
            run_time_seconds: duration of the test in seconds
            label: string label for output file
            
        Returns:
            dict: parsed trace data
        """
        print(f"Collecting Jaeger traces for {run_time_seconds} seconds")
        
        # Calculate start time (current time - run_time)
        start_time_microseconds = int((time.mktime(time.localtime()) - run_time_seconds) * 1_000_000)
        
        url = self.jaeger_endpoint_fstring.format(
            limit=str(4000),
            lookback=str(run_time_seconds),
            service="frontend",
            start=start_time_microseconds
        )
        
        print(f"Jaeger URL: {url}")
        
        try:
            response = requests.get(url)
            data = response.json()
            
            # Save traces to file
            with io.open(f'./{label}_traces.json', 'w', encoding='utf8') as outfile:
                json.dump(data, outfile, indent=4, sort_keys=True, 
                         separators=(',', ': '), ensure_ascii=False)
            
            print(f"Saved {len(data.get('data', []))} traces to {label}_traces.json")
            return data
            
        except Exception as e:
            print(f"Error collecting Jaeger traces: {e}")
            return {}
    
    def collect_all_monitoring_data(self, start_time, end_time, run_time_seconds, label):
        """
        Collect all monitoring data (Prometheus metrics and Jaeger traces).
        
        Args:
            start_time: datetime object for start of collection period
            end_time: datetime object for end of collection period
            run_time_seconds: duration of the test in seconds
            label: string label for output files
            
        Returns:
            dict: containing all collected data
        """
        print(f"Starting comprehensive monitoring data collection for label: {label}")
        print(f"Time range: {start_time} to {end_time}")
        
        # Collect Prometheus metrics
        total_metrics_df, individual_metrics_dfs = self.collect_prometheus_metrics(start_time, end_time, label)
        
        # Collect Jaeger traces
        traces_data = self.collect_jaeger_traces(run_time_seconds, label)
        
        return {
            'total_metrics': total_metrics_df,
            'individual_metrics': individual_metrics_dfs,
            'traces': traces_data,
            'start_time': start_time,
            'end_time': end_time,
            'label': label
        }


def collect_monitoring_data_for_test(start_time, end_time, run_time_seconds, label):
    """
    Convenience function to collect monitoring data for a completed load test.
    
    Args:
        start_time: datetime object for start of test
        end_time: datetime object for end of test  
        run_time_seconds: duration of the test in seconds
        label: string label for output files
        
    Returns:
        dict: containing all collected monitoring data
    """
    collector = MonitoringDataCollector()
    return collector.collect_all_monitoring_data(start_time, end_time, run_time_seconds, label)


class ContinuousMonitoringCollector:
    """
    Provides continuous monitoring data collection for the ctl.py main loop.
    This class can be used to collect monitoring data at regular intervals
    and provide it to the ctl.py optimization loop.
    """
    
    def __init__(self, prometheus_base_url=PROMETHEUS_BASE_URL, jaeger_endpoint_fstring=JAEGER_ENDPOINT_FSTRING):
        self.collector = MonitoringDataCollector(prometheus_base_url, jaeger_endpoint_fstring)
        self.last_collection_time = None
        self.collection_interval = 60  # seconds
    
    def collect_current_metrics(self, label="current", interval_seconds=10):
        """
        Collect current metrics from Prometheus for a specific time interval.
        This is useful for real-time monitoring during the optimization loop.
        
        Args:
            label: string label for the collection
            interval_seconds: time interval in seconds since last check (if None, uses 10 seconds)
            
        Returns:
            dict: current metrics data
        """
        from datetime import datetime, timedelta

        # Collect metrics for the specified interval
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=interval_seconds)
        
        print(f"Collecting current metrics for {label} (interval: {interval_seconds}s)")
        
        # Get all configuration names
        url = self.collector.prometheus_base_url + '/api/v1/label/configuration_name/values'
        response = requests.get(url)
        data = response.json()
        
        current_metrics = {}
        
        for config_name in data['data']:
            # Get revision names for this configuration
            url = self.collector.prometheus_base_url + '/api/v1/label/revision_name/values'
            params = {'match[]': f'autoscaler_desired_pods{{namespace_name="default",configuration_name="{config_name}"}}'}
            response = requests.get(url, params=params)
            revision = response.json()
            
            if not revision['data']:
                continue
                
            # Get current values for key metrics
            current_values = self._get_current_metric_values(config_name, revision['data'][-1])
            if current_values:
                current_metrics[config_name] = current_values
        
        # Collect fresh trace data for the current monitoring interval
        print(f"Collecting fresh trace data for {label} (interval: {interval_seconds}s)")
        trace_data = self.collector.collect_jaeger_traces(interval_seconds, f"{label}_traces")
        
        # If no fresh trace data collected, try to read existing data as fallback
        if not trace_data or not trace_data.get('data'):
            print(f"No fresh trace data collected, reading existing trace data for {label}")
            trace_data = self._read_existing_trace_data(label)
        
        return {
            'timestamp': end_time,
            'metrics': current_metrics,
            'traces': trace_data,
            'label': label
        }
    
    def _read_existing_trace_data(self, label):
        """
        Read existing trace data from files in the current directory.
        
        Args:
            label: Label to look for in trace files
            
        Returns:
            dict: Trace data or empty dict if not found
        """
        import glob
        import json
        
        # Look for trace files with the label
        trace_patterns = [
            f"{label}_traces.json",
            f"{label}_traces_traces.json", 
            f"{label}_continuous_traces_traces.json",
            f"{label}_baseline_traces_traces.json"
        ]
        
        for pattern in trace_patterns:
            if os.path.exists(pattern):
                try:
                    print(f"Reading existing trace data from {pattern}")
                    with open(pattern, 'r') as f:
                        trace_data = json.load(f)
                    if trace_data and trace_data.get('data'):
                        print(f"Found {len(trace_data['data'])} traces in {pattern}")
                        return trace_data
                except Exception as e:
                    print(f"Error reading trace file {pattern}: {e}")
                    continue
        
        print(f"No existing trace data found for label {label}")
        return {}
    
    def _get_current_metric_values(self, config_name, revision_name):
        """
        Get current values for key metrics for a specific configuration.
        """
        url = self.collector.prometheus_base_url + '/api/v1/query'
        
        # Key metrics to monitor in real-time
        key_metrics = [
            'autoscaler_actual_pods',
            'autoscaler_desired_pods', 
            'autoscaler_requested_pods',
            'container_cpu_usage_seconds_total',
            'activator_request_concurrency'
        ]
        
        current_values = {}
        
        for metric in key_metrics:
            if metric == 'container_cpu_usage_seconds_total':
                query = f'sum(rate({metric}{{namespace="default", pod=~"{revision_name}.*", container != "POD", container != ""}}[1m]))'
            else:
                query = f'sum({metric}{{namespace_name="default", configuration_name="{config_name}", revision_name="{revision_name}"}})'
            
            try:
                params = {'query': query}
                response = requests.get(url, params=params)
                result = response.json()
                
                if result['status'] == 'success' and result['data']['result']:
                    value = float(result['data']['result'][0]['value'][1])
                    current_values[metric] = value
                    
            except Exception as e:
                print(f"Error getting current value for {metric}: {e}")
                continue
        
        return current_values
    
    def should_collect_data(self):
        """
        Check if it's time to collect monitoring data based on the collection interval.
        
        Returns:
            bool: True if data should be collected
        """
        from datetime import datetime
        
        if self.last_collection_time is None:
            return True
            
        time_since_last = (datetime.utcnow() - self.last_collection_time).total_seconds()
        return time_since_last >= self.collection_interval
    
    def collect_and_update(self, label="continuous", interval_seconds=None):
        """
        Collect monitoring data if enough time has passed since last collection.
        
        Args:
            label: string label for the collection
            interval_seconds: time interval in seconds since last check (if None, uses 10 seconds)
            
        Returns:
            dict: monitoring data if collected, None if skipped
        """
        if not self.should_collect_data():
            return None
            
        self.last_collection_time = datetime.utcnow()
        return self.collect_current_metrics(label, interval_seconds)


def create_continuous_monitor():
    """
    Factory function to create a continuous monitoring collector.
    
    Returns:
        ContinuousMonitoringCollector: configured for continuous monitoring
    """
    return ContinuousMonitoringCollector()


class ContinuousAnomalyDetector:
    """
    Performs continuous anomaly detection on collected monitoring data.
    This class maintains the isolation forest model and continuously analyzes
    new data for anomalies.
    """
    
    def __init__(self):
        self.isolation_forest = None
        self.training_data = None
        self.training_feature_names = None
        self.anomaly_threshold = 0.2  # 20% contamination - more sensitive
        
    def train_model(self, trace_data):
        """
        Train the isolation forest model on baseline trace data.
        
        Args:
            trace_data: pandas DataFrame with trace data for training
        """
        import pandas as pd
        from sklearn.ensemble import IsolationForest
        
        print("Training isolation forest model on baseline data...")
        
        # Prepare features for training - use service operation columns from trace data
        # Exclude non-service columns that shouldn't be used for anomaly attribution
        exclude_columns = ["total", "duration", "startTime", "id", "error", "pattern", "service", "operationName"]
        
        # Get all numeric columns that represent service operations
        all_numeric = trace_data.select_dtypes(include=["number"])
        features = all_numeric.drop(columns=exclude_columns, errors='ignore')
        
        # Only keep columns that look like service operations (HTTP, gRPC, memcached, etc.)
        service_columns = []
        for col in features.columns:
            if any(pattern in col.lower() for pattern in ['http', 'grpc', '/', 'memcached', 'mongo']):
                service_columns.append(col)
        
        if service_columns:
            features = features[service_columns]
            print(f"Using service operation features for training: {service_columns}")
        else:
            print("Warning: No service operation features found for training")
            print(f"Available numeric columns: {list(all_numeric.columns)}")
            print(f"Available all columns: {list(trace_data.columns)}")
            return False
            
        # Store feature names for consistent prediction
        self.training_feature_names = list(features.columns)
        
        # Train the isolation forest
        self.isolation_forest = IsolationForest(
            contamination=self.anomaly_threshold, 
            random_state=42
        )
        self.isolation_forest.fit(features)
        self.training_data = features
        
        print(f"Model trained on {len(features)} samples with {len(features.columns)} features")
        print(f"Feature names: {self.training_feature_names}")
        return True
    
    def detect_anomalies(self, trace_data, exact_interval_seconds=None):
        """
        Detect anomalies in the given trace data for the specified time interval.
        
        Args:
            trace_data: pandas DataFrame with trace data
            exact_interval_seconds: exact time interval in seconds to filter data for
            
        Returns:
            list: list of anomaly tuples (service_name, anomaly_index, duration)
        """
        if self.isolation_forest is None:
            print("No trained model available, using statistical anomaly detection")
            return self._detect_anomalies_statistical(trace_data, exact_interval_seconds)
            
        import pandas as pd
        import heapq
        from operator import itemgetter
        from datetime import datetime, timedelta
        
        # Filter data for the exact interval
        if 'startTime' in trace_data.columns:
            # Convert startTime to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(trace_data['startTime']):
                trace_data['startTime'] = pd.to_datetime(trace_data['startTime'])
            
            # Check if we're working with historical data (all timestamps are in the past)
            latest_timestamp = trace_data['startTime'].max()
            current_time = datetime.utcnow()
            
            if latest_timestamp < current_time - timedelta(minutes=1):
                # Historical data - use all data for anomaly detection
                print(f"Using historical data (latest: {latest_timestamp}), analyzing all {len(trace_data)} samples")
                recent_data = trace_data
            else:
                # Real-time data - filter by interval
                if exact_interval_seconds is not None:
                    window_start = current_time - timedelta(seconds=exact_interval_seconds)
                    print(f"Filtering real-time data for exact interval: {exact_interval_seconds} seconds")
                else:
                    # Default to last 10 seconds if no interval provided
                    window_start = current_time - timedelta(seconds=10)
                    print("No interval provided, using default 10-second window")
                    
                recent_data = trace_data[trace_data['startTime'] >= window_start]
        else:
            recent_data = trace_data.tail(100)  # Use last 100 records if no timestamp
        
        if recent_data.empty:
            return []
        
        # Prepare features for anomaly detection with consistent feature names
        if self.training_feature_names is None:
            print("Warning: No training feature names available, using statistical detection")
            return self._detect_anomalies_statistical(trace_data, exact_interval_seconds)
        
        # Create features with the same columns as training data
        features = pd.DataFrame(index=recent_data.index)
        
        # Add features in the same order as training
        for feature_name in self.training_feature_names:
            if feature_name in recent_data.columns:
                features[feature_name] = recent_data[feature_name]
            else:
                # Fill missing features with 0
                features[feature_name] = 0
        
        # Fill any NaN values with 0
        features = features.fillna(0)
        
        if features.empty:
            return []
        
        # Predict anomalies
        anomaly_predictions = self.isolation_forest.predict(features)
        anomaly_indices = recent_data[anomaly_predictions == -1].index.tolist()
        
        # Debug: print anomaly detection results
        num_anomalies = (anomaly_predictions == -1).sum()
        print(f"    Isolation Forest detected {num_anomalies} anomalies out of {len(features)} samples")
        
        if not anomaly_indices:
            print("    No anomalies detected by isolation forest")
            return []
        
        # Analyze anomalies and return significant ones
        anomalies = []
        for idx in anomaly_indices:
            if 'total' in recent_data.columns:
                duration = pd.to_timedelta(recent_data.loc[idx, 'total'], unit="us")
                
                # Only consider anomalies lasting more than 2 seconds
                if duration < pd.Timedelta(seconds=2):
                    continue
                    
                # Find the service with the most significant contribution to the anomaly using SHAP values
                service_contributions = self._calculate_shap_contributions(recent_data, idx, features)
                
                if service_contributions:
                    # Get the service with the highest SHAP contribution
                    top_service = max(service_contributions.items(), key=itemgetter(1))
                    service_name = top_service[0]
                    
                    # Map service name to actual service
                    from config import SPAN_PROCESS_MAP
                    actual_service = SPAN_PROCESS_MAP.get(service_name, service_name)
                    
                    # Only print attribution for first few anomalies to avoid spam
                    if idx in recent_data.index[:5]:
                        print(f"    Anomaly {idx}: Attributed to {actual_service} (from {service_name}, contribution: {top_service[1]:.1f})")
                    anomalies.append((actual_service, idx, duration))
                else:
                    if idx in recent_data.index[:5]:
                        print(f"    Anomaly {idx}: No service contributions found")
        
        return sorted(anomalies, key=lambda x: -x[2])  # Sort by duration descending
    
    def _calculate_shap_contributions(self, recent_data, anomaly_idx, features):
        """
        Calculate SHAP contributions for anomaly attribution using the proven method from analysis.py.
        
        Args:
            recent_data: DataFrame with recent trace data
            anomaly_idx: Index of the anomalous trace
            features: Feature columns used for anomaly detection
            
        Returns:
            dict: Service contributions to the anomaly
        """
        try:
            import shap
            import heapq
            from operator import itemgetter
            
            # Get the anomalous sample
            anomaly_sample = features.loc[anomaly_idx:anomaly_idx]
            
            # Calculate SHAP values for this specific anomaly
            shap_values = shap.TreeExplainer(self.isolation_forest).shap_values(anomaly_sample)
            
            # Get the SHAP values for this single sample (first row)
            if len(shap_values.shape) > 1 and shap_values.shape[0] > 0:
                sample_shap_values = shap_values[0]
            elif len(shap_values.shape) == 1 and len(shap_values) > 0:
                sample_shap_values = shap_values
            else:
                return {}
            
            # Find the most negative SHAP values (these indicate anomaly contribution)
            # Use heapq.nsmallest to get the most negative values
            try:
                negative_contributions = heapq.nsmallest(3, enumerate(sample_shap_values), key=itemgetter(1))
            except (IndexError, ValueError, TypeError):
                return {}
            
            # Create service contributions dictionary
            service_contributions = {}
            for feature_idx, shap_value in negative_contributions:
                if feature_idx < len(features.columns):
                    feature_name = features.columns[feature_idx]
                    # Use absolute value of negative SHAP values as contribution
                    service_contributions[feature_name] = abs(shap_value)
            
            # Debug: print top contributions (only for first few anomalies to avoid spam)
            if service_contributions and len(service_contributions) > 0:
                sorted_contribs = sorted(service_contributions.items(), key=lambda x: x[1], reverse=True)
                # Only print detailed SHAP info for first 3 anomalies
                if anomaly_idx in recent_data.index[:3]:
                    print(f"    SHAP contributions for anomaly {anomaly_idx}:")
                    for span, contrib in sorted_contribs[:3]:  # Top 3
                        from config import SPAN_PROCESS_MAP
                        service = SPAN_PROCESS_MAP.get(span, span)
                        print(f"      {span[:30]:<30} | {contrib:8.3f} | {service}")
            
            return service_contributions
            
        except Exception as e:
            # Silently handle SHAP errors to avoid spam
            # Fallback to simple deviation method
            service_contributions = {}
            try:
                anomaly_sample = recent_data.loc[anomaly_idx, features.columns]
                baseline = recent_data[features.columns].mean()
                
                for col in features.columns:
                    if col in recent_data.columns:
                        deviation = anomaly_sample[col] - baseline[col]
                        if deviation > 0:  # Only positive deviations
                            service_contributions[col] = deviation
            except:
                pass
            
            return service_contributions
    
    def _detect_anomalies_statistical(self, trace_data, exact_interval_seconds=None):
        """
        Detect anomalies using statistical methods when no trained model is available.
        
        Args:
            trace_data: pandas DataFrame with trace data
            exact_interval_seconds: exact time interval in seconds to filter data for
            
        Returns:
            list: list of anomaly tuples (service_name, anomaly_index, duration)
        """
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Filter data for the exact interval
        if 'startTime' in trace_data.columns:
            # Convert startTime to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(trace_data['startTime']):
                trace_data['startTime'] = pd.to_datetime(trace_data['startTime'])
            
            # Use exact_interval_seconds if provided, otherwise default to 10 seconds
            if exact_interval_seconds is not None:
                current_time = datetime.utcnow()
                window_start = current_time - timedelta(seconds=exact_interval_seconds)
                print(f"Statistical detection: filtering data for exact interval: {exact_interval_seconds} seconds")
            else:
                # Default to last 10 seconds if no interval provided
                current_time = datetime.utcnow()
                window_start = current_time - timedelta(seconds=10)
                print("Statistical detection: no interval provided, using default 10-second window")
                
            recent_data = trace_data[trace_data['startTime'] >= window_start]
        else:
            recent_data = trace_data.tail(100)  # Use last 100 records if no timestamp
        
        if recent_data.empty:
            return []
        
        anomalies = []
        
        # Statistical anomaly detection based on response times
        if 'total' in recent_data.columns:
            response_times = pd.to_numeric(recent_data['total'], errors='coerce')
            response_times = response_times.dropna()
            
            if len(response_times) > 0:
                # Calculate statistical thresholds
                mean_response_time = response_times.mean()
                std_response_time = response_times.std()
                
                # Define anomaly threshold (mean + 2*std)
                anomaly_threshold = mean_response_time + (2 * std_response_time)
                
                # Find anomalies (response times above threshold)
                # Ensure we're comparing numeric values
                numeric_response_times = pd.to_numeric(response_times, errors='coerce')
                anomaly_indices = numeric_response_times[numeric_response_times > anomaly_threshold].index.tolist()
                
                print(f"Statistical detection: mean={mean_response_time:.2f}ms, std={std_response_time:.2f}ms, threshold={anomaly_threshold:.2f}ms")
                print(f"Found {len(anomaly_indices)} statistical anomalies")
                
                # Process each anomaly
                for idx in anomaly_indices:
                    duration = pd.to_timedelta(recent_data.loc[idx, 'total'], unit="us")
                    
                    # Only consider anomalies lasting more than 1ms (convert microseconds to milliseconds)
                    duration_ms = duration.total_seconds() * 1000
                    if duration_ms < 1:
                        continue
                    
                    # Find the service with the most significant contribution using SHAP values
                    # Use the same service filtering as the main detection method
                    exclude_columns = ['total', 'duration', 'startTime', 'id', 'error', 'pattern', 'service', 'operationName']
                    
                    # Get all numeric columns that represent service operations
                    all_numeric = recent_data.select_dtypes(include=["number"])
                    features = all_numeric.drop(columns=exclude_columns, errors='ignore')
                    
                    # Only keep columns that look like service operations (HTTP, gRPC, memcached, etc.)
                    service_columns = []
                    for col in features.columns:
                        if any(pattern in col.lower() for pattern in ['http', 'grpc', '/', 'memcached', 'mongo']):
                            service_columns.append(col)
                    
                    if service_columns:
                        features = features[service_columns]
                    else:
                        # Fallback to all numeric columns if no service columns found
                        features = recent_data.select_dtypes(include=["number"]).drop(columns=exclude_columns, errors='ignore')
                    
                    # For statistical method, we need to create a temporary isolation forest
                    # to calculate SHAP values
                    try:
                        if features.empty or len(features.columns) == 0:
                            raise ValueError("No features available for SHAP calculation")
                            
                        import shap
                        from sklearn.ensemble import IsolationForest
                        temp_forest = IsolationForest(contamination="auto", random_state=42)
                        temp_forest.fit(features)
                        
                        # Temporarily set the isolation forest for SHAP calculation
                        original_forest = self.isolation_forest
                        self.isolation_forest = temp_forest
                        service_contributions = self._calculate_shap_contributions(recent_data, idx, features)
                        self.isolation_forest = original_forest
                    except Exception as e:
                        # Silently handle SHAP errors to avoid spam - only log occasionally
                        if idx % 10 == 0:  # Only log every 10th error
                            print(f"Error with SHAP in statistical method: {e}")
                        # Fallback to simple method
                        service_contributions = {}
                        for col in features.columns:
                            if col in recent_data.columns:
                                value = recent_data.loc[idx, col]
                                numeric_value = pd.to_numeric(value, errors='coerce')
                                if pd.notna(numeric_value) and numeric_value > 0:
                                    service_contributions[col] = numeric_value
                    
                    if service_contributions:
                        # Get the service with the highest SHAP contribution
                        from operator import itemgetter
                        top_service = max(service_contributions.items(), key=itemgetter(1))
                        service_name = top_service[0]
                        
                        # Map service name to actual service
                        from config import SPAN_PROCESS_MAP
                        actual_service = SPAN_PROCESS_MAP.get(service_name, service_name)
                        
                        anomalies.append((actual_service, idx, duration))
        
        return sorted(anomalies, key=lambda x: -x[2])  # Sort by duration descending


class ConfigurationPerformanceLogger:
    """
    Logs and tracks performance metrics for different configurations.
    Stores KPI data for every monitoring interval to track configuration performance.
    """
    
    def __init__(self, label):
        self.label = label
        self.performance_log = []
        self.current_configuration = None
        self.configuration_start_time = None
        self.current_phase = "initialization"  # Track current phase
        self.stabilization_start_time = None  # Track when stabilization period begins
        self.stabilization_duration = None  # Track stabilization duration
        
    def start_configuration_tracking(self, configuration_name):
        """
        Start tracking performance for a new configuration.
        
        Args:
            configuration_name: Name/identifier of the configuration being applied
        """
        self.current_configuration = configuration_name
        self.configuration_start_time = datetime.datetime.utcnow()
        print(f"Started tracking performance for configuration: {configuration_name}")
        
    def set_current_phase(self, phase_name):
        """
        Set the current phase for snapshot logging.
        
        Args:
            phase_name: Name of the current phase (e.g., "initialization", "baseline", "adaptation")
        """
        self.current_phase = phase_name
        print(f"📊 Phase changed to: {phase_name}")
        
    def start_stabilization_period(self, duration_seconds=300):
        """
        Start tracking a stabilization period after configuration changes.
        
        Args:
            duration_seconds: Expected duration of stabilization period in seconds (default: 5 minutes)
        """
        self.stabilization_start_time = datetime.datetime.utcnow()
        self.stabilization_duration = duration_seconds
        self.current_phase = "stabilization"
        print(f"🔄 Started stabilization period: {duration_seconds}s")
        
    def end_stabilization_period(self):
        """
        End the stabilization period and return to adaptation phase.
        """
        if self.stabilization_start_time:
            actual_duration = (datetime.datetime.utcnow() - self.stabilization_start_time).total_seconds()
            print(f"✅ Stabilization period completed: {actual_duration:.1f}s")
            self.stabilization_start_time = None
            self.stabilization_duration = None
            self.current_phase = "adaptation"
        else:
            print("⚠️  No active stabilization period to end")
            
    def get_stabilization_statistics(self):
        """
        Get statistics about stabilization periods from the performance log.
        
        Returns:
            dict: Statistics about stabilization periods
        """
        if not self.performance_log:
            return {"stabilization_periods": 0, "total_stabilization_time": 0}
            
        df = pd.DataFrame(self.performance_log)
        stabilization_entries = df[df['phase'] == 'stabilization']
        
        if stabilization_entries.empty:
            return {"stabilization_periods": 0, "total_stabilization_time": 0}
        
        # Group by configuration to identify separate stabilization periods
        stabilization_periods = stabilization_entries.groupby('configuration').agg({
            'stabilization_elapsed': 'max',
            'timestamp': ['min', 'max']
        }).round(2)
        
        total_stabilization_time = stabilization_periods[('stabilization_elapsed', 'max')].sum()
        num_periods = len(stabilization_periods)
        
        return {
            "stabilization_periods": num_periods,
            "total_stabilization_time": total_stabilization_time,
            "average_stabilization_time": total_stabilization_time / num_periods if num_periods > 0 else 0,
            "stabilization_details": stabilization_periods.to_dict()
        }
        
    def log_interval_metrics(self, metrics_data, interval_seconds, anomalies=None):
        """
        Log KPI metrics for the current monitoring interval.
        
        Args:
            metrics_data: Dictionary containing metrics data from monitoring
            interval_seconds: Duration of the monitoring interval
            anomalies: List of detected anomalies (if any)
        """
        if not self.current_configuration:
            print("Warning: No configuration being tracked, skipping metrics logging")
            return
            
        timestamp = datetime.datetime.utcnow()
        
        # Calculate stabilization progress if in stabilization phase
        stabilization_progress = None
        if self.current_phase == "stabilization" and self.stabilization_start_time:
            elapsed = (timestamp - self.stabilization_start_time).total_seconds()
            if self.stabilization_duration:
                stabilization_progress = min(elapsed / self.stabilization_duration, 1.0)
        
        # Extract key performance indicators
        kpi_data = {
            'timestamp': timestamp,
            'configuration': self.current_configuration,
            'phase': self.current_phase,  # Add phase information
            'interval_seconds': interval_seconds,
            'anomalies_detected': len(anomalies) if anomalies else 0,
            'anomaly_details': anomalies if anomalies else [],
            'metrics_count': len(metrics_data.get('metrics', {})) if metrics_data else 0,
            'has_trace_data': bool(metrics_data.get('traces')) if metrics_data else False,
            'stabilization_progress': stabilization_progress,  # Add stabilization progress
            'stabilization_elapsed': (timestamp - self.stabilization_start_time).total_seconds() if self.stabilization_start_time else None
        }
        
        # Add service-specific metrics if available
        if metrics_data and 'metrics' in metrics_data:
            for service_name, service_metrics in metrics_data['metrics'].items():
                if isinstance(service_metrics, dict):
                    # Extract relevant KPIs from service metrics
                    kpi_data[f'{service_name}_cpu_usage'] = service_metrics.get('cpu_usage', 0)
                    kpi_data[f'{service_name}_memory_usage'] = service_metrics.get('memory_usage', 0)
                    kpi_data[f'{service_name}_request_count'] = service_metrics.get('request_count', 0)
                    kpi_data[f'{service_name}_response_time'] = service_metrics.get('response_time', 0)
                    kpi_data[f'{service_name}_error_rate'] = service_metrics.get('error_rate', 0)
        
        # Add to performance log
        self.performance_log.append(kpi_data)
        
        # Print detailed entry to console
        print(f"\n📊 PERFORMANCE LOG ENTRY")
        print(f"⏰ Timestamp: {timestamp}")
        print(f"🔧 Configuration: {self.current_configuration}")
        print(f"📋 Phase: {self.current_phase}")
        print(f"⏱️  Interval: {interval_seconds:.1f}s")
        print(f"🚨 Anomalies detected: {kpi_data['anomalies_detected']}")
        print(f"📈 Services monitored: {kpi_data['metrics_count']}")
        print(f"📊 Has trace data: {kpi_data['has_trace_data']}")
        
        # Print stabilization information if in stabilization phase
        if self.current_phase == "stabilization" and stabilization_progress is not None:
            print(f"🔄 Stabilization: {stabilization_progress*100:.1f}% complete ({kpi_data['stabilization_elapsed']:.1f}s elapsed)")
        
        # Print service-specific metrics if available
        if metrics_data and 'metrics' in metrics_data:
            print(f"🔍 Service metrics:")
            for service_name, service_metrics in metrics_data['metrics'].items():
                if isinstance(service_metrics, dict):
                    cpu = service_metrics.get('cpu_usage', 0)
                    memory = service_metrics.get('memory_usage', 0)
                    requests = service_metrics.get('request_count', 0)
                    response_time = service_metrics.get('response_time', 0)
                    error_rate = service_metrics.get('error_rate', 0)
                    print(f"  📦 {service_name}: CPU={cpu:.1f}%, Memory={memory:.1f}MB, Requests={requests}, Response={response_time:.1f}ms, Errors={error_rate:.1f}%")
        
        # Print anomaly details if any
        if anomalies:
            print(f"🚨 Anomaly details:")
            for service_name, anomaly_index, duration in anomalies:
                print(f"  ⚠️  {service_name}: {duration} (index: {anomaly_index})")
        
        print(f"📝 Total entries logged: {len(self.performance_log)}")
        print("-" * 60)
        
    def get_configuration_summary(self, configuration_name=None):
        """
        Get performance summary for a specific configuration or current configuration.
        
        Args:
            configuration_name: Name of configuration to summarize (defaults to current)
            
        Returns:
            dict: Summary statistics for the configuration
        """
        target_config = configuration_name or self.current_configuration
        
        if not target_config:
            return {}
            
        config_logs = [log for log in self.performance_log if log['configuration'] == target_config]
        
        if not config_logs:
            return {}
            
        summary = {
            'configuration': target_config,
            'total_intervals': len(config_logs),
            'total_anomalies': sum(log['anomalies_detected'] for log in config_logs),
            'avg_anomalies_per_interval': sum(log['anomalies_detected'] for log in config_logs) / len(config_logs),
            'total_monitoring_time': sum(log['interval_seconds'] for log in config_logs),
            'first_logged': config_logs[0]['timestamp'],
            'last_logged': config_logs[-1]['timestamp']
        }
        
        return summary
        
    def save_performance_log(self, filename=None):
        """
        Save the performance log to a JSON file.
        
        Args:
            filename: Optional filename (defaults to auto-generated)
        """
        import json
        from config import PATH
        
        if not filename:
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{PATH}/output/{self.label}/performance_log_{timestamp}.json"
            
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.performance_log, f, indent=2, default=str)
            
        print(f"Performance log saved to: {filename}")
        
        # Print performance log to console
        print("\n" + "="*80)
        print("PERFORMANCE LOG SUMMARY")
        print("="*80)
        
        if not self.performance_log:
            print("No performance data logged yet.")
        else:
            # Group by configuration
            configs = {}
            for entry in self.performance_log:
                config = entry.get('configuration', 'Unknown')
                if config not in configs:
                    configs[config] = []
                configs[config].append(entry)
            
            for config_name, entries in configs.items():
                print(f"\nConfiguration: {config_name}")
                print("-" * 50)
                
                total_intervals = len(entries)
                total_anomalies = sum(entry.get('anomalies_detected', 0) for entry in entries)
                total_monitoring_time = sum(entry.get('interval_seconds', 0) for entry in entries)
                
                print(f"  Total monitoring intervals: {total_intervals}")
                print(f"  Total anomalies detected: {total_anomalies}")
                print(f"  Total monitoring time: {total_monitoring_time} seconds")
                
                if entries:
                    first_entry = entries[0]
                    last_entry = entries[-1]
                    print(f"  First logged: {first_entry.get('timestamp', 'Unknown')}")
                    print(f"  Last logged: {last_entry.get('timestamp', 'Unknown')}")
                    
                    # Show recent entries
                    print(f"  Recent entries:")
                    for entry in entries[-3:]:  # Show last 3 entries
                        timestamp = entry.get('timestamp', 'Unknown')
                        anomalies = entry.get('anomalies_detected', 0)
                        interval = entry.get('interval_seconds', 0)
                        print(f"    {timestamp}: {anomalies} anomalies, {interval}s interval")
        
        print("="*80)
        
        return filename


def convert_trace_data_to_dataframe(trace_data):
    """
    Convert raw Jaeger trace data to DataFrame format expected by anomaly detector.
    
    Args:
        trace_data: Raw trace data from Jaeger
        
    Returns:
        pandas.DataFrame: Trace data in DataFrame format
    """
    import pandas as pd
    import sys
    
    if not trace_data or 'data' not in trace_data:
        return pd.DataFrame()
    
    rows = []
    for trace in trace_data["data"]:
        # Create a mapping of process ID to service name
        process_to_service = {}
        if 'processes' in trace:
            for pid, process_info in trace['processes'].items():
                service_name = process_info.get('serviceName', 'unknown')
                process_to_service[pid] = service_name
        
        # Create a row for each trace with service operation columns
        row = {"id": trace['traceID']}
        
        # Initialize all service operation columns to 0
        total_duration = 0
        min_start_time = sys.maxsize
        
        # Process each span to build service operation columns
        for s in trace["spans"]:
            operation_name = s.get("operationName", "")
            duration = s.get("duration", 0)
            start_time = s.get("startTime", 0)
            
            # Add this operation as a column with its duration
            row[operation_name] = duration
            total_duration += duration
            min_start_time = min(min_start_time, start_time)
        
        # Set trace-level information
        row["total"] = total_duration
        row["startTime"] = min_start_time
        row["error"] = False  # Will be set based on span errors if needed
        
        rows.append(row)

    # Fill NaN and create DataFrame
    trace_df = pd.DataFrame(rows).fillna(0)
    
    # Convert times
    trace_df["startTime"] = pd.to_datetime(trace_df['startTime'], unit='us')
    
    # Create operation patterns from the service operation columns
    # Get all columns that look like service operations
    service_cols = [col for col in trace_df.columns if any(pattern in col.lower() for pattern in ['http', 'grpc', '/', 'memcached', 'mongo'])]
    if service_cols:
        # Create pattern based on which operations have non-zero values
        trace_df["pattern"] = trace_df[service_cols].gt(0).astype(int).astype(str).agg("".join, axis=1)
    else:
        trace_df["pattern"] = ""

    return trace_df


def create_anomaly_detector():
    """
    Factory function to create a continuous anomaly detector.
    
    Returns:
        ContinuousAnomalyDetector: configured for anomaly detection
    """
    return ContinuousAnomalyDetector()


def calculate_latency_metrics_from_traces(trace_data, window_start=None, window_end=None):
    """
    Calculate latency metrics (p99, p50, throughput, errors) from Jaeger trace data.
    This replaces the need for history_df from Locust.
    
    Args:
        trace_data: pandas DataFrame with trace data
        window_start: datetime for window start (optional)
        window_end: datetime for window end (optional)
        
    Returns:
        dict: latency and throughput metrics
    """
    import pandas as pd
    import numpy as np
    
    if trace_data.empty:
        return {
            "p99": 0.0,
            "p50": 0.0,
            "max_throughput": 0.0,
            "total_errors": 0.0
        }
    
    # Filter by time window if provided
    if window_start and window_end and 'startTime' in trace_data.columns:
        if not pd.api.types.is_datetime64_any_dtype(trace_data['startTime']):
            trace_data['startTime'] = pd.to_datetime(trace_data['startTime'])
        
        window_data = trace_data[
            (trace_data['startTime'] >= window_start) & 
            (trace_data['startTime'] <= window_end)
        ].copy()
    else:
        window_data = trace_data.copy()
    
    if window_data.empty:
        return {
            "p99": 0.0,
            "p50": 0.0,
            "max_throughput": 0.0,
            "total_errors": 0.0
        }
    
    # Calculate latency percentiles from total duration
    if 'total' in window_data.columns:
        # Convert microseconds to milliseconds for latency
        latencies_ms = window_data['total'] / 1000.0
        
        p99 = np.percentile(latencies_ms, 99) if len(latencies_ms) > 0 else 0.0
        p50 = np.percentile(latencies_ms, 50) if len(latencies_ms) > 0 else 0.0
    else:
        p99 = p50 = 0.0
    
    # Calculate throughput (requests per second)
    if 'startTime' in window_data.columns and len(window_data) > 1:
        # Calculate time span
        time_span = (window_data['startTime'].max() - window_data['startTime'].min()).total_seconds()
        if time_span > 0:
            max_throughput = len(window_data) / time_span
        else:
            max_throughput = 0.0
    else:
        max_throughput = 0.0
    
    # For errors, we would need to check trace status or error fields
    # For now, assume no errors (this could be enhanced by checking trace status)
    total_errors = 0.0
    
    return {
        "p99": p99,
        "p50": p50,
        "max_throughput": max_throughput,
        "total_errors": total_errors
    }


def get_prometheus_throughput_and_errors(config_name, revision_name, window_start, window_end):
    """
    Get throughput and error metrics from Prometheus.
    
    Args:
        config_name: service configuration name
        revision_name: service revision name
        window_start: datetime for window start
        window_end: datetime for window end
        
    Returns:
        dict: throughput and error metrics
    """
    import requests
    from datetime import datetime
    
    try:
        url = PROMETHEUS_BASE_URL + '/api/v1/query_range'
        
        # Query for request rate (throughput)
        throughput_query = f'sum(rate(activator_request_count{{namespace_name="default", configuration_name="{config_name}", revision_name="{revision_name}"}}[1m]))'
        
        # Query for error rate
        error_query = f'sum(rate(activator_request_count{{namespace_name="default", configuration_name="{config_name}", revision_name="{revision_name}", response_code_class!="2xx"}}[1m]))'
        
        params = {
            'query': throughput_query,
            'start': window_start.isoformat() + 'Z',
            'end': window_end.isoformat() + 'Z',
            'step': '5s'
        }
        
        response = requests.get(url, params=params)
        result = response.json()
        
        max_throughput = 0.0
        total_errors = 0.0
        
        if result['status'] == 'success' and result['data']['result']:
            # Get max throughput from the time series
            values = result['data']['result'][0]['values']
            if values:
                max_throughput = max(float(v[1]) for v in values)
        
        # Get error rate
        params['query'] = error_query
        response = requests.get(url, params=params)
        result = response.json()
        
        if result['status'] == 'success' and result['data']['result']:
            values = result['data']['result'][0]['values']
            if values:
                total_errors = max(float(v[1]) for v in values)
        
        return {
            "max_throughput": max_throughput,
            "total_errors": total_errors
        }
        
    except Exception as e:
        print(f"Error getting Prometheus metrics: {e}")
        return {
            "max_throughput": 0.0,
            "total_errors": 0.0
        }
