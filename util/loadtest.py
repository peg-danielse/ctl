"""
Load testing utilities for the CTL system.
Handles Locust load test execution and data management.
"""

import os
import glob
import shutil
import subprocess
import time
import hashlib
import threading
import queue
import json
import logging
from datetime import datetime

from config import PATH
from util.analysis import read_response, read_traces, read_metrics


LOAD_TEST_TIME = '30m'

def simple_hash(s):
    """Generate a simple hash for seed generation."""
    hash_object = hashlib.sha256(s.encode())
    hex_digest = hash_object.hexdigest()
    int_hash = int(hex_digest, 16)
    return int_hash % 10000


def locust_load_test(label, base_label, seed=0):
    """
    Execute a Locust load test with the specified parameters.
    
    Args:
        label: Test label for output files
        base_label: Base label for directory structure
        seed: Seed number for reproducibility (default: 0)
    """
    if len(glob.glob(PATH + "/output/" + f"{base_label}/data/{label}")) != 0:
        print(f"Load test using label: {label} already exists. Skipping...")
        return

    cmd = [
        "locust",
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
        "--seed", str(simple_hash(f"fix_{seed}"))
    ]

    print(f"Test {label} started...\n")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=2400  # 40 minutes - increased timeout to handle 30min test + processing time
    )

    print(f"Test {label} finished")
    time.sleep(30)

    # Move experiment files to output directory
    experiment_files = glob.glob(f"./{label}_*")
    os.makedirs(PATH + f"/output/{base_label}/data/{label}", exist_ok=True)
    for file in experiment_files:
        shutil.move(file, PATH + f"/output/{base_label}/data/{label}/")

    return


class AsyncLoadTest:
    """
    Asynchronous load test runner that runs in a separate thread.
    """
    
    def __init__(self, label, base_label, seed=0):
        self.label = label
        self.base_label = base_label
        self.seed = seed
        self.process = None
        self.thread = None
        self.is_running = False
        self.is_completed = False
        self.error = None
        
    def _run_load_test(self):
        """Internal method to run the load test in a separate thread."""
        try:
            self.is_running = True
            print(f"Starting async load test: {self.label}")
            
            cmd = [
                "locust",
                "--processes", "32",
                "-f", "./locust/hotel-reservations.py",
                "-H", "http://145.100.135.11:30505",
                "-t", LOAD_TEST_TIME,
                "--csv", self.label,
                "--headless",
                "--w-user-min", str(1000),
                "--w-user-max", str(6000),
                "--w-mean", str(3000),
                "--w-ls-y", str(10000),
                "--w-dt", str(40),
                "--seed", str(simple_hash(f"fix_{self.seed}"))
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for the process to complete
            stdout, stderr = self.process.communicate(timeout=2400)  # 40 minutes timeout
            
            print(f"Async load test {self.label} finished")
            
            # Move experiment files to output directory
            experiment_files = glob.glob(f"./{self.label}_*")
            os.makedirs(PATH + f"/output/{self.base_label}/data/{self.label}", exist_ok=True)
            for file in experiment_files:
                shutil.move(file, PATH + f"/output/{self.base_label}/data/{self.label}/")
            
            self.is_completed = True
            
        except subprocess.TimeoutExpired:
            print(f"Async load test {self.label} timed out")
            if self.process:
                self.process.kill()
            self.error = "Load test timed out"
        except Exception as e:
            self.error = str(e)
            print(f"Error in async load test {self.label}: {e}")
        finally:
            self.is_running = False
    
    def start(self):
        """Start the load test in a separate thread."""
        if self.is_running or self.is_completed:
            return False
            
        self.thread = threading.Thread(target=self._run_load_test)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def stop(self):
        """Stop the load test."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.is_running = False
    
    def is_ready(self):
        """Check if the load test has completed and data is ready."""
        return self.is_completed and not self.is_running
    
    def get_status(self):
        """Get the current status of the load test."""
        return {
            'is_running': self.is_running,
            'is_completed': self.is_completed,
            'error': self.error
        }


def start_async_load_test(label, base_label):
    """
    Start an asynchronous load test.
    
    Args:
        label: Test label
        base_label: Base label for directory structure
        
    Returns:
        AsyncLoadTest: The async load test instance
    """
    async_test = AsyncLoadTest(label, base_label, 0)
    async_test.start()
    return async_test


def read_test_data(label, suffix=""):
    """
    Read test data from a completed load test.
    
    Args:
        label: Test label
        suffix: Optional suffix for the data files (e.g., "baseline", "continuous")
        
    Returns:
        tuple: (response_df, trace_df, metric_dfs)
    """
    try:
        data_label = f"{label}_{suffix}" if suffix else label
        response_df = read_response(data_label, label)
        trace_df = read_traces(data_label, label)
        metric_dfs = read_metrics(data_label, label)
        return response_df, trace_df, metric_dfs

    except FileNotFoundError:
        shutil.rmtree(PATH + f"/output/{label}/data/{data_label}")
        print("Reading the test data failed... retrying load test")
        locust_load_test(data_label, label, 0)
        return read_test_data(label, suffix)


class AsyncSnapshotLogger:
    """
    Asynchronous snapshot logger that runs monitoring and logging in a separate thread.
    """
    
    def __init__(self, label, base_label, snapshot_interval=30, continuous_monitor=None, 
                 anomaly_detector=None, performance_logger=None, workflow_orchestrator=None):
        self.label = label
        self.base_label = base_label
        self.snapshot_interval = snapshot_interval
        self.is_running = False
        self.thread = None
        self.stop_event = threading.Event()
        self.snapshot_queue = queue.Queue()
        self.anomaly_queue = queue.Queue()  # Queue for anomalies to be processed by main workflow
        
        # Phase and subphase tracking
        self.current_phase = None
        self.current_subphase = None
        
        # Monitoring components
        self.continuous_monitor = continuous_monitor
        self.anomaly_detector = anomaly_detector
        self.performance_logger = performance_logger
        self.workflow_orchestrator = workflow_orchestrator
        
        # Setup logging
        self.logger = logging.getLogger(f"snapshot_logger_{label}")
        self.logger.setLevel(logging.INFO)
        
        # Create snapshot directory
        self.snapshot_dir = f"{PATH}/output/{base_label}/snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Setup unified file handler for snapshots (no phase suffix)
        log_file = f"{self.snapshot_dir}/snapshot_log_{base_label}.jsonl"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        print(f"AsyncSnapshotLogger initialized for {label}, logging to {log_file}")
    
    def set_phase(self, phase):
        """
        Set the current phase (baseline, adaptation, etc.).
        
        Args:
            phase (str): The current phase name
        """
        self.current_phase = phase
        print(f"📊 Snapshot logger phase set to: {phase}")
    
    def set_subphase(self, subphase):
        """
        Set the current subphase (configuration_application, stabilization, etc.).
        
        Args:
            subphase (str or None): The current subphase name, or None to clear
        """
        self.current_subphase = subphase
        if subphase:
            print(f"📊 Snapshot logger subphase set to: {subphase}")
        else:
            print(f"📊 Snapshot logger subphase cleared")
    
    def get_phase_info(self):
        """
        Get current phase and subphase information.
        
        Returns:
            dict: Current phase and subphase
        """
        return {
            'phase': self.current_phase,
            'subphase': self.current_subphase
        }
    
    def start(self):
        """Start the asynchronous snapshot logging thread."""
        if self.is_running:
            return False
            
        self.is_running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._snapshot_worker)
        self.thread.daemon = True
        self.thread.start()
        print(f"AsyncSnapshotLogger started for {self.label}")
        return True
    
    def stop(self):
        """Stop the asynchronous snapshot logging thread."""
        if not self.is_running:
            return
            
        self.stop_event.set()
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        print(f"AsyncSnapshotLogger stopped for {self.label}")
    
    def _snapshot_worker(self):
        """Worker thread that runs monitoring and creates snapshots."""
        last_monitoring_time = 0
        last_snapshot_time = 0
        snapshot_count = 0
        monitoring_interval = 60  # Monitor every 60 seconds
        
        print(f"🔄 Starting monitoring and snapshot worker for {self.label}")
        
        while not self.stop_event.is_set():
            current_time = time.time()
            
            # Check if it's time for monitoring
            if current_time - last_monitoring_time >= monitoring_interval:
                try:
                    print(f"📊 Performing monitoring cycle at {datetime.now()}")
                    
                    # Collect monitoring metrics
                    if self.continuous_monitor:
                        current_metrics = self.continuous_monitor.collect_current_metrics(
                            self.label, 
                            interval_seconds=monitoring_interval
                        )
                        
                        if current_metrics and current_metrics.get('metrics'):
                            print(f"Collected metrics for {len(current_metrics['metrics'])} configurations")
                            
                            # Create snapshot from monitoring data
                            snapshot_data = self._create_monitoring_snapshot(current_metrics)
                            if snapshot_data:
                                self._log_snapshot(snapshot_data)
                                snapshot_count += 1
                                print(f"📊 Created monitoring snapshot #{snapshot_count} for {self.label}")
                                
                                # Process anomalies if available
                                if self.workflow_orchestrator and current_metrics.get('traces'):
                                    self._process_anomalies_in_thread(current_metrics)
                            
                            # Log performance metrics
                            if self.performance_logger:
                                anomalies = []
                                if self.anomaly_detector and current_metrics.get('traces'):
                                    from util.monitoring import convert_trace_data_to_dataframe
                                    trace_df = convert_trace_data_to_dataframe(current_metrics['traces'])
                                    if not trace_df.empty:
                                        anomalies = self.anomaly_detector.detect_anomalies(trace_df, exact_interval_seconds=monitoring_interval)
                                
                                self.performance_logger.log_interval_metrics(current_metrics, monitoring_interval, anomalies)
                        else:
                            print("❌ No metrics data available for monitoring cycle")
                            
                    last_monitoring_time = current_time
                    
                except Exception as e:
                    print(f"Error in monitoring cycle: {e}")
            
            # Also process any queued snapshots (for backward compatibility)
            try:
                while True:
                    try:
                        snapshot_data = self.snapshot_queue.get_nowait()
                        self._log_snapshot(snapshot_data)
                        snapshot_count += 1
                        print(f"📊 Processed queued snapshot #{snapshot_count} for {self.label}")
                    except queue.Empty:
                        break
            except Exception as e:
                print(f"Error processing queued snapshots: {e}")
            
            # Print periodic status update
            if snapshot_count > 0 and snapshot_count % 10 == 0:
                print(f"🔄 Snapshot logger status: {snapshot_count} snapshots logged")
            
            # Sleep for a short time to prevent busy waiting
            time.sleep(1)
    
    def _create_monitoring_snapshot(self, current_metrics):
        """Create a snapshot from monitoring data."""
        try:
            from util.monitoring import convert_trace_data_to_dataframe
            from util.analysis import metric_snapshot
            import pandas as pd
            
            # Get trace data (optional)
            trace_df = None
            if current_metrics.get('traces'):
                trace_df = convert_trace_data_to_dataframe(current_metrics['traces'])
            
            # Create snapshots for each service
            snapshots = []
            all_anomalies = []
            total_anomaly_count = 0
            
            for service_name in current_metrics['metrics'].keys():
                try:
                    # Get service-specific trace data (if available)
                    if trace_df is not None:
                        service_traces = self._get_service_traces(trace_df, service_name)
                        
                        if not service_traces.empty and self.anomaly_detector:
                            # Detect anomalies for this specific service
                            service_anomalies = self.anomaly_detector.detect_anomalies(service_traces, exact_interval_seconds=60)
                            service_anomaly_count = len(service_anomalies)
                            
                            # Add service prefix to anomaly tuples
                            service_anomalies_with_prefix = [(service_name, idx, duration) for _, idx, duration in service_anomalies]
                            all_anomalies.extend(service_anomalies_with_prefix)
                            total_anomaly_count += service_anomaly_count
                        else:
                            service_anomalies = []
                            service_anomaly_count = 0
                    else:
                        # No trace data available, create empty DataFrame for metric_snapshot
                        service_traces = pd.DataFrame()
                        service_anomalies = []
                        service_anomaly_count = 0
                    
                    # Create snapshot for this service
                    timestamp, duration, snapshot = metric_snapshot(
                        service_name, service_traces, current_metrics['metrics'],
                        phase=self.current_phase, subphase=self.current_subphase
                    )
                    
                    # Add anomaly information to the snapshot
                    snapshot['anomaly_count'] = service_anomaly_count
                    snapshot['anomalies'] = service_anomalies
                    snapshots.append(snapshot)
                    
                except Exception as e:
                    print(f"Error creating snapshot for {service_name}: {e}")
            
            # Create the main snapshot data structure
            snapshot_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'label': self.label,
                'type': 'monitoring_snapshot',
                'phase': self.current_phase,
                'subphase': self.current_subphase,
                'snapshots': snapshots,
                'anomaly_count': total_anomaly_count,
                'service_count': len(snapshots),
                'monitoring_interval': 60
            }
            
            return snapshot_data
            
        except Exception as e:
            print(f"Error creating monitoring snapshot: {e}")
            return None
    
    def _get_service_traces(self, trace_df, service_name):
        """Get traces for a specific service using SPAN_PROCESS_MAP."""
        try:
            import pandas as pd
            if trace_df is None or trace_df.empty:
                return pd.DataFrame()
            
            # Import SPAN_PROCESS_MAP to map operations to services
            from config import SPAN_PROCESS_MAP
            
            # Find all operation columns that belong to this service
            service_operations = []
            for operation, mapped_service in SPAN_PROCESS_MAP.items():
                if mapped_service == service_name:
                    if operation in trace_df.columns:
                        service_operations.append(operation)
            
            if service_operations:
                # Filter traces where any of these operations have duration > 0
                service_traces = trace_df[trace_df[service_operations].sum(axis=1) > 0].copy()
                return service_traces
            else:
                # No operations found for this service
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting service traces for {service_name}: {e}")
            import pandas as pd
            return pd.DataFrame()
    
    def _process_anomalies_in_thread(self, current_metrics):
        """Detect anomalies and queue them for processing by main workflow."""
        try:
            if not self.workflow_orchestrator or not current_metrics.get('traces'):
                return
            
            from util.monitoring import convert_trace_data_to_dataframe
            trace_df = convert_trace_data_to_dataframe(current_metrics['traces'])
            
            if not trace_df.empty and self.anomaly_detector:
                anomalies = self.anomaly_detector.detect_anomalies(trace_df, exact_interval_seconds=60)
                
                if anomalies:
                    print(f"🔍 Found {len(anomalies)} anomalies in monitoring thread:")
                    for service_name, anomaly_index, duration in anomalies:
                        print(f"  - {service_name}: {duration} (index: {anomaly_index})")
                    
                    # Queue anomalies for processing by main workflow
                    anomaly_data = {
                        'anomalies': anomalies,
                        'trace_df': trace_df,
                        'current_metrics': current_metrics,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    try:
                        self.anomaly_queue.put_nowait(anomaly_data)
                        print(f"📝 Queued {len(anomalies)} anomalies for processing by main workflow")
                    except queue.Full:
                        print("⚠️ Anomaly queue full, dropping anomalies")
                    
        except Exception as e:
            print(f"Error processing anomalies in thread: {e}")
    
    def _create_basic_snapshot(self):
        """Create a basic snapshot when no data is available."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'label': self.label,
            'type': 'basic_snapshot',
            'message': 'No specific snapshot data available',
            'snapshot_interval': self.snapshot_interval
        }
    
    def _log_snapshot(self, snapshot_data):
        """Log snapshot data to the JSONL file and print to console."""
        try:
            # Add metadata
            snapshot_data['logged_at'] = datetime.utcnow().isoformat()
            snapshot_data['logger_label'] = self.label
            
            # Log as JSON line
            self.logger.info(json.dumps(snapshot_data, default=str))
            
            # Print to console for visibility
            self._print_snapshot_to_console(snapshot_data)
            
        except Exception as e:
            print(f"Error logging snapshot: {e}")
    
    def _print_snapshot_to_console(self, snapshot_data):
        """Print snapshot data to console in a readable format."""
        try:
            timestamp = snapshot_data.get('timestamp', 'Unknown time')
            snapshot_type = snapshot_data.get('type', 'unknown')
            
            print(f"\n📊 SNAPSHOT LOGGED [{timestamp}] - Type: {snapshot_type}")
            
            if snapshot_type in ['metric_snapshot', 'baseline_metric_snapshot'] and 'snapshots' in snapshot_data:
                snapshots = snapshot_data['snapshots']
                service_count = snapshot_data.get('service_count', len(snapshots))
                total_anomalies = snapshot_data.get('anomaly_count', 0)
                print(f"   Services monitored: {service_count}")
                if total_anomalies > 0:
                    print(f"   🚨 Total anomalies detected: {total_anomalies}")
                
                for snapshot in snapshots:
                    service_name = snapshot.get('service_name', 'Unknown')
                    response_time = snapshot.get('response_time_avg', 0)
                    total_requests = snapshot.get('total_requests', 0)
                    error_rate = snapshot.get('error_rate', 0)
                    service_anomalies = snapshot.get('anomaly_count', 0)
                    
                    anomaly_indicator = "🚨" if service_anomalies > 0 else "✅"
                    print(f"   {anomaly_indicator} {service_name}: {response_time:.2f}ms avg, {total_requests} requests, {error_rate:.1f}% errors, {service_anomalies} anomalies")
                    
                    # Print service metrics if available
                    service_metrics = snapshot.get('service_metrics', {})
                    if service_metrics:
                        cpu_usage = service_metrics.get('cpu_usage', 0)
                        memory_usage = service_metrics.get('memory_usage', 0)
                        if cpu_usage > 0 or memory_usage > 0:
                            print(f"     └─ CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%")
                    
                    # Print anomaly details if any
                    if service_anomalies > 0:
                        anomalies = snapshot.get('anomalies', [])
                        for anomaly in anomalies[:3]:  # Show first 3 anomalies
                            if len(anomaly) >= 3:
                                service, index, duration = anomaly[0], anomaly[1], anomaly[2]
                                
                                # Handle different duration formats
                                if hasattr(duration, 'total_seconds'):
                                    # Timedelta object
                                    duration_ms = duration.total_seconds() * 1000
                                    print(f"     └─ 🚨 Anomaly: {duration_ms:.2f}ms duration (index: {index})")
                                elif isinstance(duration, (int, float)):
                                    # Numeric value (already in microseconds or milliseconds)
                                    if duration > 1000:  # Likely microseconds
                                        duration_ms = duration / 1000
                                    else:  # Likely already milliseconds
                                        duration_ms = duration
                                    print(f"     └─ 🚨 Anomaly: {duration_ms:.2f}ms duration (index: {index})")
                                else:
                                    # Fallback for other types
                                    print(f"     └─ 🚨 Anomaly: {duration} duration (index: {index})")
                        if len(anomalies) > 3:
                            print(f"     └─ ... and {len(anomalies) - 3} more anomalies")
            
            elif snapshot_type == 'basic_snapshot':
                message = snapshot_data.get('message', 'No message')
                print(f"   {message}")
            
            else:
                # Generic snapshot data
                if 'message' in snapshot_data:
                    print(f"   {snapshot_data['message']}")
                if 'data' in snapshot_data:
                    data = snapshot_data['data']
                    for key, value in data.items():
                        print(f"   • {key}: {value}")
            
            print(f"   Logged to: {self.snapshot_dir}/snapshot_log_{self.label}.jsonl")
            print("─" * 60)
            
        except Exception as e:
            print(f"Error printing snapshot to console: {e}")
    
    def queue_snapshot(self, snapshot_data):
        """Queue snapshot data for logging."""
        try:
            self.snapshot_queue.put_nowait(snapshot_data)
        except queue.Full:
            print(f"Snapshot queue full, dropping snapshot for {self.label}")
    
    def get_queued_anomalies(self):
        """Get all queued anomalies for processing by main workflow."""
        anomalies = []
        try:
            while True:
                anomaly_data = self.anomaly_queue.get_nowait()
                anomalies.append(anomaly_data)
        except queue.Empty:
            pass
        return anomalies
    
    def get_status(self):
        """Get the current status of the snapshot logger."""
        return {
            'is_running': self.is_running,
            'queue_size': self.snapshot_queue.qsize(),
            'anomaly_queue_size': self.anomaly_queue.qsize(),
            'snapshot_interval': self.snapshot_interval,
            'log_file': f"{self.snapshot_dir}/snapshot_log_{self.label}.jsonl"
        }


def create_async_snapshot_logger(label, base_label, snapshot_interval=30, continuous_monitor=None, 
                                 anomaly_detector=None, performance_logger=None, workflow_orchestrator=None):
    """
    Create and start an asynchronous snapshot logger with monitoring capabilities.
    
    Args:
        label: Test label
        base_label: Base label for directory structure
        snapshot_interval: Interval between snapshots in seconds
        continuous_monitor: Continuous monitoring component
        anomaly_detector: Anomaly detection component
        performance_logger: Performance logging component
        workflow_orchestrator: Workflow orchestrator for anomaly processing
        
    Returns:
        AsyncSnapshotLogger: The async snapshot logger instance
    """
    logger = AsyncSnapshotLogger(label, base_label, snapshot_interval, continuous_monitor, 
                                anomaly_detector, performance_logger, workflow_orchestrator)
    logger.start()
    return logger
