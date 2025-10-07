"""
Workflow orchestration utilities for the CTL system.
Handles the main workflow execution and coordination between components.
"""

import time
import datetime
from pprint import pprint

from util.monitoring import create_continuous_monitor, create_anomaly_detector, collect_monitoring_data_for_test, convert_trace_data_to_dataframe, ConfigurationPerformanceLogger
from util.loadtest import locust_load_test, read_test_data, start_async_load_test, create_async_snapshot_logger
from util.config_manager import setup_experiment_directory, load_yaml_as_dict, get_knative_knobs, get_vscaling_knobs
from util.llm_client import generate_configuration_for_anomaly, apply_configuration_changes
from util.openai_client import generate_configuration_with_openai
from util.optimization import analyze_metrics_for_optimization, evolve_configurations
from util.analysis import metric_snapshot
from util.config_logger import create_config_logger
from config import PATH

class WorkflowOrchestrator:
    """
    Orchestrates the main workflow execution for the CTL system.
    """
    
    def __init__(self, label="unknown", llm_provider="local", runtime_minutes=30):
        self.label = label
        self.llm_provider = llm_provider
        self.runtime_minutes = runtime_minutes
        self.continuous_monitor = create_continuous_monitor()
        self.anomaly_detector = create_anomaly_detector()
        self.performance_logger = ConfigurationPerformanceLogger(label)
        self.config_logger = create_config_logger()
        self.snapshot_logger = None
        
    def setup_experiment(self):
        """Set up the experiment directory and initialize monitoring."""
        # Reset Kubernetes to base configuration for consistent experiments
        print("🔄 Resetting Kubernetes cluster to base configuration...")
        try:
            from util.square import get_k8s_api_client, reset_k8s
            api_client = get_k8s_api_client()
            reset_k8s(api_client)
            print("✅ Kubernetes cluster reset to base configuration")
        except Exception as e:
            print(f"⚠️  Could not reset Kubernetes: {e}")
            print("   Continuing with current cluster state...")
        
        setup_experiment_directory(self.label)
        print("Continuous monitoring and anomaly detection initialized")
        
        # Start tracking the initial/base configuration immediately
        initial_config_name = f"initial_config_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.performance_logger.start_configuration_tracking(initial_config_name)
        self.performance_logger.set_current_phase("initialization")
        print(f"📊 Started tracking initial configuration: {initial_config_name}")
        
        # Initialize anomaly detector with existing datasets
        print("Initializing anomaly detector with existing datasets...")
        success = self.initialize_anomaly_detector_with_datasets()
        if success:
            print("✅ Anomaly detector ready for monitoring")
        else:
            print("⚠️  Anomaly detector will use statistical methods (no pre-trained model)")
    
    def initialize_anomaly_detector_with_datasets(self, dataset_folder="dataset"):
        """
        Initialize the anomaly detector using existing dataset files.
        
        Args:
            dataset_folder: Path to the folder containing dataset files
            
        Returns:
            bool: True if initialization was successful
        """
        import os
        import json
        import glob
        
        print(f"Initializing anomaly detector with datasets from {dataset_folder}...")
        
        # Find all trace files in the dataset folder
        dataset_path = os.path.join(PATH, dataset_folder)
        trace_files = glob.glob(os.path.join(dataset_path, "*_traces.json"))
        
        if not trace_files:
            print(f"No trace files found in {dataset_path}")
            return False
        
        print(f"Found {len(trace_files)} trace files: {[os.path.basename(f) for f in trace_files]}")
        
        all_traces = []
        
        # Load and combine all trace files
        for trace_file in trace_files:
            try:
                print(f"Loading traces from {os.path.basename(trace_file)}...")
                with open(trace_file, 'r') as f:
                    data = json.load(f)
                
                if 'data' in data:
                    all_traces.extend(data['data'])
                    print(f"  - Loaded {len(data['data'])} traces")
                else:
                    print(f"  - No 'data' field found in {os.path.basename(trace_file)}")
                    
            except Exception as e:
                print(f"  - Error loading {os.path.basename(trace_file)}: {e}")
                continue
        
        if not all_traces:
            print("No traces loaded from dataset files")
            return False
        
        print(f"Total traces loaded: {len(all_traces)}")
        
        # Convert traces to DataFrame format
        try:
            # Wrap the traces in the expected format
            trace_data = {'data': all_traces}
            trace_df = convert_trace_data_to_dataframe(trace_data)
            
            if trace_df.empty:
                print("No valid trace data after conversion")
                return False
            
            print(f"Converted to DataFrame with {len(trace_df)} rows and {len(trace_df.columns)} columns")
            print(f"Columns: {list(trace_df.columns)}")
            
            # Train the anomaly detector
            success = self.anomaly_detector.train_model(trace_df)
            
            if success:
                print("✅ Anomaly detector successfully initialized with dataset files")
                return True
            else:
                print("❌ Failed to train anomaly detector on dataset files")
                return False
                
        except Exception as e:
            print(f"Error converting traces to DataFrame: {e}")
            return False
    
    def _get_service_traces(self, trace_df, service_name):
        """
        Filter trace data to get traces relevant to a specific service.
        
        Args:
            trace_df: DataFrame with all trace data
            service_name: Name of the service to filter for
            
        Returns:
            pandas.DataFrame: Filtered trace data for the specific service
        """
        import pandas as pd
        from config import SPAN_PROCESS_MAP
        
        if trace_df.empty:
            return pd.DataFrame()
        
        # Get the span names that belong to this service
        service_spans = []
        for span_name, mapped_service in SPAN_PROCESS_MAP.items():
            if mapped_service == service_name:
                service_spans.append(span_name)
        
        if not service_spans:
            # If no specific spans found, return empty DataFrame
            return pd.DataFrame()
        
        # Filter traces that have any of the service's spans
        service_trace_mask = pd.Series([False] * len(trace_df), index=trace_df.index)
        
        for span_name in service_spans:
            if span_name in trace_df.columns:
                # Include traces where this span has a positive value (was executed)
                service_trace_mask |= (trace_df[span_name] > 0)
        
        # Return filtered traces
        service_traces = trace_df[service_trace_mask].copy()
        
        # If we have service-specific traces, create a simplified version with only relevant columns
        if not service_traces.empty:
            # Keep only the spans relevant to this service plus essential columns
            relevant_columns = ['id', 'startTime', 'total'] + service_spans
            available_columns = [col for col in relevant_columns if col in service_traces.columns]
            service_traces = service_traces[available_columns]
        
        return service_traces
    
    def run_baseline_phase(self):
        """
        Run the baseline load test phase to train the anomaly detection model.
        
        Returns:
            bool: True if baseline phase completed successfully
        """
        print("Step 1: Running baseline load test to train isolation forest...")
        baseline_label = f"{self.label}_baseline"
        
        # Start tracking performance for baseline configuration
        baseline_config_name = f"baseline_config_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.performance_logger.start_configuration_tracking(baseline_config_name)
        self.performance_logger.set_current_phase("baseline")
        
        # Run baseline load test asynchronously
        print(f"Starting async baseline load test: {baseline_label}")
        async_load_test = start_async_load_test(baseline_label, self.label)
        
        # Start asynchronous snapshot logger for baseline
        print(f"🔄 Starting async snapshot logger for baseline: {baseline_label}")
        baseline_snapshot_logger = create_async_snapshot_logger(
            baseline_label, self.label, snapshot_interval=30,
            continuous_monitor=self.continuous_monitor,
            anomaly_detector=self.anomaly_detector,
            performance_logger=self.performance_logger,
            workflow_orchestrator=self
        )
        
        # Set phase to baseline
        baseline_snapshot_logger.set_phase("baseline")
        
        print(f"✅ Baseline snapshot logger started - snapshots will be logged every 30 seconds")
        
        # Monitor baseline load test with periodic snapshots
        baseline_duration_seconds = self.runtime_minutes * 60
        print(f"Monitoring baseline load test with periodic snapshots ({self.runtime_minutes} minutes)...")
        baseline_start_time = time.time()
        last_baseline_snapshot_time = 0
        baseline_snapshot_interval = 30  # seconds
        
        while time.time() - baseline_start_time < baseline_duration_seconds:
            current_time = time.time()
            
            # Check if it's time for a baseline snapshot
            if current_time - last_baseline_snapshot_time >= baseline_snapshot_interval:
                try:
                    # Collect current metrics for baseline snapshot
                    current_metrics = self.continuous_monitor.collect_current_metrics(baseline_label, interval_seconds=30)
                    
                    if current_metrics and current_metrics.get('metrics'):
                        # Queue snapshot for async logging
                        baseline_snapshot_data = {
                            'timestamp': datetime.datetime.utcnow().isoformat(),
                            'label': baseline_label,
                            'type': 'baseline_metric_snapshot',
                            'snapshots': [],
                            'service_count': len(current_metrics['metrics'])
                        }
                        
                        # Create snapshots for each service
                        if current_metrics.get('traces'):
                            trace_df = convert_trace_data_to_dataframe(current_metrics['traces'])
                            if not trace_df.empty:
                                
                                # Detect anomalies for each service individually
                                all_anomalies = []
                                total_anomaly_count = 0
                                
                                for service_name in current_metrics['metrics'].keys():
                                    try:
                                        # Get service-specific trace data
                                        service_traces = self._get_service_traces(trace_df, service_name)
                                        
                                        if not service_traces.empty:
                                            # Detect anomalies for this specific service
                                            service_anomalies = self.anomaly_detector.detect_anomalies(service_traces, exact_interval_seconds=30)
                                            service_anomaly_count = len(service_anomalies)
                                            
                                            # Add service prefix to anomaly tuples
                                            service_anomalies_with_prefix = [(service_name, idx, duration) for _, idx, duration in service_anomalies]
                                            all_anomalies.extend(service_anomalies_with_prefix)
                                            total_anomaly_count += service_anomaly_count
                                        else:
                                            service_anomalies = []
                                            service_anomaly_count = 0
                                        
                                        # Create snapshot for this service
                                        timestamp, duration, snapshot = metric_snapshot(
                                            service_name, service_traces, current_metrics['metrics'],
                                            phase="baseline", subphase=None
                                        )
                                        
                                        # Add anomaly information to the snapshot
                                        snapshot['anomaly_count'] = service_anomaly_count
                                        snapshot['anomalies'] = service_anomalies
                                        
                                        baseline_snapshot_data['snapshots'].append(snapshot)
                                        
                                    except Exception as e:
                                        print(f"  - Error creating baseline snapshot for {service_name}: {e}")
                                
                                # Add overall anomaly information to snapshot data
                                baseline_snapshot_data['anomaly_count'] = total_anomaly_count
                                baseline_snapshot_data['anomalies'] = all_anomalies
                        
                        baseline_snapshot_logger.queue_snapshot(baseline_snapshot_data)
                        print(f"📊 Baseline snapshot queued for async logging ({len(baseline_snapshot_data['snapshots'])} services, {baseline_snapshot_data.get('anomaly_count', 0)} anomalies)")
                    
                    last_baseline_snapshot_time = current_time
                    
                except Exception as e:
                    print(f"Error during baseline monitoring: {e}")
                    last_baseline_snapshot_time = current_time
            
            # Short sleep to prevent busy waiting
            time.sleep(1)
        
        # Stop the async load test
        print("Stopping baseline load test...")
        async_load_test.stop()
        
        # Stop the baseline snapshot logger
        print("Stopping baseline snapshot logger...")
        baseline_snapshot_logger.stop()
        
        # Read baseline data and train the model
        try:
            response_df, trace_df, metric_dfs = read_test_data(self.label, "baseline")
            
            # Store baseline data
            print("Storing baseline metrics...")
            baseline_data = {
                'response': response_df, 
                'traces': trace_df,
                'metrics': metric_dfs,
                'timestamp': datetime.datetime.utcnow(),
                'label': baseline_label
            }
            
            # Take metric snapshots for baseline data
            print("Taking baseline metric snapshots...")
            if not trace_df.empty and metric_dfs:
                # Create a mock metrics structure for snapshot
                mock_metrics = {
                    'timestamp': datetime.datetime.utcnow(),
                    'metrics': metric_dfs,
                    'traces': trace_df
                }
                
                # Take snapshots for each service
                for service_name in metric_dfs.keys():
                    try:
                        from util.analysis import metric_snapshot
                        timestamp, duration, snapshot = metric_snapshot(
                            service_name, trace_df, metric_dfs,
                            phase="baseline", subphase=None
                        )
                        print(f"  - {service_name}: {duration:.2f}ms avg response time, {snapshot['total_requests']} requests")
                    except Exception as e:
                        print(f"  - Error taking baseline snapshot for {service_name}: {e}")
                
                # Log baseline performance metrics
                self.performance_logger.log_interval_metrics(mock_metrics, baseline_duration_seconds, [])  # baseline duration
                
                # Save baseline performance log
                self.performance_logger.save_performance_log()
            
            # Train the anomaly detection model
            if not trace_df.empty:
                success = self.anomaly_detector.train_model(trace_df)
                if not success:
                    print("Failed to train anomaly detection model")
                    return False
            else:
                print("No trace data available for training")
                return False
                
            # Generate plots for baseline phase
            try:
                from util.plotting import generate_adaptation_plots
                print(f"\n📊 Generating baseline analysis plots...")
                plot_success = generate_adaptation_plots(baseline_label)
                if plot_success:
                    print(f"✅ Baseline analysis plots generated successfully!")
                else:
                    print(f"⚠️  Could not generate baseline plots")
            except Exception as e:
                print(f"❌ Error generating baseline plots: {e}")
            
            return True
                
        except Exception as e:
            print(f"Error processing baseline data: {e}")
            return False
    
    def run_adaptation_phase(self, duration_seconds=None):
        """
        Run the adaptation phase with continuous monitoring and anomaly detection.
        
        Args:
            duration_seconds: Duration of the monitoring phase in seconds (default: uses self.runtime_minutes)
            
        Returns:
            bool: True if adaptation phase completed successfully
        """
        if duration_seconds is None:
            duration_seconds = self.runtime_minutes * 60
        print("Step 2: Starting adaptation phase with load test and anomaly detection...")
        
        # Set phase to adaptation
        self.performance_logger.set_current_phase("adaptation")
        
        # Start an asynchronous load test (no "continuous" suffix)
        adaptation_label = self.label
        print(f"Starting async load test: {adaptation_label}")
        
        async_load_test = start_async_load_test(adaptation_label, self.label)
        
        # Start asynchronous snapshot logger with monitoring capabilities
        print(f"🔄 Starting async snapshot logger with monitoring for adaptation phase: {adaptation_label}")
        self.snapshot_logger = create_async_snapshot_logger(
            adaptation_label, self.label, snapshot_interval=30,
            continuous_monitor=self.continuous_monitor,
            anomaly_detector=self.anomaly_detector,
            performance_logger=self.performance_logger,
            workflow_orchestrator=self
        )
        
        # Set phase to adaptation
        self.snapshot_logger.set_phase("adaptation")
        
        print(f"✅ Snapshot logger with monitoring started - monitoring and snapshots every 60 seconds")
        
        # Wait a bit for the load test to start generating data
        time.sleep(30)
        
        start_time = time.time()
        
        print(f"Starting adaptation phase for {duration_seconds} seconds ({duration_seconds//60} minutes)...")
        print("📊 Monitoring and snapshot logging are running in background thread")
        print("🔄 The system will automatically detect and process anomalies")
        
        # Wait loop - monitoring is handled by AsyncSnapshotLogger, but we process queued anomalies
        while time.time() - start_time < duration_seconds:
            # Check for queued anomalies from monitoring thread
            if self.snapshot_logger:
                queued_anomalies = self.snapshot_logger.get_queued_anomalies()
                
                for anomaly_data in queued_anomalies:
                    try:
                        anomalies = anomaly_data['anomalies']
                        trace_df = anomaly_data['trace_df']
                        current_metrics = anomaly_data['current_metrics']
                        timestamp = anomaly_data['timestamp']
                        
                        print(f"🔄 Processing {len(anomalies)} queued anomalies from {timestamp}")
                        
                        # Process anomalies using the existing method
                        success = self.process_anomalies(anomalies, trace_df, current_metrics['metrics'])
                        
                        if success:
                            print("✅ Configuration changes applied successfully")
                        else:
                            print("❌ Failed to process anomalies")
                            
                    except Exception as e:
                        print(f"Error processing queued anomaly: {e}")
            
            # Wait a bit before checking again
            time.sleep(5)
            
            # Print periodic status updates
            elapsed = time.time() - start_time
            remaining = duration_seconds - elapsed
            print(f"⏱️ Adaptation phase: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
            
            # Check if snapshot logger is still running
            if self.snapshot_logger and not self.snapshot_logger.is_running:
                print("❌ Snapshot logger stopped unexpectedly")
                break
        
        # Stop the async load test
        print("Stopping continuous load test...")
        async_load_test.stop()
        
        # Stop the async snapshot logger
        if self.snapshot_logger:
            print("Stopping continuous snapshot logger...")
            self.snapshot_logger.stop()
            self.snapshot_logger = None
        
        print(f"Adaptation phase completed")
        
        # Generate plots at the end of adaptation phase
        try:
            from util.plotting import generate_adaptation_plots
            print(f"\n📊 Generating adaptation analysis plots...")
            plot_success = generate_adaptation_plots(self.label)
            if plot_success:
                print(f"✅ Adaptation analysis plots generated successfully!")
            else:
                print(f"⚠️  Could not generate adaptation plots")
        except Exception as e:
            print(f"❌ Error generating adaptation plots: {e}")
        
        # Generate configuration summary and evolution report
        try:
            print(f"\n📝 Generating configuration change summary...")
            self.config_logger.save_config_summary(self.label)
            print(f"✅ Configuration summary generated successfully!")
            
            print(f"\n📊 Generating comprehensive configuration evolution report...")
            self.config_logger.save_configuration_evolution_report(self.label)
            print(f"✅ Configuration evolution report generated successfully!")
        except Exception as e:
            print(f"❌ Error generating configuration reports: {e}")
        
        return True
    
    def process_anomalies(self, anomalies, trace_df, metric_dfs):
        """
        Process detected anomalies and generate configuration changes.
        Groups anomalies by service and processes each service only once per cycle.
        
        Args:
            anomalies: List of detected anomalies
            trace_df: Trace data
            metric_dfs: Metrics data
            
        Returns:
            bool: True if at least one anomaly was processed successfully
        """
        print(f"Processing {len(anomalies)} anomalies for configuration generation...")
        
        # Group anomalies by service (only process each service once per cycle)
        service_anomalies = {}
        for anom in anomalies:
            service_name, ai, duration = anom
            print(f"DEBUG: Processing anomaly - service_name: {service_name} (type: {type(service_name)}), ai: {ai} (type: {type(ai)}), duration: {duration} (type: {type(duration)})")
            if service_name not in service_anomalies:
                service_anomalies[service_name] = []
            service_anomalies[service_name].append((ai, duration))
        
        print(f"Grouped anomalies by service: {list(service_anomalies.keys())}")
        
        # Process each service only once
        successful_updates = []
        for service_name, service_anom_list in service_anomalies.items():
            # Use the most severe anomaly (longest duration) for this service
            most_severe_anom = max(service_anom_list, key=lambda x: x[1])
            ai, duration = most_severe_anom
            
            print(f"Processing most severe anomaly for service: {service_name} (duration: {duration})")
            
            # Debug: Check if service_name is valid
            if not isinstance(service_name, str) or not service_name:
                print(f"ERROR: Invalid service_name: {service_name} (type: {type(service_name)})")
                continue
                
            # Debug: Check if duration is valid
            if not isinstance(duration, (int, float)) and not hasattr(duration, 'total_seconds'):
                print(f"ERROR: Invalid duration: {duration} (type: {type(duration)})")
                continue

            # Load current configuration
            try:
                service_config_path = PATH + f"/output/{self.label}/config/{service_name}.yaml"
                auto_config_path = PATH + f"/output/{self.label}/config/config-autoscaler.yaml"
                
                print(f"Loading service config from: {service_config_path}")
                print(f"Loading auto config from: {auto_config_path}")
                
                service_config = load_yaml_as_dict(service_config_path)
                auto_config = load_yaml_as_dict(auto_config_path)

                kn_knobs = get_knative_knobs(service_config, auto_config)
                vscale_knobs = get_vscaling_knobs(service_config, auto_config)

                print("Current knobs:")
                pprint(kn_knobs)
                pprint(vscale_knobs)

                # Generate configuration for this anomaly using selected LLM provider
                if self.llm_provider == "openai":
                    success, configuration_update = generate_configuration_with_openai(
                        service_name, trace_df, metric_dfs, ai, anomalies, kn_knobs, self.label
                    )
                else:  # default to local LLM
                    success, configuration_update = generate_configuration_for_anomaly(
                        service_name, trace_df, metric_dfs, ai, anomalies, kn_knobs, self.label
                    )

                if success and configuration_update:
                    # Start tracking performance for this new configuration
                    config_name = f"{service_name}_config_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                    self.performance_logger.start_configuration_tracking(config_name)
                    self.performance_logger.set_current_phase("adaptation")
                    
                    # Store the configuration update for batch application
                    successful_updates.append((service_name, configuration_update))
                    print(f"Configuration generated for {service_name}")
                else:
                    print(f"No valid configuration update received for {service_name}")
                    
            except Exception as e:
                print(f"Error processing anomaly for {service_name}: {e}")
                print(f"  Service name: {service_name} (type: {type(service_name)})")
                print(f"  Duration: {duration} (type: {type(duration)})")
                print(f"  Anomaly index: {ai} (type: {type(ai)})")
                import traceback
                print(f"  Full traceback: {traceback.format_exc()}")
                continue
        
        # Apply all configuration changes in batch
        if successful_updates:
            print(f"Applying {len(successful_updates)} configuration updates...")
            
            # Set subphase to configuration application
            if self.snapshot_logger:
                self.snapshot_logger.set_subphase("configuration_application")
            
            success_count = 0
            
            for service_name, configuration_update in successful_updates:
                try:
                    # Get current configuration before applying changes
                    old_config = load_yaml_as_dict(PATH + f"/output/{self.label}/config/{service_name}.yaml")
                    
                    apply_success = apply_configuration_changes(service_name, configuration_update, self.label)
                    if apply_success:
                        success_count += 1
                        print(f"Configuration changes applied for {service_name}")
                        
                        # Get new configuration after applying changes
                        new_config = load_yaml_as_dict(PATH + f"/output/{self.label}/config/{service_name}.yaml")
                        
                        # Log the configuration change
                        self.config_logger.log_configuration_change(
                            service_name=service_name,
                            old_config=old_config,
                            new_config=new_config,
                            label=self.label,
                            reason="anomaly_detection",
                            performance_metrics=None  # Will be filled in later with snapshot data
                        )
                    else:
                        print(f"Failed to apply configuration changes for {service_name}")
                except Exception as e:
                    print(f"Error applying configuration for {service_name}: {e}")
            
            if success_count > 0:
                print(f"Successfully applied {success_count} configuration updates")
                
                # Set subphase to stabilization
                if self.snapshot_logger:
                    self.snapshot_logger.set_subphase("stabilization")
                
                # Start stabilization period tracking
                stabilization_duration = 240  # 4 minutes in seconds
                self.performance_logger.start_stabilization_period(stabilization_duration)
                print("🔄 Configuration changes applied - waiting 4 minutes for system stabilization...")
                
                # Wait 4 minutes for system stabilization
                stabilization_start = time.time()
                
                while time.time() - stabilization_start < stabilization_duration:
                    remaining = stabilization_duration - (time.time() - stabilization_start)
                    if remaining > 0:
                        print(f"⏳ Stabilization in progress... {remaining:.0f}s remaining")
                        time.sleep(30)  # Check every 30 seconds
                
                # End stabilization period tracking
                self.performance_logger.end_stabilization_period()
                
                # Clear subphase after stabilization
                if self.snapshot_logger:
                    self.snapshot_logger.set_subphase(None)
                
                print("✅ System stabilization period complete")
                
                # Perform lookback analysis over monitoring data
                print("📊 Performing lookback analysis over monitoring data...")
                lookback_success = self.perform_lookback_analysis()
                
                if lookback_success:
                    print("✅ Lookback analysis completed - system ready for next monitoring cycle")
                else:
                    print("⚠️ Lookback analysis had issues but continuing monitoring")
        
        return len(successful_updates) > 0
    
    def perform_lookback_analysis(self):
        """
        Perform lookback analysis over monitoring data after system stabilization.
        Analyzes the system behavior during and after configuration changes.
        
        Returns:
            bool: True if lookback analysis completed successfully
        """
        try:
            print("🔍 Starting lookback analysis over recent monitoring data...")
            
            # Collect current metrics to analyze the stabilization period
            continuous_label = f"{self.label}_continuous"
            current_metrics = self.continuous_monitor.collect_current_metrics(
                continuous_label, 
                interval_seconds=300  # Look back over 5 minutes (4 min stabilization + 1 min buffer)
            )
            
            if not current_metrics:
                print("❌ No monitoring data available for lookback analysis")
                return False
            
            print(f"📊 Collected {len(current_metrics.get('metrics', {}))} service metrics for analysis")
            
            # Analyze trace data if available
            if current_metrics.get('traces'):
                trace_df = convert_trace_data_to_dataframe(current_metrics['traces'])
                
                if not trace_df.empty:
                    print(f"📈 Analyzing {len(trace_df)} traces from stabilization period")
                    
                    # Detect anomalies in the stabilization period
                    stabilization_anomalies = self.anomaly_detector.detect_anomalies(
                        trace_df, 
                        exact_interval_seconds=300  # 5-minute lookback window
                    )
                    
                    if stabilization_anomalies:
                        print(f"⚠️ Found {len(stabilization_anomalies)} anomalies during stabilization period:")
                        for service_name, anomaly_index, duration in stabilization_anomalies:
                            print(f"  - {service_name}: {duration} (index: {anomaly_index})")
                        
                        # Log these anomalies for future reference
                        self.performance_logger.log_interval_metrics(
                            current_metrics, 
                            300, 
                            stabilization_anomalies
                        )
                        
                        print("📝 Stabilization anomalies logged for analysis")
                    else:
                        print("✅ No anomalies detected during stabilization period - system appears stable")
                    
                    # Analyze system performance trends
                    self._analyze_performance_trends(trace_df, current_metrics['metrics'])
                    
                else:
                    print("❌ No valid trace data for lookback analysis")
                    return False
            else:
                print("❌ No trace data available for lookback analysis")
                return False
            
            # Update configuration logs with stabilization period performance
            self._update_config_logs_with_stabilization_data(current_metrics)
            
            print("✅ Lookback analysis completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error during lookback analysis: {e}")
            return False
    
    def _analyze_performance_trends(self, trace_df, metrics_dfs):
        """
        Analyze performance trends during the stabilization period.
        
        Args:
            trace_df: Trace data from stabilization period
            metrics_dfs: Metrics data from stabilization period
        """
        try:
            print("📊 Analyzing performance trends during stabilization...")
            
            for service_name, service_metrics in metrics_dfs.items():
                if service_name in trace_df.columns:
                    # Get service-specific traces
                    service_traces = self._get_service_traces(trace_df, service_name)
                    
                    if not service_traces.empty:
                        # Calculate basic performance metrics
                        avg_response_time = service_traces['total'].mean()
                        max_response_time = service_traces['total'].max()
                        request_count = len(service_traces)
                        
                        print(f"  - {service_name}:")
                        print(f"    * Average response time: {avg_response_time:.2f}ms")
                        print(f"    * Max response time: {max_response_time:.2f}ms")
                        print(f"    * Request count: {request_count}")
                        
                        # Check for performance degradation
                        if avg_response_time > 1000:  # More than 1 second
                            print(f"    ⚠️ High average response time detected for {service_name}")
                        elif max_response_time > 5000:  # More than 5 seconds
                            print(f"    ⚠️ High max response time detected for {service_name}")
                        else:
                            print(f"    ✅ Performance appears stable for {service_name}")
            
        except Exception as e:
            print(f"Error analyzing performance trends: {e}")
    
    def _update_config_logs_with_stabilization_data(self, current_metrics):
        """
        Update configuration logs with stabilization period performance data.
        
        Args:
            current_metrics: Current metrics data from stabilization period
        """
        try:
            print("📝 Updating configuration logs with stabilization data...")
            
            # Get all recent configuration changes
            all_changes = self.config_logger.get_all_config_changes(self.label)
            
            # Update each service's latest configuration change with stabilization metrics
            for service_name in current_metrics.get('metrics', {}).keys():
                if service_name in all_changes and all_changes[service_name]:
                    # Get the most recent configuration change for this service
                    latest_change = all_changes[service_name][-1]
                    
                    # Check if this change was recent (within last 10 minutes)
                    change_time = datetime.datetime.fromisoformat(latest_change['timestamp'].replace('Z', '+00:00'))
                    current_time = datetime.datetime.utcnow().replace(tzinfo=change_time.tzinfo)
                    time_diff = (current_time - change_time).total_seconds()
                    
                    if time_diff < 600:  # Within 10 minutes
                        # Create stabilization metrics
                        stabilization_metrics = {
                            'stabilization_period_seconds': 240,  # 4 minutes
                            'lookback_analysis_timestamp': datetime.datetime.utcnow().isoformat(),
                            'service_metrics': current_metrics['metrics'].get(service_name, {}),
                            'analysis_type': 'post_configuration_stabilization'
                        }
                        
                        # Update the configuration log entry
                        self._update_config_log_entry(service_name, self.label, latest_change, stabilization_metrics)
                        print(f"  - Updated {service_name} configuration log with stabilization data")
            
        except Exception as e:
            print(f"Error updating configuration logs with stabilization data: {e}")
    
    def queue_metric_snapshot(self, label, current_metrics=None):
        """
        Queue a metric snapshot for asynchronous logging.
        
        Args:
            label: Label for the current test run
            current_metrics: Current metrics data (optional, will be collected if not provided)
        """
        try:
            if not self.snapshot_logger:
                print("No snapshot logger available, skipping snapshot")
                return
            
            # Collect current metrics if not provided
            if current_metrics is None:
                print(f"Collecting fresh metrics for snapshot (label: {label})")
                current_metrics = self.continuous_monitor.collect_current_metrics(label, interval_seconds=30)
            
            if not current_metrics:
                print("❌ No metrics data returned from monitoring collection")
                return
            
            if not current_metrics.get('metrics'):
                print("❌ No metrics found in collected data")
                print(f"   Available keys: {list(current_metrics.keys()) if current_metrics else 'None'}")
                return
            
            # Get trace data (optional for snapshot creation)
            trace_df = None
            if current_metrics.get('traces'):
                print(f"📊 Converting trace data to DataFrame...")
                trace_df = convert_trace_data_to_dataframe(current_metrics['traces'])
                print(f"   Converted {len(trace_df)} traces")
            else:
                print("⚠️ No trace data available, creating snapshot from metrics only")
                print(f"   Available keys: {list(current_metrics.keys())}")
            
            # Detect anomalies for each service individually
            all_anomalies = []
            total_anomaly_count = 0
            snapshots = []
            
            for service_name in current_metrics['metrics'].keys():
                try:
                    # Get service-specific trace data (if available)
                    if trace_df is not None:
                        service_traces = self._get_service_traces(trace_df, service_name)
                        
                        if not service_traces.empty:
                            # Detect anomalies for this specific service
                            service_anomalies = self.anomaly_detector.detect_anomalies(service_traces, exact_interval_seconds=30)
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
                        import pandas as pd
                        service_traces = pd.DataFrame()
                        service_anomalies = []
                        service_anomaly_count = 0
                    
                    # Create snapshot for this service
                    # Get current phase and subphase from snapshot logger
                    phase_info = self.snapshot_logger.get_phase_info() if self.snapshot_logger else {}
                    timestamp, duration, snapshot = metric_snapshot(
                        service_name, service_traces, current_metrics['metrics'],
                        phase=phase_info.get('phase'), subphase=phase_info.get('subphase')
                    )
                    
                    # Add anomaly information to the snapshot
                    snapshot['anomaly_count'] = service_anomaly_count
                    snapshot['anomalies'] = service_anomalies
                    
                    snapshots.append(snapshot)
                    print(f"  - {service_name}: {duration:.2f}ms avg response time, {snapshot['total_requests']} requests, {service_anomaly_count} anomalies")
                    
                except Exception as e:
                    print(f"  - Error taking snapshot for {service_name}: {e}")
            
            # Queue snapshot data for async logging
            if snapshots:
                # Get current phase and subphase from snapshot logger
                phase_info = self.snapshot_logger.get_phase_info() if self.snapshot_logger else {}
                
                snapshot_data = {
                    'timestamp': datetime.datetime.utcnow().isoformat(),
                    'label': label,
                    'type': 'metric_snapshot',
                    'phase': phase_info.get('phase'),
                    'subphase': phase_info.get('subphase'),
                    'snapshots': snapshots,
                    'service_count': len(snapshots),
                    'anomaly_count': total_anomaly_count,
                    'anomalies': all_anomalies
                }
                
                self.snapshot_logger.queue_snapshot(snapshot_data)
                print(f"📊 Metric snapshot queued for async logging ({len(snapshots)} services, {total_anomaly_count} anomalies)")
                
                # Update configuration logs with current performance metrics
                self._update_config_logs_with_performance(snapshots, label)
            
        except Exception as e:
            print(f"Error queuing metric snapshot: {e}")
    
    def _update_config_logs_with_performance(self, snapshots, label):
        """
        Update configuration logs with current performance metrics.
        
        Args:
            snapshots: List of service snapshots with performance data
            label: Test run label
        """
        try:
            # Get all configuration changes for this label
            all_changes = self.config_logger.get_all_config_changes(label)
            
            # Update each service's latest configuration change with performance metrics
            for snapshot in snapshots:
                service_name = snapshot['service_name']
                
                if service_name in all_changes and all_changes[service_name]:
                    # Get the most recent configuration change for this service
                    latest_change = all_changes[service_name][-1]
                    
                    # Check if this change doesn't already have performance metrics
                    if not latest_change.get('performance_metrics'):
                        # Create performance metrics from snapshot
                        performance_metrics = {
                            'response_time_avg': snapshot.get('response_time_avg', 0),
                            'response_time_p50': snapshot.get('response_time_p50', 0),
                            'response_time_p99': snapshot.get('response_time_p99', 0),
                            'response_time_max': snapshot.get('response_time_max', 0),
                            'total_requests': snapshot.get('total_requests', 0),
                            'error_count': snapshot.get('error_count', 0),
                            'error_rate': snapshot.get('error_rate', 0),
                            'anomaly_count': snapshot.get('anomaly_count', 0),
                            'service_metrics': snapshot.get('service_metrics', {}),
                            'snapshot_timestamp': snapshot.get('timestamp')
                        }
                        
                        # Update the configuration log entry
                        self._update_config_log_entry(service_name, label, latest_change, performance_metrics)
                        
        except Exception as e:
            print(f"Error updating configuration logs with performance: {e}")
    
    def _update_config_log_entry(self, service_name, label, log_entry, performance_metrics):
        """
        Update a specific configuration log entry with performance metrics.
        
        Args:
            service_name: Name of the service
            label: Test run label
            log_entry: The log entry to update
            performance_metrics: Performance metrics to add
        """
        try:
            import json
            import os
            
            log_file = os.path.join("output", label, "config_logs", f"{service_name}_config_history.jsonl")
            
            if not os.path.exists(log_file):
                return
            
            # Read all entries
            entries = []
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line.strip()))
            
            # Find and update the matching entry
            for i, entry in enumerate(entries):
                if (entry.get('timestamp') == log_entry.get('timestamp') and 
                    entry.get('service_name') == service_name):
                    entries[i]['performance_metrics'] = performance_metrics
                    break
            
            # Write back all entries
            with open(log_file, 'w') as f:
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')
                    
        except Exception as e:
            print(f"Error updating config log entry for {service_name}: {e}")
    
    def take_metric_snapshot(self, label):
        """
        Take a metric snapshot for all services (synchronous version for backward compatibility).
        
        Args:
            label: Label for the current test run
        """
        try:
            from util.analysis import metric_snapshot
            import json
            import os
            
            print(f"Taking metric snapshot at {datetime.datetime.now()}")
            
            # Collect current metrics
            current_metrics = self.continuous_monitor.collect_current_metrics(label, interval_seconds=30)
            
            if not current_metrics or not current_metrics.get('metrics'):
                print("No metrics available for snapshot")
                return
            
            # Get trace data
            if current_metrics.get('traces'):
                trace_df = convert_trace_data_to_dataframe(current_metrics['traces'])
            else:
                print("No trace data available for snapshot")
                return
            
            if trace_df.empty:
                print("Empty trace data for snapshot")
                return
            
            # Take snapshots for each service
            snapshots = []
            for service_name in current_metrics['metrics'].keys():
                try:
                    # Get current phase and subphase from snapshot logger
                    phase_info = self.snapshot_logger.get_phase_info() if self.snapshot_logger else {}
                    timestamp, duration, snapshot = metric_snapshot(
                        service_name, trace_df, current_metrics['metrics'],
                        phase=phase_info.get('phase'), subphase=phase_info.get('subphase')
                    )
                    snapshots.append(snapshot)
                    print(f"  - {service_name}: {duration:.2f}ms avg response time, {snapshot['total_requests']} requests")
                except Exception as e:
                    print(f"  - Error taking snapshot for {service_name}: {e}")
            
            # Save snapshots to file
            if snapshots:
                snapshot_data = {
                    'timestamp': datetime.datetime.utcnow().isoformat(),
                    'label': label,
                    'snapshots': snapshots
                }
                
                # Ensure directory exists
                snapshot_dir = f"{PATH}/output/{self.label}/snapshots"
                os.makedirs(snapshot_dir, exist_ok=True)
                
                # Save to file with timestamp
                timestamp_str = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                snapshot_file = f"{snapshot_dir}/metric_snapshot_{timestamp_str}.json"
                
                with open(snapshot_file, 'w') as f:
                    json.dump(snapshot_data, f, indent=2, default=str)
                
                print(f"Metric snapshot saved to: {snapshot_file}")
            
        except Exception as e:
            print(f"Error taking metric snapshot: {e}")
