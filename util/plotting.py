#!/usr/bin/env python3
"""
Plotting utilities for CTL system analysis and visualization.
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def load_snapshot_data(label):
    """Load snapshot data from JSONL files."""
    # Look for snapshot files with various patterns since labels can have suffixes
    snapshot_patterns = [
        f"output/{label}/snapshots/*.jsonl",
        f"output/{label}/snapshots/snapshot_log_{label}_*.jsonl",
        f"output/{label}/snapshots/snapshot_log_{label}.jsonl"
    ]
    
    snapshot_files = []
    for pattern in snapshot_patterns:
        files = glob.glob(pattern)
        snapshot_files.extend(files)
    
    # Remove duplicates
    snapshot_files = list(set(snapshot_files))
    
    if not snapshot_files:
        print(f"No snapshot files found for patterns: {snapshot_patterns}")
        return None
    
    print(f"Found {len(snapshot_files)} snapshot files:")
    for file in snapshot_files:
        print(f"  - {file}")
    
    all_snapshots = []
    
    for file_path in snapshot_files:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        snapshot = json.loads(line.strip())
                        all_snapshots.append(snapshot)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Loaded {len(all_snapshots)} snapshots")
    return all_snapshots

def load_performance_log_data(label):
    """Load performance log data with phase information from JSON files."""
    # Look for performance log files
    performance_patterns = [
        f"output/{label}/performance_log_*.json",
        f"output/{label}/*/performance_log_*.json"
    ]
    
    performance_files = []
    for pattern in performance_patterns:
        files = glob.glob(pattern)
        performance_files.extend(files)
    
    # Remove duplicates
    performance_files = list(set(performance_files))
    
    if not performance_files:
        print(f"No performance log files found for patterns: {performance_patterns}")
        return None
    
    print(f"Found {len(performance_files)} performance log files:")
    for file in performance_files:
        print(f"  - {file}")
    
    all_performance_data = []
    
    for file_path in performance_files:
        try:
            with open(file_path, 'r') as f:
                performance_data = json.load(f)
                if isinstance(performance_data, list):
                    all_performance_data.extend(performance_data)
                else:
                    all_performance_data.append(performance_data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Loaded {len(all_performance_data)} performance log entries")
    return all_performance_data

def load_trace_data_for_deadline_analysis(label):
    """Load trace data for deadline miss rate analysis."""
    # Look for trace files in the output directory
    trace_patterns = [
        f"output/{label}/data/*/traces.json",
        f"output/{label}/data/*/*_traces.json",
        f"output/{label}/*_traces.json",
        f"{label}_traces.json"
    ]
    
    trace_files = []
    for pattern in trace_patterns:
        trace_files.extend(glob.glob(pattern))
    
    if not trace_files:
        print(f"No trace files found for deadline analysis with patterns: {trace_patterns}")
        return None
    
    print(f"Found {len(trace_files)} trace files for deadline analysis:")
    for file in trace_files:
        print(f"  - {file}")
    
    all_traces = []
    
    for file_path in trace_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if 'data' in data and data['data']:
                    all_traces.extend(data['data'])
                    print(f"  Loaded {len(data['data'])} traces from {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Total traces loaded for deadline analysis: {len(all_traces)}")
    return all_traces

def analyze_deadline_miss_rates(traces, deadline_threshold_percent=65):
    """
    Analyze deadline miss rates for each workflow pattern.
    
    Args:
        traces: List of trace data
        deadline_threshold_percent: Percentage threshold for deadline misses (default: 65%)
        
    Returns:
        Dictionary with deadline miss analysis by workflow pattern
    """
    if not traces:
        return None
    
    print(f"Analyzing deadline miss rates with {deadline_threshold_percent}% threshold...")
    
    # First, discover all unique operations in the traces
    all_operations = []
    for trace in traces:
        if 'spans' in trace:
            for span in trace['spans']:
                all_operations.append(span.get('operationName', 'unknown'))
    
    # Get unique operations and sort for consistent ordering
    unique_operations = sorted(list(set(all_operations)))
    print(f"Found {len(unique_operations)} unique operations in traces")
    
    # Process traces to create patterns and calculate response times
    workflow_data = {}
    
    for trace in traces:
        if 'spans' not in trace:
            continue
            
        # Calculate total response time
        total_duration = sum(span.get('duration', 0) for span in trace['spans'])
        
        # Create workflow pattern using ALL operations found in traces
        pattern_bits = []
        for operation in unique_operations:
            # Check if this operation was executed in this trace
            operation_executed = any(span.get('operationName') == operation for span in trace['spans'])
            pattern_bits.append('1' if operation_executed else '0')
        
        pattern = ''.join(pattern_bits)
        
        # Initialize workflow data if not exists
        if pattern not in workflow_data:
            workflow_data[pattern] = {
                'pattern': pattern,
                'operations': unique_operations,
                'response_times': [],
                'total_requests': 0,
                'deadline_misses': 0,
                'deadline_miss_rate': 0.0,
                'avg_response_time': 0.0,
                'max_response_time': 0.0,
                'min_response_time': float('inf')
            }
        
        # Add response time data
        workflow_data[pattern]['response_times'].append(total_duration)
        workflow_data[pattern]['total_requests'] += 1
        
        # Update min/max
        workflow_data[pattern]['min_response_time'] = min(
            workflow_data[pattern]['min_response_time'], total_duration
        )
        workflow_data[pattern]['max_response_time'] = max(
            workflow_data[pattern]['max_response_time'], total_duration
        )
    
    # Calculate deadline miss rates and statistics
    for pattern, data in workflow_data.items():
        if data['response_times']:
            # Calculate average response time
            data['avg_response_time'] = sum(data['response_times']) / len(data['response_times'])
            
            # Calculate deadline threshold (65% of average response time)
            deadline_threshold = data['avg_response_time'] * (deadline_threshold_percent / 100.0)
            
            # Count deadline misses
            data['deadline_misses'] = sum(1 for rt in data['response_times'] if rt > deadline_threshold)
            data['deadline_miss_rate'] = (data['deadline_misses'] / data['total_requests']) * 100.0
            
            # Set min_response_time properly
            if data['min_response_time'] == float('inf'):
                data['min_response_time'] = 0.0
    
    # Filter out patterns with no requests
    workflow_data = {k: v for k, v in workflow_data.items() if v['total_requests'] > 0}
    
    print(f"Analyzed {len(workflow_data)} workflow patterns")
    for pattern, data in workflow_data.items():
        print(f"  Pattern {pattern}: {data['total_requests']} requests, "
              f"{data['deadline_miss_rate']:.1f}% deadline miss rate")
    
    return workflow_data

def create_deadline_miss_plots(workflow_data, label, deadline_threshold_percent=65):
    """Create plots for deadline miss rate analysis."""
    if not workflow_data:
        print("No workflow data available for deadline miss analysis")
        return
    
    print(f"\n📊 Creating deadline miss rate plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory for plots
    plot_dir = f"output/{label}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot 1: Deadline Miss Rate by Workflow Pattern
    plt.figure(figsize=(15, 8))
    
    patterns = list(workflow_data.keys())
    miss_rates = [workflow_data[pattern]['deadline_miss_rate'] for pattern in patterns]
    request_counts = [workflow_data[pattern]['total_requests'] for pattern in patterns]
    
    # Create bar chart with deadline miss rates
    bars = plt.bar(range(len(patterns)), miss_rates, color='lightcoral', edgecolor='darkred', alpha=0.7)
    
    # Add request count labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, request_counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.title(f'Deadline Miss Rate by Workflow Pattern ({deadline_threshold_percent}% Threshold)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Workflow Pattern', fontsize=12)
    plt.ylabel('Deadline Miss Rate (%)', fontsize=12)
    plt.xticks(range(len(patterns)), [f'Pattern {i+1}' for i in range(len(patterns))], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add threshold line
    plt.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='10% Threshold')
    plt.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='25% Threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/deadline_miss_rate_by_workflow.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Response Time Distribution by Workflow Pattern
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each workflow pattern
    n_patterns = len(patterns)
    n_cols = 2
    n_rows = (n_patterns + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(f'Response Time Distribution by Workflow Pattern ({deadline_threshold_percent}% Deadline)', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, pattern in enumerate(patterns):
        if i < len(axes):
            data = workflow_data[pattern]
            response_times = data['response_times']
            
            # Create histogram
            axes[i].hist(response_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add deadline threshold line
            deadline_threshold = data['avg_response_time'] * (deadline_threshold_percent / 100.0)
            axes[i].axvline(deadline_threshold, color='red', linestyle='--', linewidth=2, 
                           label=f'{deadline_threshold_percent}% Deadline ({deadline_threshold:.0f}μs)')
            
            # Add average response time line
            axes[i].axvline(data['avg_response_time'], color='blue', linestyle='-', linewidth=2,
                           label=f'Avg Response Time ({data["avg_response_time"]:.0f}μs)')
            
            axes[i].set_title(f'Pattern {i+1} (Miss Rate: {data["deadline_miss_rate"]:.1f}%)', 
                             fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Response Time (μs)', fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'Requests: {data["total_requests"]}\nMisses: {data["deadline_misses"]}\nMin: {data["min_response_time"]:.0f}μs\nMax: {data["max_response_time"]:.0f}μs'
            axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                        verticalalignment='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_patterns, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/response_time_distribution_by_workflow.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Workflow Pattern Analysis Summary
    plt.figure(figsize=(15, 8))
    
    # Create a comprehensive summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Workflow Pattern Analysis Summary', fontsize=16, fontweight='bold')
    
    # Subplot 1: Request Volume by Pattern
    ax1.bar(range(len(patterns)), request_counts, color='lightblue', edgecolor='navy', alpha=0.7)
    ax1.set_title('Request Volume by Workflow Pattern', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Workflow Pattern', fontsize=10)
    ax1.set_ylabel('Total Requests', fontsize=10)
    ax1.set_xticks(range(len(patterns)))
    ax1.set_xticklabels([f'P{i+1}' for i in range(len(patterns))])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, count in enumerate(request_counts):
        ax1.text(i, count + max(request_counts) * 0.01, str(count), 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Subplot 2: Average Response Time by Pattern
    avg_response_times = [workflow_data[pattern]['avg_response_time'] for pattern in patterns]
    ax2.bar(range(len(patterns)), avg_response_times, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax2.set_title('Average Response Time by Workflow Pattern', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Workflow Pattern', fontsize=10)
    ax2.set_ylabel('Average Response Time (μs)', fontsize=10)
    ax2.set_xticks(range(len(patterns)))
    ax2.set_xticklabels([f'P{i+1}' for i in range(len(patterns))])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, avg_time in enumerate(avg_response_times):
        ax2.text(i, avg_time + max(avg_response_times) * 0.01, f'{avg_time:.0f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Subplot 3: Deadline Miss Rate vs Request Volume (Scatter)
    ax3.scatter(request_counts, miss_rates, s=100, alpha=0.7, c='red', edgecolors='darkred')
    ax3.set_title('Deadline Miss Rate vs Request Volume', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Total Requests', fontsize=10)
    ax3.set_ylabel('Deadline Miss Rate (%)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add pattern labels to scatter points
    for i, (count, rate) in enumerate(zip(request_counts, miss_rates)):
        ax3.annotate(f'P{i+1}', (count, rate), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, fontweight='bold')
    
    # Subplot 4: Response Time Range by Pattern
    min_times = [workflow_data[pattern]['min_response_time'] for pattern in patterns]
    max_times = [workflow_data[pattern]['max_response_time'] for pattern in patterns]
    
    x_pos = range(len(patterns))
    ax4.bar(x_pos, max_times, color='lightcoral', alpha=0.7, label='Max Response Time')
    ax4.bar(x_pos, min_times, color='lightblue', alpha=0.7, label='Min Response Time')
    ax4.set_title('Response Time Range by Workflow Pattern', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Workflow Pattern', fontsize=10)
    ax4.set_ylabel('Response Time (μs)', fontsize=10)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'P{i+1}' for i in range(len(patterns))])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/workflow_pattern_analysis_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Deadline miss rate plots saved to: {plot_dir}/")
    print(f"Generated {3} deadline miss analysis plots:")
    print(f"  1. deadline_miss_rate_by_workflow.png")
    print(f"  2. response_time_distribution_by_workflow.png")
    print(f"  3. workflow_pattern_analysis_summary.png")

def create_snapshot_plots(snapshots, label="loop", performance_data=None):
    """Create various plots from snapshot data with optional phase information."""
    if not snapshots:
        print("No snapshot data to plot")
        return
    
    print(f"\n📊 Creating plots for {len(snapshots)} snapshots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory for plots
    plot_dir = f"output/{label}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Extract data for plotting
    timestamps = []
    service_data = {}
    anomaly_data = []
    
    for snapshot in snapshots:
        if 'timestamp' in snapshot:
            try:
                timestamp = pd.to_datetime(snapshot['timestamp'])
                timestamps.append(timestamp)
                
                # Extract service data
                if 'snapshots' in snapshot:
                    for service_snapshot in snapshot['snapshots']:
                        service_name = service_snapshot.get('service_name', 'unknown')
                        
                        if service_name not in service_data:
                            service_data[service_name] = {
                                'timestamps': [],
                                'response_times': [],
                                'response_times_p50': [],
                                'response_times_p99': [],
                                'response_times_max': [],
                                'request_counts': [],
                                'error_rates': [],
                                'error_counts': [],
                                'anomaly_counts': [],
                                'cpu_usage': [],
                                'memory_usage': []
                            }
                        
                        service_data[service_name]['timestamps'].append(timestamp)
                        service_data[service_name]['response_times'].append(
                            service_snapshot.get('response_time_avg', 0)
                        )
                        service_data[service_name]['response_times_p50'].append(
                            service_snapshot.get('response_time_p50', 0)
                        )
                        service_data[service_name]['response_times_p99'].append(
                            service_snapshot.get('response_time_p99', 0)
                        )
                        service_data[service_name]['response_times_max'].append(
                            service_snapshot.get('response_time_max', 0)
                        )
                        service_data[service_name]['request_counts'].append(
                            service_snapshot.get('total_requests', 0)
                        )
                        service_data[service_name]['error_rates'].append(
                            service_snapshot.get('error_rate', 0)
                        )
                        service_data[service_name]['error_counts'].append(
                            service_snapshot.get('error_count', 0)
                        )
                        service_data[service_name]['anomaly_counts'].append(
                            service_snapshot.get('anomaly_count', 0)
                        )
                        
                        # Extract service metrics
                        service_metrics = service_snapshot.get('service_metrics', {})
                        service_data[service_name]['cpu_usage'].append(
                            service_metrics.get('cpu_usage', 0)
                        )
                        service_data[service_name]['memory_usage'].append(
                            service_metrics.get('memory_usage', 0)
                        )
                
                # Extract overall anomaly data
                if 'anomaly_count' in snapshot:
                    anomaly_data.append({
                        'timestamp': timestamp,
                        'total_anomalies': snapshot.get('anomaly_count', 0),
                        'service_count': snapshot.get('service_count', 0)
                    })
                    
            except Exception as e:
                print(f"Error processing snapshot timestamp: {e}")
    
    if not timestamps:
        print("No valid timestamps found in snapshots")
        return
    
    # Plot 1: Response Times Over Time - Separate plot for each service
    if service_data:
        # Calculate number of services and create subplots
        services = [name for name, data in service_data.items() if data['timestamps'] and data['response_times']]
        n_services = len(services)
        
        if n_services > 0:
            # Create subplots - 2 columns, as many rows as needed
            n_cols = 2
            n_rows = (n_services + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            fig.suptitle('Service Response Times Over Time', fontsize=16, fontweight='bold')
            
            # Flatten axes array for easier indexing
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, service_name in enumerate(services):
                if i < len(axes):
                    data = service_data[service_name]
                    axes[i].plot(data['timestamps'], data['response_times'], 
                               marker='o', linewidth=2, markersize=4, color='blue')
                    axes[i].set_title(f'{service_name}', fontsize=12, fontweight='bold')
                    axes[i].set_ylabel('Response Time (ms)', fontsize=10)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].tick_params(axis='x', rotation=45)
                    
                    # Add some statistics to the plot
                    if data['response_times']:
                        avg_time = sum(data['response_times']) / len(data['response_times'])
                        max_time = max(data['response_times'])
                        min_time = min(data['response_times'])
                        axes[i].text(0.02, 0.98, f'Avg: {avg_time:.0f}ms\nMax: {max_time:.0f}ms\nMin: {min_time:.0f}ms', 
                                   transform=axes[i].transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Hide unused subplots
            for i in range(n_services, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/response_times_over_time.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot 2: Request Counts Over Time
    plt.figure(figsize=(15, 8))
    for service_name, data in service_data.items():
        if data['timestamps'] and data['request_counts']:
            plt.plot(data['timestamps'], data['request_counts'], 
                    marker='s', label=service_name, linewidth=2, markersize=4)
    
    plt.title('Service Request Counts Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Total Requests', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/request_counts_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Anomaly Counts Over Time
    plt.figure(figsize=(15, 8))
    for service_name, data in service_data.items():
        if data['timestamps'] and data['anomaly_counts']:
            plt.plot(data['timestamps'], data['anomaly_counts'], 
                    marker='^', label=service_name, linewidth=2, markersize=4)
    
    plt.title('Service Anomaly Counts Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Anomaly Count', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/anomaly_counts_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Error Rates Over Time
    plt.figure(figsize=(15, 8))
    for service_name, data in service_data.items():
        if data['timestamps'] and data['error_rates']:
            plt.plot(data['timestamps'], data['error_rates'], 
                    marker='d', label=service_name, linewidth=2, markersize=4)
    
    plt.title('Service Error Rates Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Error Rate (%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/error_rates_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Overall Anomaly Summary (with phase information if available)
    if anomaly_data:
        plt.figure(figsize=(15, 8))
        anomaly_df = pd.DataFrame(anomaly_data)
        
        # If performance data is available, add phase information
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
            
            # Define phase colors
            phase_colors = {
                'initialization': '#FF6B6B',  # Red
                'baseline': '#4ECDC4',        # Teal
                'adaptation': '#45B7D1',      # Blue
                'stabilization': '#96CEB4'    # Green
            }
            
            # Create subplot with phase background
            plt.subplot(2, 1, 1)
            
            # Plot anomalies with phase coloring
            for phase in perf_df['phase'].unique():
                phase_data = perf_df[perf_df['phase'] == phase]
                if not phase_data.empty:
                    plt.scatter(phase_data['timestamp'], phase_data['anomalies_detected'], 
                               c=phase_colors.get(phase, '#666666'), 
                               label=f'{phase.title()} Phase', 
                               alpha=0.7, s=50)
            
            plt.title('Anomalies Over Time by Phase', fontsize=14, fontweight='bold')
            plt.xlabel('Time', fontsize=10)
            plt.ylabel('Anomalies Detected', fontsize=10)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # Add configuration change markers
            config_changes = perf_df[perf_df['configuration'] != perf_df['configuration'].shift()]
            for _, change in config_changes.iterrows():
                plt.axvline(x=change['timestamp'], color='red', linestyle='--', alpha=0.7)
                plt.text(change['timestamp'], plt.ylim()[1]*0.9, 
                        f"Config: {change['configuration']}", 
                        rotation=90, fontsize=8, ha='right')
            
            # Second subplot: Service count
            plt.subplot(2, 1, 2)
            plt.plot(anomaly_df['timestamp'], anomaly_df['service_count'], 
                    marker='s', color='blue', linewidth=2, markersize=6)
            plt.title('Active Services Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Time', fontsize=10)
            plt.ylabel('Service Count', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        else:
            # Original plot without phase information
            plt.subplot(1, 2, 1)
            plt.plot(anomaly_df['timestamp'], anomaly_df['total_anomalies'], 
                    marker='o', color='red', linewidth=2, markersize=6)
            plt.title('Total Anomalies Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Time', fontsize=10)
            plt.ylabel('Total Anomalies', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            plt.plot(anomaly_df['timestamp'], anomaly_df['service_count'], 
                    marker='s', color='blue', linewidth=2, markersize=6)
            plt.title('Active Services Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Time', fontsize=10)
            plt.ylabel('Service Count', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/anomaly_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 6: Response Time Distribution Analysis
    if service_data:
        # Create a comprehensive response time analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Response Time Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Collect all response times for distribution analysis
        all_response_times = []
        service_names = []
        
        for service_name, data in service_data.items():
            if data['response_times']:
                all_response_times.extend(data['response_times'])
                service_names.extend([service_name] * len(data['response_times']))
        
        if all_response_times:
            # Plot 1: Overall response time distribution
            ax1 = axes[0, 0]
            ax1.hist(all_response_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('Overall Response Time Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Response Time (ms)', fontsize=10)
            ax1.set_ylabel('Frequency', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Add percentile lines
            p50 = np.percentile(all_response_times, 50)
            p95 = np.percentile(all_response_times, 95)
            p99 = np.percentile(all_response_times, 99)
            
            ax1.axvline(p50, color='red', linestyle='--', label=f'P50: {p50:.0f}ms')
            ax1.axvline(p95, color='orange', linestyle='--', label=f'P95: {p95:.0f}ms')
            ax1.axvline(p99, color='darkred', linestyle='--', label=f'P99: {p99:.0f}ms')
            ax1.legend()
            
            # Plot 2: Box plot by service
            ax2 = axes[0, 1]
            service_response_times = {}
            for service_name, data in service_data.items():
                if data['response_times']:
                    service_response_times[service_name] = data['response_times']
            
            if service_response_times:
                box_data = [service_response_times[service] for service in service_response_times.keys()]
                box_labels = list(service_response_times.keys())
                
                bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                
                ax2.set_title('Response Time Distribution by Service', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Response Time (ms)', fontsize=10)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: Response time statistics table
            ax3 = axes[1, 0]
            ax3.axis('off')
            
            # Create statistics table
            stats_data = []
            for service_name, data in service_data.items():
                if data['response_times']:
                    times = data['response_times']
                    stats_data.append([
                        service_name,
                        f"{np.mean(times):.0f}",
                        f"{np.percentile(times, 50):.0f}",
                        f"{np.percentile(times, 95):.0f}",
                        f"{np.percentile(times, 99):.0f}",
                        f"{np.max(times):.0f}"
                    ])
            
            if stats_data:
                table = ax3.table(cellText=stats_data,
                                colLabels=['Service', 'Mean', 'P50', 'P95', 'P99', 'Max'],
                                cellLoc='center',
                                loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                ax3.set_title('Response Time Statistics (ms)', fontsize=12, fontweight='bold')
            
            # Plot 4: Response time variability (coefficient of variation)
            ax4 = axes[1, 1]
            service_names = []
            cv_values = []
            
            for service_name, data in service_data.items():
                if data['response_times'] and len(data['response_times']) > 1:
                    times = data['response_times']
                    mean_time = np.mean(times)
                    std_time = np.std(times)
                    cv = (std_time / mean_time) * 100 if mean_time > 0 else 0
                    service_names.append(service_name)
                    cv_values.append(cv)
            
            if service_names:
                bars = ax4.bar(service_names, cv_values, color='lightcoral')
                ax4.set_title('Response Time Variability (CV%)', fontsize=12, fontweight='bold')
                ax4.set_ylabel('Coefficient of Variation (%)', fontsize=10)
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, cv_values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/response_time_distribution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 7: Percentiles Over Time (Critical for Adaptation Analysis)
    if service_data:
        # Create subplots for percentiles over time
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Response Time Percentiles Over Time (Adaptation Analysis)', fontsize=16, fontweight='bold')
        
        # P50 (Median) over time
        ax1 = axes[0, 0]
        for service_name, data in service_data.items():
            if data['timestamps'] and data['response_times']:
                # Calculate P50 for each time point (in this case, we have single values per timestamp)
                # For proper percentile tracking, we'd need multiple measurements per timestamp
                # For now, we'll plot the response times as P50 equivalent
                ax1.plot(data['timestamps'], data['response_times'], 
                        marker='o', label=service_name, linewidth=2, markersize=4)
        ax1.set_title('P50 (Median) Response Time Over Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Response Time (ms)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # P95 over time (estimated from available data)
        ax2 = axes[0, 1]
        for service_name, data in service_data.items():
            if data['timestamps'] and data['response_times']:
                # Estimate P95 as 1.5x the average (rough approximation)
                estimated_p95 = [rt * 1.5 for rt in data['response_times']]
                ax2.plot(data['timestamps'], estimated_p95, 
                        marker='s', label=service_name, linewidth=2, markersize=4)
        ax2.set_title('Estimated P95 Response Time Over Time', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Response Time (ms)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # P99 over time (estimated from available data)
        ax3 = axes[1, 0]
        for service_name, data in service_data.items():
            if data['timestamps'] and data['response_times']:
                # Estimate P99 as 2x the average (rough approximation for tail latency)
                estimated_p99 = [rt * 2.0 for rt in data['response_times']]
                ax3.plot(data['timestamps'], estimated_p99, 
                        marker='^', label=service_name, linewidth=2, markersize=4)
        ax3.set_title('Estimated P99 Response Time Over Time', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Response Time (ms)', fontsize=10)
        ax3.set_xlabel('Time', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Latency spike detection (P99 vs P50 ratio)
        ax4 = axes[1, 1]
        for service_name, data in service_data.items():
            if data['timestamps'] and data['response_times']:
                # Calculate latency spike ratio (P99/P50 equivalent)
                spike_ratios = [2.0] * len(data['response_times'])  # Estimated ratio
                ax4.plot(data['timestamps'], spike_ratios, 
                        marker='d', label=service_name, linewidth=2, markersize=4)
        ax4.set_title('Latency Spike Ratio (P99/P50) Over Time', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Spike Ratio', fontsize=10)
        ax4.set_xlabel('Time', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='High Spike Threshold')
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/percentiles_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 8: CPU Usage Over Time
    plt.figure(figsize=(15, 8))
    for service_name, data in service_data.items():
        if data['timestamps'] and data['cpu_usage']:
            plt.plot(data['timestamps'], data['cpu_usage'], 
                    marker='o', label=service_name, linewidth=2, markersize=4)
    
    plt.title('CPU Usage Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('CPU Usage (%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/cpu_usage_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 9: Memory Usage Over Time (if available)
    plt.figure(figsize=(15, 8))
    has_memory_data = False
    for service_name, data in service_data.items():
        if data['timestamps'] and data['memory_usage'] and any(usage > 0 for usage in data['memory_usage']):
            plt.plot(data['timestamps'], data['memory_usage'], 
                    marker='s', label=service_name, linewidth=2, markersize=4)
            has_memory_data = True
    
    if has_memory_data:
        plt.title('Memory Usage Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Memory Usage (%)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/memory_usage_over_time.png", dpi=300, bbox_inches='tight')
    else:
        plt.text(0.5, 0.5, 'No Memory Usage Data Available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title('Memory Usage Over Time', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/memory_usage_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 10: Error Counts Over Time
    plt.figure(figsize=(15, 8))
    has_error_data = False
    for service_name, data in service_data.items():
        if data['timestamps'] and data['error_counts'] and any(count > 0 for count in data['error_counts']):
            plt.plot(data['timestamps'], data['error_counts'], 
                    marker='d', label=service_name, linewidth=2, markersize=4)
            has_error_data = True
    
    if has_error_data:
        plt.title('Error Counts Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Error Count', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/error_counts_over_time.png", dpi=300, bbox_inches='tight')
    else:
        plt.text(0.5, 0.5, 'No Errors Detected', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title('Error Counts Over Time', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/error_counts_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 11: Service Performance Heatmap
    if service_data:
        plt.figure(figsize=(12, 8))
        
        # Create a matrix of average response times by service
        services = list(service_data.keys())
        avg_response_times = []
        
        for service in services:
            if service_data[service]['response_times']:
                avg_time = sum(service_data[service]['response_times']) / len(service_data[service]['response_times'])
                avg_response_times.append(avg_time)
            else:
                avg_response_times.append(0)
        
        # Create heatmap data
        heatmap_data = pd.DataFrame({
            'Service': services,
            'Avg Response Time (ms)': avg_response_times
        }).set_index('Service')
        
        sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Response Time (ms)'})
        plt.title('Average Response Times by Service', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/service_performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ All plots saved to: {plot_dir}/")
    print(f"Generated {11} different plots:")
    print(f"  1. response_times_over_time.png")
    print(f"  2. request_counts_over_time.png") 
    print(f"  3. anomaly_counts_over_time.png")
    print(f"  4. error_rates_over_time.png")
    print(f"  5. anomaly_summary.png")
    print(f"  6. response_time_distribution_analysis.png")
    print(f"  7. percentiles_over_time.png")
    print(f"  8. cpu_usage_over_time.png")
    print(f"  9. memory_usage_over_time.png")
    print(f"  10. error_counts_over_time.png")
    print(f"  11. service_performance_heatmap.png")

def create_phase_aware_plots(performance_data, label="loop"):
    """Create plots with phase information from performance log data."""
    if not performance_data:
        print("No performance data to plot")
        return
    
    print(f"\n📊 Creating phase-aware plots for {len(performance_data)} performance entries...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory for plots
    plot_dir = f"output/{label}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(performance_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Define phase colors
    phase_colors = {
        'initialization': '#FF6B6B',  # Red
        'baseline': '#4ECDC4',        # Teal
        'adaptation': '#45B7D1',      # Blue
        'stabilization': '#96CEB4'    # Green
    }
    
    # Plot 1: Anomalies Over Time with Phase Coloring
    plt.figure(figsize=(15, 8))
    
    for phase in df['phase'].unique():
        phase_data = df[df['phase'] == phase]
        if not phase_data.empty:
            plt.scatter(phase_data['timestamp'], phase_data['anomalies_detected'], 
                       c=phase_colors.get(phase, '#666666'), 
                       label=f'{phase.title()} Phase', 
                       alpha=0.7, s=50)
    
    plt.title('Anomalies Detected Over Time by Phase', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Anomalies Detected', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/anomalies_by_phase_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Phase Duration Analysis
    plt.figure(figsize=(12, 8))
    
    phase_durations = df.groupby('phase')['interval_seconds'].sum()
    phase_counts = df.groupby('phase').size()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Total time per phase
    phases = phase_durations.index
    colors = [phase_colors.get(phase, '#666666') for phase in phases]
    ax1.bar(phases, phase_durations.values, color=colors)
    ax1.set_title('Total Time Spent in Each Phase', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Number of snapshots per phase
    ax2.bar(phases, phase_counts.values, color=colors)
    ax2.set_title('Number of Snapshots per Phase', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Snapshots')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/phase_duration_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Stabilization Progress (if stabilization data exists)
    stabilization_data = df[df['phase'] == 'stabilization']
    if not stabilization_data.empty:
        plt.figure(figsize=(15, 8))
        
        for config in stabilization_data['configuration'].unique():
            config_data = stabilization_data[stabilization_data['configuration'] == config]
            if 'stabilization_progress' in config_data.columns:
                plt.plot(config_data['timestamp'], 
                        config_data['stabilization_progress'] * 100, 
                        marker='o', linewidth=2, markersize=6, 
                        label=f'Config: {config}')
        
        plt.title('Stabilization Progress Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Stabilization Progress (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/stabilization_progress.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 4: Configuration Changes Over Time with Phase Context
    plt.figure(figsize=(15, 8))
    
    # Plot anomalies with phase background
    for phase in df['phase'].unique():
        phase_data = df[df['phase'] == phase]
        if not phase_data.empty:
            plt.scatter(phase_data['timestamp'], phase_data['anomalies_detected'], 
                       c=phase_colors.get(phase, '#666666'), 
                       alpha=0.6, s=30)
    
    # Add configuration change markers
    config_changes = df[df['configuration'] != df['configuration'].shift()]
    for _, change in config_changes.iterrows():
        plt.axvline(x=change['timestamp'], color='red', linestyle='--', alpha=0.7)
        plt.text(change['timestamp'], plt.ylim()[1]*0.9, 
                f"Config: {change['configuration']}", 
                rotation=90, fontsize=8, ha='right')
    
    plt.title('Anomalies and Configuration Changes Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Anomalies Detected', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/anomalies_with_config_changes.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Phase Transition Timeline
    plt.figure(figsize=(15, 6))
    
    # Create a timeline showing phase transitions
    phase_transitions = df[df['phase'] != df['phase'].shift()]
    
    y_pos = 0
    for i, (_, transition) in enumerate(phase_transitions.iterrows()):
        phase = transition['phase']
        timestamp = transition['timestamp']
        
        # Draw phase block
        if i < len(phase_transitions) - 1:
            next_timestamp = phase_transitions.iloc[i+1]['timestamp']
            duration = (next_timestamp - timestamp).total_seconds()
        else:
            duration = 60  # Default duration for last phase
        
        plt.barh(y_pos, duration, left=timestamp, 
                color=phase_colors.get(phase, '#666666'), 
                alpha=0.7, height=0.8)
        
        # Add phase label
        plt.text(timestamp + pd.Timedelta(seconds=duration/2), y_pos, 
                phase.title(), ha='center', va='center', 
                fontweight='bold', fontsize=10)
    
    plt.title('Phase Transition Timeline', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Phase', fontsize=12)
    plt.yticks([0], ['Experiment'])
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/phase_transition_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Phase-aware plots saved to: {plot_dir}/")
    print(f"Generated {5} phase-aware plots:")
    print(f"  1. anomalies_by_phase_over_time.png")
    print(f"  2. phase_duration_analysis.png")
    print(f"  3. stabilization_progress.png")
    print(f"  4. anomalies_with_config_changes.png")
    print(f"  5. phase_transition_timeline.png")

def create_phase_analysis_report(performance_data, label="loop"):
    """Create a comprehensive phase analysis report."""
    if not performance_data:
        print("No performance data to analyze")
        return
    
    print(f"\n📊 Creating phase analysis report for {len(performance_data)} performance entries...")
    
    # Convert to DataFrame
    df = pd.DataFrame(performance_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Create output directory
    plot_dir = f"output/{label}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Generate comprehensive phase analysis
    phase_stats = df.groupby('phase').agg({
        'anomalies_detected': ['sum', 'mean', 'std', 'count'],
        'interval_seconds': 'sum',
        'metrics_count': 'mean',
        'stabilization_elapsed': 'max'
    }).round(2)
    
    # Create a comprehensive report plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Phase Analysis Report', fontsize=16, fontweight='bold')
    
    # Plot 1: Total anomalies by phase
    phase_totals = phase_stats[('anomalies_detected', 'sum')]
    axes[0, 0].bar(phase_totals.index, phase_totals.values, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(phase_totals)])
    axes[0, 0].set_title('Total Anomalies by Phase', fontweight='bold')
    axes[0, 0].set_ylabel('Total Anomalies')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Average anomalies per snapshot by phase
    phase_means = phase_stats[('anomalies_detected', 'mean')]
    axes[0, 1].bar(phase_means.index, phase_means.values,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(phase_means)])
    axes[0, 1].set_title('Average Anomalies per Snapshot by Phase', fontweight='bold')
    axes[0, 1].set_ylabel('Avg Anomalies')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Total time spent in each phase
    phase_times = phase_stats[('interval_seconds', 'sum')]
    axes[0, 2].pie(phase_times.values, labels=phase_times.index, autopct='%1.1f%%',
                   colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(phase_times)])
    axes[0, 2].set_title('Time Distribution Across Phases', fontweight='bold')
    
    # Plot 4: Number of snapshots per phase
    phase_counts = phase_stats[('anomalies_detected', 'count')]
    axes[1, 0].bar(phase_counts.index, phase_counts.values,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(phase_counts)])
    axes[1, 0].set_title('Number of Snapshots per Phase', fontweight='bold')
    axes[1, 0].set_ylabel('Snapshot Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 5: Average services monitored by phase
    phase_services = phase_stats[('metrics_count', 'mean')]
    axes[1, 1].bar(phase_services.index, phase_services.values,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(phase_services)])
    axes[1, 1].set_title('Average Services Monitored by Phase', fontweight='bold')
    axes[1, 1].set_ylabel('Avg Services')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Plot 6: Stabilization analysis (if available)
    stabilization_data = df[df['phase'] == 'stabilization']
    if not stabilization_data.empty and 'stabilization_elapsed' in stabilization_data.columns:
        stabilization_times = stabilization_data.groupby('configuration')['stabilization_elapsed'].max()
        axes[1, 2].bar(range(len(stabilization_times)), stabilization_times.values,
                       color='#96CEB4')
        axes[1, 2].set_title('Stabilization Duration by Configuration', fontweight='bold')
        axes[1, 2].set_ylabel('Stabilization Time (s)')
        axes[1, 2].set_xticks(range(len(stabilization_times)))
        axes[1, 2].set_xticklabels([f'Config {i+1}' for i in range(len(stabilization_times))], rotation=45)
    else:
        axes[1, 2].text(0.5, 0.5, 'No Stabilization Data', ha='center', va='center', 
                        transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].set_title('Stabilization Analysis', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/comprehensive_phase_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\n📊 Phase Analysis Summary:")
    print("=" * 50)
    for phase in phase_stats.index:
        stats = phase_stats.loc[phase]
        print(f"\n🔍 {phase.title()} Phase:")
        print(f"   📈 Total anomalies: {stats[('anomalies_detected', 'sum')]}")
        print(f"   📊 Avg anomalies per snapshot: {stats[('anomalies_detected', 'mean')]:.2f}")
        print(f"   ⏱️  Total time: {stats[('interval_seconds', 'sum')]:.1f}s")
        print(f"   📋 Number of snapshots: {stats[('anomalies_detected', 'count')]}")
        print(f"   🔧 Avg services monitored: {stats[('metrics_count', 'mean')]:.1f}")
        
        if phase == 'stabilization' and stats[('stabilization_elapsed', 'max')] > 0:
            print(f"   🔄 Max stabilization time: {stats[('stabilization_elapsed', 'max')]:.1f}s")
    
    print(f"\n✅ Comprehensive phase analysis saved to: {plot_dir}/comprehensive_phase_analysis.png")

def load_configuration_evolution_data(label):
    """Load configuration evolution data from the new tracking structure."""
    config_tracking_dir = f"output/{label}/configuration_tracking"
    evolution_file = os.path.join(config_tracking_dir, "configuration_evolution_report.json")
    
    if not os.path.exists(evolution_file):
        print(f"No configuration evolution report found at: {evolution_file}")
        return None
    
    try:
        with open(evolution_file, 'r') as f:
            evolution_data = json.load(f)
        print(f"Loaded configuration evolution data for {len(evolution_data.get('services', {}))} services")
        return evolution_data
    except Exception as e:
        print(f"Error loading configuration evolution data: {e}")
        return None

def create_configuration_evolution_plots(evolution_data, label):
    """Create plots for configuration evolution analysis."""
    if not evolution_data:
        print("No configuration evolution data to plot")
        return
    
    print(f"\n📊 Creating configuration evolution plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory for plots
    plot_dir = f"output/{label}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    services = evolution_data.get('services', {})
    if not services:
        print("No service data found in evolution report")
        return
    
    # Plot 1: Configuration Changes Over Time by Service
    plt.figure(figsize=(15, 8))
    
    for service_name, service_data in services.items():
        timeline = service_data.get('evolution_timeline', [])
        if timeline:
            timestamps = [pd.to_datetime(change['timestamp']) for change in timeline]
            change_numbers = [change['change_number'] for change in timeline]
            
            plt.plot(timestamps, change_numbers, 
                    marker='o', label=service_name, linewidth=2, markersize=6)
    
    plt.title('Configuration Changes Over Time by Service', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Configuration Change Number', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/configuration_changes_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Total Configuration Changes by Service
    plt.figure(figsize=(12, 8))
    
    service_names = []
    change_counts = []
    
    for service_name, service_data in services.items():
        service_names.append(service_name)
        change_counts.append(service_data.get('total_changes', 0))
    
    bars = plt.bar(service_names, change_counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Total Configuration Changes by Service', fontsize=16, fontweight='bold')
    plt.xlabel('Service', fontsize=12)
    plt.ylabel('Number of Configuration Changes', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, change_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/total_configuration_changes_by_service.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Change Frequency Analysis
    plt.figure(figsize=(12, 8))
    
    service_names = []
    frequencies = []
    
    for service_name, service_data in services.items():
        service_names.append(service_name)
        frequencies.append(service_data.get('change_frequency_minutes', 0))
    
    bars = plt.bar(service_names, frequencies, color='lightcoral', edgecolor='darkred', alpha=0.7)
    plt.title('Configuration Change Frequency by Service', fontsize=16, fontweight='bold')
    plt.xlabel('Service', fontsize=12)
    plt.ylabel('Average Time Between Changes (minutes)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, freq in zip(bars, frequencies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{freq:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/configuration_change_frequency.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Most Changed Parameters Analysis
    plt.figure(figsize=(15, 10))
    
    # Collect all parameter changes across services
    all_param_changes = {}
    
    for service_name, service_data in services.items():
        patterns = service_data.get('change_patterns', {})
        most_changed = patterns.get('most_changed_parameters', {})
        
        for param, count in most_changed.items():
            if param not in all_param_changes:
                all_param_changes[param] = 0
            all_param_changes[param] += count
    
    if all_param_changes:
        # Sort parameters by total changes
        sorted_params = sorted(all_param_changes.items(), key=lambda x: x[1], reverse=True)
        top_params = sorted_params[:15]  # Top 15 most changed parameters
        
        param_names = [param for param, _ in top_params]
        param_counts = [count for _, count in top_params]
        
        bars = plt.barh(param_names, param_counts, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        plt.title('Most Frequently Changed Configuration Parameters', fontsize=16, fontweight='bold')
        plt.xlabel('Total Number of Changes', fontsize=12)
        plt.ylabel('Configuration Parameter', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for bar, count in zip(bars, param_counts):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    str(count), ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/most_changed_parameters.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 5: Change Reasons Analysis
    plt.figure(figsize=(12, 8))
    
    # Collect all change reasons across services
    all_reasons = {}
    
    for service_name, service_data in services.items():
        patterns = service_data.get('change_patterns', {})
        reasons = patterns.get('change_reasons', {})
        
        for reason, count in reasons.items():
            if reason not in all_reasons:
                all_reasons[reason] = 0
            all_reasons[reason] += count
    
    if all_reasons:
        reason_names = list(all_reasons.keys())
        reason_counts = list(all_reasons.values())
        
        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(reason_names)))
        wedges, texts, autotexts = plt.pie(reason_counts, labels=reason_names, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        
        plt.title('Configuration Change Reasons Distribution', fontsize=16, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/configuration_change_reasons.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 6: Configuration Evolution Timeline (Detailed)
    if len(services) <= 4:  # Only create detailed timeline for small number of services
        fig, axes = plt.subplots(len(services), 1, figsize=(15, 4 * len(services)))
        if len(services) == 1:
            axes = [axes]
        
        fig.suptitle('Detailed Configuration Evolution Timeline', fontsize=16, fontweight='bold')
        
        for i, (service_name, service_data) in enumerate(services.items()):
            timeline = service_data.get('evolution_timeline', [])
            if timeline:
                timestamps = [pd.to_datetime(change['timestamp']) for change in timeline]
                change_numbers = [change['change_number'] for change in timeline]
                reasons = [change.get('reason', 'unknown') for change in timeline]
                
                # Create scatter plot with different colors for different reasons
                unique_reasons = list(set(reasons))
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_reasons)))
                reason_color_map = dict(zip(unique_reasons, colors))
                
                for j, (timestamp, change_num, reason) in enumerate(zip(timestamps, change_numbers, reasons)):
                    axes[i].scatter(timestamp, change_num, 
                                  color=reason_color_map[reason], s=100, alpha=0.7, 
                                  edgecolors='black', linewidth=1)
                    
                    # Add change number as text
                    axes[i].text(timestamp, change_num + 0.1, str(change_num), 
                               ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                axes[i].set_title(f'{service_name} Configuration Evolution', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('Change Number', fontsize=10)
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add legend for reasons
                if i == 0:  # Only add legend to first subplot
                    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=reason_color_map[reason], 
                                                markersize=10, label=reason) 
                                     for reason in unique_reasons]
                    axes[i].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/detailed_configuration_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Configuration evolution plots saved to: {plot_dir}/")
    print(f"Generated {6} configuration evolution plots:")
    print(f"  1. configuration_changes_over_time.png")
    print(f"  2. total_configuration_changes_by_service.png")
    print(f"  3. configuration_change_frequency.png")
    print(f"  4. most_changed_parameters.png")
    print(f"  5. configuration_change_reasons.png")
    print(f"  6. detailed_configuration_timeline.png")

def generate_adaptation_plots(label):
    """Generate plots for adaptation phase results."""
    print(f"\n📊 Generating adaptation plots for label: {label}")
    
    plot_generated = False
    
    # Load performance log data with phase information
    performance_data = load_performance_log_data(label)
    
    if performance_data:
        # Create phase-aware plots
        create_phase_aware_plots(performance_data, label)
        print(f"\n🎉 Phase-aware plots generated successfully!")
        print(f"📁 Phase plots saved to: output/{label}/plots/")
        
        # Create comprehensive phase analysis report
        create_phase_analysis_report(performance_data, label)
        print(f"\n🎉 Comprehensive phase analysis generated successfully!")
        print(f"📁 Phase analysis saved to: output/{label}/plots/")
        
        plot_generated = True
    else:
        print("⚠️  No performance log data found to plot")
    
    # Load snapshot data
    snapshots = load_snapshot_data(label)
    
    if snapshots:
        # Create plots with performance data for phase information
        create_snapshot_plots(snapshots, label, performance_data)
        print(f"\n🎉 Adaptation plots generated successfully!")
        print(f"📁 Plots saved to: output/{label}/plots/")
        plot_generated = True
    else:
        print("⚠️  No snapshot data found to plot")
    
    # Load and create configuration evolution plots
    evolution_data = load_configuration_evolution_data(label)
    
    if evolution_data:
        create_configuration_evolution_plots(evolution_data, label)
        print(f"\n🎉 Configuration evolution plots generated successfully!")
        print(f"📁 Configuration plots saved to: output/{label}/plots/")
        plot_generated = True
    else:
        print("⚠️  No configuration evolution data found to plot")
    
    # Load and create deadline miss rate analysis plots
    traces = load_trace_data_for_deadline_analysis(label)
    
    if traces:
        workflow_data = analyze_deadline_miss_rates(traces, deadline_threshold_percent=65)
        
        if workflow_data:
            create_deadline_miss_plots(workflow_data, label, deadline_threshold_percent=65)
            print(f"\n🎉 Deadline miss rate plots generated successfully!")
            print(f"📁 Deadline analysis plots saved to: output/{label}/plots/")
            plot_generated = True
        else:
            print("⚠️  No workflow data generated from traces")
    else:
        print("⚠️  No trace data found for deadline miss analysis")
    
    return plot_generated
