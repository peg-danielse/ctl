#!/usr/bin/env python3
"""
Plotting library for analyzing snapshot data from THREADING_THE_NEEDLE experiments.
Loads snapshot.json files and creates various visualizations.
"""

import json
import os
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SnapshotPlotter:
    """Main class for plotting snapshot data with various visualization options."""
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the plotter.
        
        Args:
            output_dir: Base directory containing experiment outputs
        """
        self.output_dir = output_dir
        self.snapshots_data = {}
        
    def load_snapshots(self, label: str) -> List[Dict[str, Any]]:
        """
        Load snapshot data from a specific experiment label.
        
        Args:
            label: Experiment label (e.g., 'run', 'stopgap')
            
        Returns:
            List of snapshot dictionaries
        """
        snapshot_file = os.path.join(self.output_dir, label, "data", "snapshots.json")
        
        if not os.path.exists(snapshot_file):
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_file}")
            
        with open(snapshot_file, 'r') as f:
            snapshots = json.load(f)
            
        self.snapshots_data[label] = snapshots
        return snapshots
    
    def _add_stabilization_background(self, ax, timestamps: List[datetime], subphases: List[str]):
        """
        Add gray background highlighting for stabilization subphases.
        
        Args:
            ax: Matplotlib axis object
            timestamps: List of datetime objects
            subphases: List of subphase strings
            
        Returns:
            bool: True if stabilization periods were found and added
        """
        if len(timestamps) != len(subphases):
            return False
            
        # Find stabilization periods
        stabilization_periods = []
        current_start = None
        
        for i, subphase in enumerate(subphases):
            if subphase == "stabilization":
                if current_start is None:
                    current_start = timestamps[i]
            else:
                if current_start is not None:
                    stabilization_periods.append((current_start, timestamps[i]))
                    current_start = None
        
        # Handle case where stabilization continues to the end
        if current_start is not None:
            stabilization_periods.append((current_start, timestamps[-1]))
        
        # Add gray vertical spans for stabilization periods
        for i, (start_time, end_time) in enumerate(stabilization_periods):
            ax.axvspan(
                start_time, 
                end_time, 
                alpha=0.3, 
                color='gray', 
                zorder=0,
                label='Stabilization Phase' if i == 0 else ""  # Only label the first span
            )
        
        return len(stabilization_periods) > 0
    
    def plot_response_time_trends(self, label: str, save_path: Optional[str] = None) -> None:
        """
        Plot mean and P90 response times over time with stabilization background.
        
        Args:
            label: Experiment label
            save_path: Optional path to save the plot
        """
        if label not in self.snapshots_data:
            self.load_snapshots(label)
            
        snapshots = self.snapshots_data[label]
        
        # Extract data
        timestamps = [datetime.fromisoformat(s['timestamp'].replace('Z', '+00:00')) for s in snapshots]
        mean_response_times = [s['mean_response_time'] / 1000 for s in snapshots]  # Convert to ms
        p90_response_times = [s['p90_response_time'] / 1000 for s in snapshots]  # Convert to ms
        subphases = [s['subphase'] for s in snapshots]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

    

        # diffs = []

        # i = 1
        # while i < len(timestamps):
        #     diffs.append((timestamps[i] - timestamps[i-1]).total_seconds())
        #     i += 1

        # print(diffs)
        # print("max: ", max(diffs))
        # print("avg: ", sum(diffs) / len(diffs))
        
        # Plot response times first
        ax.plot(timestamps, mean_response_times, 'b-', linewidth=2, label='Mean Response Time', marker='o', markersize=4)
        ax.plot(timestamps, p90_response_times, 'r-', linewidth=2, label='P90 Response Time', marker='s', markersize=4)
        
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel('Response Time (ms)')
        ax.set_title(f'Response Time Trends - {label.title()}')
        ax.grid(True, alpha=0.3)
        
        # Add stabilization background AFTER plotting data (so y-limits are set correctly)
        has_stabilization = self._add_stabilization_background(ax, timestamps, subphases)
        
        # Add legend
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            os.makedirs(f"./output/{label}/plots", exist_ok=True)
            plt.savefig(f"./output/{label}/plots/response_time_trends_{label}.png", dpi=300, bbox_inches='tight')

        plt.close()
        plt.clf()
        # plt.show()
    
    def plot_deadline_miss_rate(self, label: str, save_path: Optional[str] = None) -> None:
        """
        Plot deadline miss rate over time with stabilization background.
        
        Args:
            label: Experiment label
            save_path: Optional path to save the plot
        """
        if label not in self.snapshots_data:
            self.load_snapshots(label)
            
        snapshots = self.snapshots_data[label]
        
        # Extract data
        timestamps = [datetime.fromisoformat(s['timestamp'].replace('Z', '+00:00')) for s in snapshots]
        deadline_miss_rates = [s['deadline_miss_rate'] for s in snapshots]
        subphases = [s['subphase'] for s in snapshots]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Plot deadline miss rate first
        ax.plot(timestamps, deadline_miss_rates, 'g-', linewidth=2, label='Deadline Miss Rate', marker='o', markersize=4)
        
        # Add horizontal line at 50% for reference
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% Reference')
        
        # Set y-limits before adding stabilization background
        ax.set_ylim(0, 100)
        
        # Add stabilization background AFTER setting y-limits
        has_stabilization = self._add_stabilization_background(ax, timestamps, subphases)
        
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel('Deadline Miss Rate (%)')
        ax.set_title(f'Deadline Miss Rate Over Time - {label.title()}')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            os.makedirs(f"./output/{label}/plots", exist_ok=True)
            plt.savefig(f"./output/{label}/plots/deadline_miss_rate_{label}.png", dpi=300, bbox_inches='tight')
        
        plt.close()
        plt.clf()
        # plt.show()
    
    def plot_anomaly_rate(self, label: str, save_path: Optional[str] = None) -> None:
        """
        Plot anomaly detection rate over time with stabilization background.
        
        Args:
            label: Experiment label
            save_path: Optional path to save the plot
        """
        if label not in self.snapshots_data:
            self.load_snapshots(label)
            
        snapshots = self.snapshots_data[label]
        

        print("snapshots: ", len(snapshots))
        # Extract data
        timestamps = [datetime.fromisoformat(s['timestamp'].replace('Z', '+00:00')) for s in snapshots]
        anomaly_rates = [s['anomaly_rate'] for s in snapshots]
        subphases = [s['subphase'] for s in snapshots]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Plot anomaly rate first
        ax.plot(timestamps, anomaly_rates, 'purple', linewidth=2, label='Anomaly Rate', marker='o', markersize=4)
        
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel('Anomaly Rate (%)')
        ax.set_title(f'Anomaly Detection Rate Over Time - {label.title()}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(anomaly_rates) * 1.1 if anomaly_rates else 100)
        
        # Add stabilization background AFTER plotting data (so y-limits are set correctly)
        has_stabilization = self._add_stabilization_background(ax, timestamps, subphases)
        
        # Add legend
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            os.makedirs(f"./output/{label}/plots", exist_ok=True)
            plt.savefig(f"./output/{label}/plots/anomaly_rate_{label}.png", dpi=300, bbox_inches='tight')
        
        plt.close()
        plt.clf()
        # plt.show()
    
    def plot_cpu_utilization(self, label: str, save_path: Optional[str] = None) -> None:
        """
        Plot total and per-service CPU utilization over time with stabilization background.
        
        Args:
            label: Experiment label
            save_path: Optional path to save the plot
        """
        if label not in self.snapshots_data:
            self.load_snapshots(label)
            
        snapshots = self.snapshots_data[label]
        
        # Extract data
        timestamps = [datetime.fromisoformat(s['timestamp'].replace('Z', '+00:00')) for s in snapshots]
        total_cpu = [s['sum_cpu_utilization'] for s in snapshots]
        subphases = [s['subphase'] for s in snapshots]
        
        # Get service names - collect from all snapshots to handle cases where services appear/disappear
        service_names = set()
        for s in snapshots:
            if 'services' in s:
                service_names.update(s['services'].keys())
        service_names = list(service_names)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('white')
        ax2.set_facecolor('white')
        
        # Plot 1: Total CPU utilization first
        ax1.plot(timestamps, total_cpu, 'b-', linewidth=2, label='Total CPU Utilization', marker='o', markersize=4)
        ax1.set_ylabel('Total CPU Utilization')
        ax1.set_title(f'CPU Utilization Over Time - {label.title()}')
        ax1.grid(True, alpha=0.3)
        
        # Let matplotlib set the y-limits automatically, then add stabilization background
        ax1.relim()  # Recalculate limits
        ax1.autoscale_view()  # Apply the limits
        
        # Add stabilization background AFTER setting y-limits
        has_stabilization1 = self._add_stabilization_background(ax1, timestamps, subphases)
        ax1.legend()
        
        # Plot 2: Per-service CPU utilization (stacked area)
        
        # Prepare data for stacked area plot
        service_cpu_data = {}
        for service in service_names:
            service_cpu_data[service] = []
            for s in snapshots:
                # Handle cases where service might not exist in all snapshots
                if 'services' in s and service in s['services'] and 'cpu_utilization' in s['services'][service]:
                    service_cpu_data[service].append(s['services'][service]['cpu_utilization'])
                else:
                    service_cpu_data[service].append(0)  # Default to 0 if data is missing
        
        # Create stacked area plot
        bottom = np.zeros(len(timestamps))
        colors = plt.cm.Set3(np.linspace(0, 1, len(service_names)))
        
        for i, (service, cpu_data) in enumerate(service_cpu_data.items()):
            ax2.fill_between(timestamps, bottom, bottom + cpu_data, 
                           label=service, alpha=0.7, color=colors[i])
            bottom += cpu_data
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('CPU Utilization per Service')
        ax2.set_title('Per-Service CPU Utilization (Stacked)')
        ax2.grid(True, alpha=0.3)
        
        # Let matplotlib set the y-limits automatically, then add stabilization background
        ax2.relim()  # Recalculate limits
        ax2.autoscale_view()  # Apply the limits
        
        # Add stabilization background AFTER setting y-limits
        has_stabilization2 = self._add_stabilization_background(ax2, timestamps, subphases)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format x-axis for both plots
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            os.makedirs(f"./output/{label}/plots", exist_ok=True)
            plt.savefig(f"./output/{label}/plots/cpu_utilization_{label}.png", dpi=300, bbox_inches='tight')
        
        plt.close()
        plt.clf()
        # plt.show()
    
    def plot_request_volume(self, label: str, save_path: Optional[str] = None) -> None:
        """
        Plot request volume over time with stabilization background.
        
        Args:
            label: Experiment label
            save_path: Optional path to save the plot
        """
        if label not in self.snapshots_data:
            self.load_snapshots(label)
            
        snapshots = self.snapshots_data[label]
        
        # Extract data
        timestamps = [datetime.fromisoformat(s['timestamp'].replace('Z', '+00:00')) for s in snapshots]
        total_requests = [s['total_requests'] for s in snapshots]
        subphases = [s['subphase'] for s in snapshots]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Plot request volume first
        ax.plot(timestamps, total_requests, 'orange', linewidth=2, label='Total Requests', marker='o', markersize=4)
        
        # Let matplotlib set the y-limits automatically, then add stabilization background
        ax.relim()  # Recalculate limits
        ax.autoscale_view()  # Apply the limits
        
        # Add stabilization background AFTER setting y-limits
        has_stabilization = self._add_stabilization_background(ax, timestamps, subphases)
        
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Requests')
        ax.set_title(f'Request Volume Over Time - {label.title()}')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            os.makedirs(f"./output/{label}/plots", exist_ok=True)
            plt.savefig(f"./output/{label}/plots/request_volume_{label}.png", dpi=300, bbox_inches='tight')
        
        plt.close()
        plt.clf()
        # plt.show()
    
    def plot_all_timeseries(self, label: str) -> None:
        """
        Generate all time series plots for a given experiment label.
        
        Args:
            label: Experiment label
        """
        print(f"Generating time series plots for {label}...")
        
        self.plot_response_time_trends(label)
        self.plot_deadline_miss_rate(label)
        self.plot_anomaly_rate(label)
        self.plot_cpu_utilization(label)
        self.plot_request_volume(label)
        
        print(f"All time series plots saved to ./output/{label}/plots/")


def main():
    import argparse

    plotter = SnapshotPlotter()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True)
    args = parser.parse_args()
    
    # Generate plots for both experiments
    try:
        label = args.label
        plotter.plot_all_timeseries(label)
    except FileNotFoundError as e:
        print(f"label not found: {e}")
    except Exception as e:
        print(f"Error generating plots for {label}: {e}")


if __name__ == "__main__":
    main()
