#!/usr/bin/env python3
"""
Plotting library for analyzing snapshot data from THREADING_THE_NEEDLE experiments.
Loads snapshot.json files and creates various visualizations.
"""

import json
import os
import glob
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns
import yaml

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SnapshotPlotter:
    """Main class for plotting snapshot data with various visualization options."""
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the plotter.
        
        Args:
            output_dir: Base directory containing experiment outputs (made absolute so it works from any cwd)
        """
        self.output_dir = os.path.abspath(output_dir)
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

    def _get_snapshot_time(self, snapshot: Dict[str, Any]) -> datetime:
        time_str = snapshot.get("collection_end_time") or snapshot.get("timestamp")
        if not time_str:
            raise ValueError("Snapshot is missing a timestamp")
        return datetime.fromisoformat(time_str.replace('Z', '+00:00'))

    def _get_sorted_snapshots(self, label: str) -> List[Dict[str, Any]]:
        if label not in self.snapshots_data:
            self.load_snapshots(label)
        snapshots = self.snapshots_data[label]
        return sorted(snapshots, key=self._get_snapshot_time)
    
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

    def _get_adaptation_phase_timeline(self, label: str) -> Optional[Tuple[List[datetime], List[str]]]:
        """
        Get (timestamps, subphases) for snapshots in the adaptation phase only.
        Used to correlate configuration applied_at with phase/subphase (e.g. stabilization).
        Returns None if snapshots are missing or no adaptation phase present.
        """
        try:
            snapshots = self._get_sorted_snapshots(label)
        except Exception:
            return None
        if not snapshots:
            return None
        timestamps = []
        subphases = []
        for s in snapshots:
            if s.get('phase') != 'adaptation':
                continue
            try:
                t = self._get_snapshot_time(s)
                timestamps.append(t)
                subphases.append(s.get('subphase', ''))
            except (ValueError, KeyError):
                continue
        if not timestamps:
            return None
        return (timestamps, subphases)

    def _format_time_axis(self, ax, timestamps: List[datetime], rotation: int = 45) -> None:
        """
        Set x-axis locator and formatter based on time span so long experiments
        (e.g. 6 hours) don't get too many tick labels.
        """
        if not timestamps or len(timestamps) < 2:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
            return
        t_min = min(timestamps)
        t_max = max(timestamps)
        span_seconds = (t_max - t_min).total_seconds()
        span_hours = span_seconds / 3600.0
        # Target ~6–12 major ticks
        if span_hours <= 0.5:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        elif span_hours <= 1.0:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif span_hours <= 2.0:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif span_hours <= 6.0:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif span_hours <= 24.0:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(span_hours / 12))))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)

    def plot_response_time_trends(self, label: str, save_path: Optional[str] = None) -> None:
        """
        Plot mean and P90 response times over time with stabilization background.
        
        Args:
            label: Experiment label
            save_path: Optional path to save the plot
        """
        snapshots = self._get_sorted_snapshots(label)
        
        # Extract data
        timestamps = [self._get_snapshot_time(s) for s in snapshots]
        mean_response_times = [s['mean_response_time'] / 1000 for s in snapshots]  # Convert to ms
        p90_response_times = [s['p90_response_time'] / 1000 for s in snapshots]  # Convert to ms
        subphases = [s['subphase'] for s in snapshots]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
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
        
        # Format x-axis (adapts to time span to avoid crowded ticks on long runs)
        self._format_time_axis(ax, timestamps)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            os.makedirs(os.path.join(self.output_dir, label, "plots"), exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, label, "plots", f"response_time_trends_{label}.png"), dpi=300, bbox_inches='tight')

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
        snapshots = self._get_sorted_snapshots(label)
        
        # Extract data
        timestamps = [self._get_snapshot_time(s) for s in snapshots]
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
        
        # Format x-axis (adapts to time span to avoid crowded ticks on long runs)
        self._format_time_axis(ax, timestamps)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            os.makedirs(os.path.join(self.output_dir, label, "plots"), exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, label, "plots", f"deadline_miss_rate_{label}.png"), dpi=300, bbox_inches='tight')
        
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
        snapshots = self._get_sorted_snapshots(label)
        

        # Extract data
        timestamps = [self._get_snapshot_time(s) for s in snapshots]
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
        
        # Format x-axis (adapts to time span to avoid crowded ticks on long runs)
        self._format_time_axis(ax, timestamps)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            os.makedirs(os.path.join(self.output_dir, label, "plots"), exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, label, "plots", f"anomaly_rate_{label}.png"), dpi=300, bbox_inches='tight')
        
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
        snapshots = self._get_sorted_snapshots(label)
        
        # Extract data
        timestamps = [self._get_snapshot_time(s) for s in snapshots]
        total_cpu = [
            s.get('sum_cpu_utilization_cores', s.get('sum_cpu_utilization', 0))
            for s in snapshots
        ]
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
        ax1.plot(timestamps, total_cpu, 'b-', linewidth=2, label='Total CPU Utilization (cores)', marker='o', markersize=4)
        ax1.set_ylabel('Total CPU Utilization (cores)')
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
                if 'services' in s and service in s['services']:
                    service_cpu_data[service].append(
                        s['services'][service].get('cpu_utilization_cores', s['services'][service].get('cpu_utilization', 0))
                    )
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
        ax2.set_ylabel('CPU Utilization per Service (cores)')
        ax2.set_title('Per-Service CPU Utilization (Stacked, cores)')
        ax2.grid(True, alpha=0.3)
        
        # Let matplotlib set the y-limits automatically, then add stabilization background
        ax2.relim()  # Recalculate limits
        ax2.autoscale_view()  # Apply the limits
        
        # Add stabilization background AFTER setting y-limits
        has_stabilization2 = self._add_stabilization_background(ax2, timestamps, subphases)
        handles2, labels2 = ax2.get_legend_handles_labels()
        if handles2:
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format x-axis for both plots (adapts to time span)
        for ax in [ax1, ax2]:
            self._format_time_axis(ax, timestamps)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            os.makedirs(os.path.join(self.output_dir, label, "plots"), exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, label, "plots", f"cpu_utilization_{label}.png"), dpi=300, bbox_inches='tight')
        
        plt.close()
        plt.clf()
        # plt.show()

    def plot_memory_utilization(self, label: str, save_path: Optional[str] = None) -> None:
        """
        Plot total and per-service memory utilization over time with stabilization background.
        
        Args:
            label: Experiment label
            save_path: Optional path to save the plot
        """
        snapshots = self._get_sorted_snapshots(label)
        
        # Extract data
        timestamps = [self._get_snapshot_time(s) for s in snapshots]
        total_memory = [
            s.get('sum_memory_utilization_mb', s.get('sum_memory_utilization', 0))
            for s in snapshots
        ]
        subphases = [s['subphase'] for s in snapshots]
        
        # Get service names - collect from all snapshots to handle cases where services appear/disappear
        service_names = set()
        for s in snapshots:
            if 'services' in s:
                service_names.update(s['services'].keys())
        # Order so memcached and database services appear first in the legend
        def _memcache_db_order(name: str) -> tuple:
            if 'memcached' in name.lower():
                return (0, name)
            if 'mongo' in name.lower() or 'database' in name.lower() or 'db' in name.lower():
                return (1, name)
            return (2, name)
        service_names = sorted(service_names, key=_memcache_db_order)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('white')
        ax2.set_facecolor('white')
        
        # Plot 1: Total memory utilization first
        ax1.plot(timestamps, total_memory, 'g-', linewidth=2, label='Total Memory Utilization (MB)', marker='o', markersize=4)
        ax1.set_ylabel('Total Memory Utilization (MB)')
        ax1.set_title(f'Memory Utilization Over Time - {label.title()}')
        ax1.grid(True, alpha=0.3)
        
        # Let matplotlib set the y-limits automatically, then add stabilization background
        ax1.relim()  # Recalculate limits
        ax1.autoscale_view()  # Apply the limits
        
        # Add stabilization background AFTER setting y-limits
        has_stabilization1 = self._add_stabilization_background(ax1, timestamps, subphases)
        ax1.legend()
        
        # Plot 2: Per-service memory utilization (stacked area)
        service_memory_data = {}
        for service in service_names:
            service_memory_data[service] = []
            for s in snapshots:
                if 'services' in s and service in s['services']:
                    service_memory_data[service].append(
                        s['services'][service].get('memory_utilization_mb', s['services'][service].get('memory_utilization', 0))
                    )
                else:
                    service_memory_data[service].append(0)  # Default to 0 if data is missing
        
        # Create stacked area plot
        bottom = np.zeros(len(timestamps))
        colors = plt.cm.Set3(np.linspace(0, 1, len(service_names)))
        
        for i, (service, mem_data) in enumerate(service_memory_data.items()):
            ax2.fill_between(timestamps, bottom, bottom + mem_data, 
                           label=service, alpha=0.7, color=colors[i])
            bottom += mem_data
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Memory Utilization per Service (MB)')
        ax2.set_title('Per-Service Memory Utilization (Stacked, MB)')
        ax2.grid(True, alpha=0.3)
        
        # Let matplotlib set the y-limits automatically, then add stabilization background
        ax2.relim()  # Recalculate limits
        ax2.autoscale_view()  # Apply the limits
        
        # Add stabilization background AFTER setting y-limits
        has_stabilization2 = self._add_stabilization_background(ax2, timestamps, subphases)
        handles2, labels2 = ax2.get_legend_handles_labels()
        if handles2:
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format x-axis for both plots (adapts to time span)
        for ax in [ax1, ax2]:
            self._format_time_axis(ax, timestamps)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            os.makedirs(os.path.join(self.output_dir, label, "plots"), exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, label, "plots", f"memory_utilization_{label}.png"), dpi=300, bbox_inches='tight')
        
        plt.close()
        plt.clf()
        # plt.show()

    def _sanitize_filename(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

    def plot_node_resources(self, label: str) -> None:
        """
        Plot per-node CPU/memory usage with limits.
        Creates one figure per node with two subplots (CPU, Memory).
        """
        snapshots = self._get_sorted_snapshots(label)
        timestamps = [self._get_snapshot_time(s) for s in snapshots]
        subphases = [s['subphase'] for s in snapshots]

        node_names = set()
        for s in snapshots:
            node_names.update((s.get('nodes') or {}).keys())
        if not node_names:
            return

        for node in sorted(node_names):
            cpu_usage = []
            cpu_limit = []
            mem_usage = []
            mem_limit = []
            for s in snapshots:
                node_data = (s.get('nodes') or {}).get(node, {})
                cpu_usage.append(node_data.get('cpu_usage_cores', 0))
                cpu_limit.append(node_data.get('cpu_limit_cores', 0))
                mem_usage.append(node_data.get('memory_usage_mb', 0))
                mem_limit.append(node_data.get('memory_limit_mb', 0))

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.patch.set_facecolor('white')
            ax1.set_facecolor('white')
            ax2.set_facecolor('white')

            ax1.plot(timestamps, cpu_usage, 'b-', linewidth=2, label='CPU Usage (cores)', marker='o', markersize=3)
            ax1.plot(timestamps, cpu_limit, 'b--', linewidth=1.5, label='CPU Limit (cores)')
            ax1.set_ylabel('CPU (cores)')
            ax1.set_title(f'Node CPU Usage vs Limit - {node}')
            ax1.grid(True, alpha=0.3)
            ax1.relim()
            ax1.autoscale_view()
            self._add_stabilization_background(ax1, timestamps, subphases)
            ax1.legend()

            ax2.plot(timestamps, mem_usage, 'g-', linewidth=2, label='Memory Usage (MB)', marker='o', markersize=3)
            ax2.plot(timestamps, mem_limit, 'g--', linewidth=1.5, label='Memory Limit (MB)')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Memory (MB)')
            ax2.set_title(f'Node Memory Usage vs Limit - {node}')
            ax2.grid(True, alpha=0.3)
            ax2.relim()
            ax2.autoscale_view()
            self._add_stabilization_background(ax2, timestamps, subphases)
            ax2.legend()

            for ax in [ax1, ax2]:
                self._format_time_axis(ax, timestamps)

            plt.tight_layout()
            os.makedirs(os.path.join(self.output_dir, label, "plots"), exist_ok=True)
            safe_node = self._sanitize_filename(node)
            plt.savefig(os.path.join(self.output_dir, label, "plots", f"node_resources_{label}_{safe_node}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            plt.clf()

    def plot_node_services(self, label: str) -> None:
        """
        Plot per-node service pod counts over time (stacked area).
        Creates one figure per node.
        """
        snapshots = self._get_sorted_snapshots(label)
        timestamps = [self._get_snapshot_time(s) for s in snapshots]
        subphases = [s['subphase'] for s in snapshots]

        node_names = set()
        for s in snapshots:
            node_names.update((s.get('nodes') or {}).keys())
        if not node_names:
            return

        for node in sorted(node_names):
            service_names = set()
            for s in snapshots:
                node_services = (s.get('nodes') or {}).get(node, {}).get('services', {})
                service_names.update(node_services.keys())
            if not service_names:
                continue

            service_names = list(service_names)
            service_data = {service: [] for service in service_names}
            for s in snapshots:
                node_services = (s.get('nodes') or {}).get(node, {}).get('services', {})
                for service in service_names:
                    service_data[service].append(node_services.get(service, 0))

            fig, ax = plt.subplots(figsize=(12, 7))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            bottom = np.zeros(len(timestamps))
            colors = plt.cm.Set3(np.linspace(0, 1, len(service_names)))
            for i, service in enumerate(service_names):
                ax.fill_between(timestamps, bottom, bottom + service_data[service],
                                label=service, alpha=0.7, color=colors[i])
                bottom += service_data[service]

            ax.set_xlabel('Time')
            ax.set_ylabel('Average Pods per Service')
            ax.set_title(f'Node Services (Pod Counts) - {node}')
            ax.grid(True, alpha=0.3)
            ax.relim()
            ax.autoscale_view()
            if bottom.size > 0:
                max_total = float(np.max(bottom))
                ax.set_ylim(0, max_total * 1.1 if max_total > 0 else 1)
            self._add_stabilization_background(ax, timestamps, subphases)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            self._format_time_axis(ax, timestamps)

            plt.tight_layout()
            os.makedirs(os.path.join(self.output_dir, label, "plots"), exist_ok=True)
            safe_node = self._sanitize_filename(node)
            plt.savefig(os.path.join(self.output_dir, label, "plots", f"node_services_{label}_{safe_node}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            plt.clf()

    def plot_affinity_over_time(self, label: str) -> None:
        """
        Heatmap: service@node placement over time (pod count per snapshot).
        Shows whether affinity/placement changes over the run.
        """
        snapshots = self._get_sorted_snapshots(label)
        timestamps = [self._get_snapshot_time(s) for s in snapshots]
        if not timestamps:
            return

        # Collect all (service, node) pairs that appear in any snapshot
        placement_pairs: List[Tuple[str, str]] = []
        seen = set()
        for s in snapshots:
            for node_name, info in (s.get('nodes') or {}).items():
                for svc in (info.get('services') or {}).keys():
                    key = (svc, node_name)
                    if key not in seen:
                        seen.add(key)
                        placement_pairs.append(key)

        if not placement_pairs:
            return

        # Build matrix: rows = placement pairs, cols = time (snapshot index)
        n_pairs = len(placement_pairs)
        n_snap = len(snapshots)
        matrix = np.zeros((n_pairs, n_snap))
        for j, s in enumerate(snapshots):
            nodes = s.get('nodes') or {}
            for i, (svc, node_name) in enumerate(placement_pairs):
                matrix[i, j] = ((nodes.get(node_name) or {}).get('services') or {}).get(svc, 0)

        y_labels = [f"{svc} @ {node}" for svc, node in placement_pairs]

        fig, ax = plt.subplots(figsize=(max(10, n_snap * 0.4), max(6, n_pairs * 0.35)))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        im = ax.imshow(matrix, aspect='auto', cmap='Blues', interpolation='nearest', vmin=0)
        ax.set_yticks(np.arange(n_pairs))
        ax.set_yticklabels(y_labels, fontsize=8)
        # Limit x-axis labels for long runs (e.g. 6 hours) to avoid crowding
        max_xticks = 12
        if n_snap <= max_xticks:
            tick_indices = np.arange(n_snap)
        else:
            step = max(1, (n_snap - 1) // (max_xticks - 1))
            tick_indices = np.arange(0, n_snap, step)
            if tick_indices[-1] != n_snap - 1:
                tick_indices = np.append(tick_indices, n_snap - 1)
        span_seconds = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0
        time_fmt = '%m/%d %H:%M' if span_seconds > 86400 else '%H:%M'
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([timestamps[i].strftime(time_fmt) for i in tick_indices], rotation=45, ha='right')
        ax.set_xlabel('Time')
        ax.set_ylabel('Service @ Node (affinity)')
        ax.set_title(f'Node affinity over time (pod count) - {label}')
        plt.colorbar(im, ax=ax, label='Pods')
        plt.tight_layout()
        os.makedirs(os.path.join(self.output_dir, label, "plots"), exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, label, "plots", f"affinity_over_time_{label}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

    def plot_affinity_summary(self, label: str) -> None:
        """
        Heatmap: nodes x services for latest snapshot (current placement).
        Quick view of which services run on which nodes.
        """
        snapshots = self._get_sorted_snapshots(label)
        if not snapshots:
            return
        latest = snapshots[-1]
        nodes_data = latest.get('nodes') or {}
        if not nodes_data:
            return

        node_names = sorted(nodes_data.keys())
        service_names = set()
        for info in nodes_data.values():
            service_names.update((info.get('services') or {}).keys())
        service_names = sorted(service_names)
        if not service_names:
            return

        matrix = np.zeros((len(node_names), len(service_names)))
        for i, node in enumerate(node_names):
            svcs = (nodes_data.get(node) or {}).get('services') or {}
            for j, svc in enumerate(service_names):
                matrix[i, j] = svcs.get(svc, 0)

        fig, ax = plt.subplots(figsize=(max(8, len(service_names) * 0.8), max(5, len(node_names) * 0.5)))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        im = ax.imshow(matrix, aspect='auto', cmap='Blues', interpolation='nearest', vmin=0)
        ax.set_xticks(np.arange(len(service_names)))
        ax.set_xticklabels(service_names, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(node_names)))
        ax.set_yticklabels(node_names)
        ax.set_xlabel('Service')
        ax.set_ylabel('Node')
        ax.set_title(f'Node affinity (current placement, pod count) - {label}')
        plt.colorbar(im, ax=ax, label='Pods')
        plt.tight_layout()
        os.makedirs(os.path.join(self.output_dir, label, "plots"), exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, label, "plots", f"affinity_summary_{label}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

    def plot_request_volume(self, label: str, save_path: Optional[str] = None) -> None:
        """
        Plot request volume over time with stabilization background.
        
        Args:
            label: Experiment label
            save_path: Optional path to save the plot
        """
        snapshots = self._get_sorted_snapshots(label)
        
        # Extract data
        timestamps = [self._get_snapshot_time(s) for s in snapshots]
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
        
        # Format x-axis (adapts to time span)
        self._format_time_axis(ax, timestamps)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            os.makedirs(os.path.join(self.output_dir, label, "plots"), exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, label, "plots", f"request_volume_{label}.png"), dpi=300, bbox_inches='tight')
        
        plt.close()
        plt.clf()
        # plt.show()
    
    def plot_all_timeseries(self, label: str) -> None:
        """
        Generate all time series plots for a given experiment label.
        
        Args:
            label: Experiment label
        """
        plt.close('all')  # release any existing figures before generating many plots
        prev_max_open = plt.rcParams.get('figure.max_open_warning', 20)
        plt.rcParams['figure.max_open_warning'] = 100  # we close after each save; avoid warning during batch
        try:
            plots_dir = os.path.join(self.output_dir, label, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            self.plot_response_time_trends(label)
            self.plot_deadline_miss_rate(label)
            self.plot_anomaly_rate(label)
            self.plot_cpu_utilization(label)
            self.plot_memory_utilization(label)
            self.plot_request_volume(label)
            self.plot_node_resources(label)
            self.plot_node_services(label)
            self.plot_affinity_over_time(label)
            self.plot_affinity_summary(label)
            self.plot_configuration_overhead(label)
            try:
                # self.plot_configuration_changes(label)
                pass
            except Exception as e:
                import traceback
                traceback.print_exc()
        finally:
            plt.rcParams['figure.max_open_warning'] = prev_max_open

    def plot_configuration_overhead(self, label: str) -> None:
        """
        Plot the effect of the treatment (config changes) on performance and faults.
        Effect = difference (after - before) per change: for each change k, before_k vs after_k
        windows, then mean(after_k) - mean(before_k) for each metric.
        All data from the current run. Produces:
        - Violin plot: distribution of treatment effect across changes, per metric.
        - Line plot: effect vs change index, one line per metric.
        
        Args:
            label: Experiment label (adaptation snapshots loaded from this run only)
        """
        # Load snapshots from current run only
        try:
            adaptation_snapshots = self._get_sorted_snapshots(label)
        except FileNotFoundError:
            return

        # Filter to adaptation phase only; keep time order (already sorted by _get_sorted_snapshots)
        adaptation_only = [s for s in adaptation_snapshots if s.get('phase') == 'adaptation']
        if not adaptation_only:
            return

        # Build during-blocks by *sequence order* (subphase), not timestamp, so windows are unambiguous.
        # Each block = consecutive indices with subphase in (stabilization, configuration_application).
        during_block_ranges: List[Tuple[int, int]] = []  # (start_idx_incl, end_idx_incl) in adaptation_only
        i = 0
        while i < len(adaptation_only):
            subphase = adaptation_only[i].get('subphase', '')
            if subphase in ['stabilization', 'configuration_application']:
                start = i
                while i < len(adaptation_only) and adaptation_only[i].get('subphase', '') in ['stabilization', 'configuration_application']:
                    i += 1
                during_block_ranges.append((start, i - 1))
            else:
                i += 1

        if not during_block_ranges:
            return

        # Per-change before/after windows by *index* in adaptation_only (no timestamp ambiguity).
        # before_k = normal snapshots immediately before block k (the "state before change k").
        # after_k  = normal snapshots immediately after block k (the "state after change k").
        def get_metric_means(snapshots: List[Dict[str, Any]]) -> Dict[str, float]:
            if not snapshots:
                return {}
            keys_and_getters = [
                ('mean_response_time', lambda s: s.get('mean_response_time', 0) / 1000.0),
                ('p90_response_time', lambda s: s.get('p90_response_time', 0) / 1000.0),
                ('cpu_utilization', lambda s: s.get('sum_cpu_utilization_cores', 0) * 1000),  # cores → millicores
                ('memory_utilization', lambda s: s.get('sum_memory_utilization_mb', 0)),
                ('anomaly_rate', lambda s: s.get('anomaly_rate', 0)),
                ('deadline_miss_rate', lambda s: s.get('deadline_miss_rate', 0)),
            ]
            out = {}
            for key, getter in keys_and_getters:
                vals = [getter(s) for s in snapshots]
                out[key] = sum(vals) / len(vals) if vals else 0.0
            return out

        effect_by_metric: Dict[str, List[float]] = {
            'mean_response_time': [], 'p90_response_time': [], 'cpu_utilization': [],
            'memory_utilization': [], 'anomaly_rate': [], 'deadline_miss_rate': [],
        }
        toxicity_by_metric: Dict[str, List[float]] = {
            'mean_response_time': [], 'p90_response_time': [], 'cpu_utilization': [],
            'memory_utilization': [], 'anomaly_rate': [], 'deadline_miss_rate': [],
        }
        # Optional: collect per-change means for verification CSV
        verification_rows: List[Dict[str, Any]] = []
        for k in range(len(during_block_ranges)):
            start_k, end_k = during_block_ranges[k]
            during_k = [adaptation_only[j] for j in range(start_k, end_k + 1)]
            # before_k = all normal snapshots *before* this block (indices 0 .. start_k-1 for k=0; gap between block k-1 and block k for k>=1)
            if k == 0:
                before_start, before_end = 0, start_k - 1  # [0, start_k-1] inclusive
            else:
                _, end_prev = during_block_ranges[k - 1]
                before_start, before_end = end_prev + 1, start_k - 1  # (end_prev, start_k) exclusive of blocks
            # after_k = all normal snapshots *after* this block until next block
            if k + 1 < len(during_block_ranges):
                start_next, _ = during_block_ranges[k + 1]
                after_start, after_end = end_k + 1, start_next - 1  # (end_k, start_next) exclusive
            else:
                after_start, after_end = end_k + 1, len(adaptation_only) - 1

            if before_start > before_end or after_start > after_end:
                continue
            before_k = [adaptation_only[j] for j in range(before_start, before_end + 1)]
            after_k = [adaptation_only[j] for j in range(after_start, after_end + 1)]
            if not before_k or not after_k:
                continue
            mean_before = get_metric_means(before_k)
            mean_after = get_metric_means(after_k)
            mean_during = get_metric_means(during_k)
            mean_before_after = {key: (mean_before[key] + mean_after[key]) / 2.0 for key in effect_by_metric}
            # Effect = (after - before): positive = metric increased after this change (e.g. anomaly got worse)
            for key in effect_by_metric:
                effect_by_metric[key].append(mean_after[key] - mean_before[key])
            # Toxicity = during - mean(before, after): how much worse (or better) the "during" window is vs the average of before and after
            for key in toxicity_by_metric:
                toxicity_by_metric[key].append(mean_during[key] - mean_before_after[key])
            verification_rows.append({
                'change_index': k + 1,
                'before_snapshots': len(before_k),
                'during_snapshots': len(during_k),
                'after_snapshots': len(after_k),
                **{f'before_{key}': mean_before[key] for key in effect_by_metric},
                **{f'during_{key}': mean_during[key] for key in effect_by_metric},
                **{f'after_{key}': mean_after[key] for key in effect_by_metric},
                **{f'effect_{key}': mean_after[key] - mean_before[key] for key in effect_by_metric},
                **{f'toxicity_{key}': mean_during[key] - mean_before_after[key] for key in toxicity_by_metric},
            })

        # Drop metrics that have no effects (e.g. no change had both before and after)
        effect_by_metric = {k: v for k, v in effect_by_metric.items() if v}
        toxicity_by_metric = {k: v for k, v in toxicity_by_metric.items() if v}
        if not effect_by_metric:
            return

        plots_dir = os.path.join(self.output_dir, label, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        # Write verification CSV (per-change before/during/after, effect, toxicity)
        if verification_rows:
            import csv
            out_csv = os.path.join(plots_dir, f"config_treatment_effect_verification_{label}.csv")
            with open(out_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=verification_rows[0].keys())
                writer.writeheader()
                writer.writerows(verification_rows)
        self._plot_treatment_effect_violin(label, effect_by_metric, plots_dir)
        self._plot_treatment_effect_line(label, effect_by_metric, plots_dir)
        self._plot_treatment_effect_line_zscore(label, effect_by_metric, plots_dir)
        if toxicity_by_metric:
            self._plot_toxicity_violin(label, toxicity_by_metric, plots_dir)
            self._plot_toxicity_line(label, toxicity_by_metric, plots_dir)
            self._plot_toxicity_line_zscore(label, toxicity_by_metric, plots_dir)
    
    def _get_config_application_timestamps(self, label: str) -> List[datetime]:
        """Get timestamps when configurations were applied from .meta.json files."""
        config_dir = os.path.join(self.output_dir, label, "config")
        if not os.path.exists(config_dir):
            return []
        
        timestamps = []
        meta_files = glob.glob(os.path.join(config_dir, "*.meta.json"))
        for meta_file in meta_files:
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                ts_str = meta.get("applied_at")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    timestamps.append(ts)
            except Exception:
                continue
        return sorted(timestamps)

    def _treatment_effect_metric_names(self) -> Dict[str, str]:
        """Display names and units for treatment effect (difference = after - before)."""
        return {
            'mean_response_time': 'Mean Response Time (ms)',
            'p90_response_time': 'P90 Response Time (ms)',
            'cpu_utilization': 'CPU Utilization (millicores)',
            'memory_utilization': 'Memory Utilization (MB)',
            'anomaly_rate': 'Anomaly Rate (%)',
            'deadline_miss_rate': 'Deadline Miss Rate (%)',
        }

    def _plot_treatment_effect_violin(self, label: str, effect_by_metric: Dict[str, List[float]], plots_dir: str) -> None:
        """Violin plot: distribution of treatment effect (after - before) per metric. Dual y-axis: left = response time, CPU, memory; right = anomaly %, deadline miss %."""
        names = self._treatment_effect_metric_names()
        left_metrics = ['mean_response_time', 'p90_response_time', 'cpu_utilization', 'memory_utilization']
        right_metrics = ['anomaly_rate', 'deadline_miss_rate']
        left_order = [k for k in left_metrics if k in effect_by_metric]
        right_order = [k for k in right_metrics if k in effect_by_metric]
        if not left_order and not right_order:
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax2 = ax.twinx()
        positions = []
        all_labels = []
        if left_order:
            left_data = [effect_by_metric[k] for k in left_order]
            left_pos = list(range(len(left_order)))
            positions.extend(left_pos)
            all_labels.extend([names[k] for k in left_order])
            parts_left = ax.violinplot(left_data, positions=left_pos, showmeans=True, showmedians=True)
            for pc in parts_left['bodies']:
                pc.set_facecolor('#3498db')
                pc.set_alpha(0.7)
        if right_order:
            right_data = [effect_by_metric[k] for k in right_order]
            right_pos = list(range(len(left_order), len(left_order) + len(right_order)))
            positions.extend(right_pos)
            all_labels.extend([names[k] for k in right_order])
            parts_right = ax2.violinplot(right_data, positions=right_pos, showmeans=True, showmedians=True)
            for pc in parts_right['bodies']:
                pc.set_facecolor('#e67e22')
                pc.set_alpha(0.7)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        ax.set_ylabel('Effect (after − before)\nresponse time (ms), CPU (millicores), memory (MB)', fontsize=10)
        ax2.set_ylabel('Effect (after − before)\nanomaly rate (%), deadline miss (%)', fontsize=10)
        ax.set_title(f'Treatment Effect on Metrics - {label}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"config_treatment_effect_violin_{label}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

    def _plot_treatment_effect_line(self, label: str, effect_by_metric: Dict[str, List[float]], plots_dir: str) -> None:
        """Line plot: treatment effect (after - before) per change index. Dual y-axis: left = large-scale (response time, CPU, memory), right = percent (anomaly, deadline miss)."""
        names = self._treatment_effect_metric_names()
        left_metrics = ['mean_response_time', 'p90_response_time', 'cpu_utilization', 'memory_utilization']
        right_metrics = ['anomaly_rate', 'deadline_miss_rate']
        left_order = [k for k in left_metrics if k in effect_by_metric]
        right_order = [k for k in right_metrics if k in effect_by_metric]
        if not left_order and not right_order:
            return
        n_changes = max((len(effect_by_metric[k]) for k in (left_order + right_order)), default=0)
        if n_changes == 0:
            return
        x = list(range(1, n_changes + 1))
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax2 = ax.twinx()
        left_colors = plt.cm.tab10(np.linspace(0, 0.5, max(1, len(left_order))))
        right_colors = plt.cm.tab10(np.linspace(0.5, 1.0, max(1, len(right_order))))
        for i, key in enumerate(left_order):
            effects = effect_by_metric[key]
            ax.plot(x[:len(effects)], effects, 'o-', label=names[key], color=left_colors[i], linewidth=2, markersize=6)
        for i, key in enumerate(right_order):
            effects = effect_by_metric[key]
            ax2.plot(x[:len(effects)], effects, 's-', label=names[key], color=right_colors[i], linewidth=2, markersize=6)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('Change index', fontsize=11)
        ax.set_ylabel('Effect (after − before)\nresponse time (ms), CPU (millicores), memory (MB)', fontsize=10)
        ax2.set_ylabel('Effect (after − before)\nanomaly rate (%), deadline miss (%)', fontsize=10)
        ax.set_title(f'Treatment Effect Over Changes - {label}', fontsize=12, fontweight='bold')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"config_treatment_effect_line_{label}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

    def _plot_treatment_effect_line_zscore(self, label: str, effect_by_metric: Dict[str, List[float]], plots_dir: str) -> None:
        """Line plot: z-scored treatment effect per change index (one scale so all metrics comparable)."""
        names = self._treatment_effect_metric_names()
        metric_order = [k for k in names if k in effect_by_metric]
        if not metric_order:
            return
        n_changes = max(len(effect_by_metric[k]) for k in metric_order)
        if n_changes == 0:
            return
        # Z-score per metric: (effect - mean(effect)) / std(effect); std=0 -> 0
        z_by_metric: Dict[str, List[float]] = {}
        for key in metric_order:
            vals = np.array(effect_by_metric[key], dtype=float)
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            if std_v > 0:
                z_by_metric[key] = ((vals - mean_v) / std_v).tolist()
            else:
                z_by_metric[key] = [0.0] * len(vals)
        x = list(range(1, n_changes + 1))
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        colors = plt.cm.tab10(np.linspace(0, 1, len(metric_order)))
        for i, key in enumerate(metric_order):
            z_effects = z_by_metric[key]
            ax.plot(x[:len(z_effects)], z_effects, 'o-', label=names[key], color=colors[i], linewidth=2, markersize=6)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('Change index', fontsize=11)
        ax.set_ylabel('Effect (z-score, per metric)', fontsize=11)
        ax.set_title(f'Treatment Effect Over Changes (z-score) - {label}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"config_treatment_effect_line_zscore_{label}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

    def _plot_toxicity_violin(self, label: str, toxicity_by_metric: Dict[str, List[float]], plots_dir: str) -> None:
        """Violin plot: distribution of toxicity (during − mean(before, after)) per metric. Dual y-axis: left = response time, CPU, memory; right = anomaly %, deadline miss %."""
        names = self._treatment_effect_metric_names()
        left_metrics = ['mean_response_time', 'p90_response_time', 'cpu_utilization', 'memory_utilization']
        right_metrics = ['anomaly_rate', 'deadline_miss_rate']
        left_order = [k for k in left_metrics if k in toxicity_by_metric]
        right_order = [k for k in right_metrics if k in toxicity_by_metric]
        if not left_order and not right_order:
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax2 = ax.twinx()
        positions = []
        all_labels = []
        if left_order:
            left_data = [toxicity_by_metric[k] for k in left_order]
            left_pos = list(range(len(left_order)))
            positions.extend(left_pos)
            all_labels.extend([names[k] for k in left_order])
            parts_left = ax.violinplot(left_data, positions=left_pos, showmeans=True, showmedians=True)
            for pc in parts_left['bodies']:
                pc.set_facecolor('#3498db')
                pc.set_alpha(0.7)
        if right_order:
            right_data = [toxicity_by_metric[k] for k in right_order]
            right_pos = list(range(len(left_order), len(left_order) + len(right_order)))
            positions.extend(right_pos)
            all_labels.extend([names[k] for k in right_order])
            parts_right = ax2.violinplot(right_data, positions=right_pos, showmeans=True, showmedians=True)
            for pc in parts_right['bodies']:
                pc.set_facecolor('#e67e22')
                pc.set_alpha(0.7)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        ax.set_ylabel('Toxicity (during − mean(before, after))\nresponse time (ms), CPU (millicores), memory (MB)', fontsize=10)
        ax2.set_ylabel('Toxicity (during − mean(before, after))\nanomaly rate (%), deadline miss (%)', fontsize=10)
        ax.set_title(f'Configuration Change Toxicity - {label}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"config_toxicity_violin_{label}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

    def _plot_toxicity_line(self, label: str, toxicity_by_metric: Dict[str, List[float]], plots_dir: str) -> None:
        """Line plot: toxicity (during − mean(before, after)) per change index. Dual y-axis: left = large-scale, right = percent metrics."""
        names = self._treatment_effect_metric_names()
        left_metrics = ['mean_response_time', 'p90_response_time', 'cpu_utilization', 'memory_utilization']
        right_metrics = ['anomaly_rate', 'deadline_miss_rate']
        left_order = [k for k in left_metrics if k in toxicity_by_metric]
        right_order = [k for k in right_metrics if k in toxicity_by_metric]
        if not left_order and not right_order:
            return
        n_changes = max((len(toxicity_by_metric[k]) for k in (left_order + right_order)), default=0)
        if n_changes == 0:
            return
        x = list(range(1, n_changes + 1))
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax2 = ax.twinx()
        left_colors = plt.cm.tab10(np.linspace(0, 0.5, max(1, len(left_order))))
        right_colors = plt.cm.tab10(np.linspace(0.5, 1.0, max(1, len(right_order))))
        for i, key in enumerate(left_order):
            vals = toxicity_by_metric[key]
            ax.plot(x[:len(vals)], vals, 'o-', label=names[key], color=left_colors[i], linewidth=2, markersize=6)
        for i, key in enumerate(right_order):
            vals = toxicity_by_metric[key]
            ax2.plot(x[:len(vals)], vals, 's-', label=names[key], color=right_colors[i], linewidth=2, markersize=6)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('Change index', fontsize=11)
        ax.set_ylabel('Toxicity (during − mean(before, after))\nresponse time (ms), CPU (millicores), memory (MB)', fontsize=10)
        ax2.set_ylabel('Toxicity (during − mean(before, after))\nanomaly rate (%), deadline miss (%)', fontsize=10)
        ax.set_title(f'Configuration Change Toxicity Over Changes - {label}', fontsize=12, fontweight='bold')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"config_toxicity_line_{label}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

    def _plot_toxicity_line_zscore(self, label: str, toxicity_by_metric: Dict[str, List[float]], plots_dir: str) -> None:
        """Line plot: z-scored toxicity per change index (one scale so all metrics comparable)."""
        names = self._treatment_effect_metric_names()
        metric_order = [k for k in names if k in toxicity_by_metric]
        if not metric_order:
            return
        n_changes = max(len(toxicity_by_metric[k]) for k in metric_order)
        if n_changes == 0:
            return
        z_by_metric: Dict[str, List[float]] = {}
        for key in metric_order:
            vals = np.array(toxicity_by_metric[key], dtype=float)
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            if std_v > 0:
                z_by_metric[key] = ((vals - mean_v) / std_v).tolist()
            else:
                z_by_metric[key] = [0.0] * len(vals)
        x = list(range(1, n_changes + 1))
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        colors = plt.cm.tab10(np.linspace(0, 1, len(metric_order)))
        for i, key in enumerate(metric_order):
            z_vals = z_by_metric[key]
            ax.plot(x[:len(z_vals)], z_vals, 'o-', label=names[key], color=colors[i], linewidth=2, markersize=6)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('Change index', fontsize=11)
        ax.set_ylabel('Toxicity (z-score, per metric)', fontsize=11)
        ax.set_title(f'Configuration Change Toxicity Over Changes (z-score) - {label}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"config_toxicity_line_zscore_{label}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

    def _plot_overhead_comparison(self, label: str, metrics: Dict[str, Dict[str, List[float]]]) -> None:
        """Create box plots comparing metrics across before/during/after periods."""
        # Filter out metrics with no data
        available_metrics = {k: v for k, v in metrics.items() 
                            if any(len(v[phase]) > 0 for phase in ['before', 'during', 'after'])}
        
        if not available_metrics:
            return
        
        # Create subplots
        num_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.patch.set_facecolor('white')
        axes = axes.flatten()
        
        metric_names = {
            'mean_response_time': 'Mean Response Time (ms)',
            'p90_response_time': 'P90 Response Time (ms)',
            'cpu_utilization': 'CPU Utilization (millicores)',
            'memory_utilization': 'Memory Utilization (MB)',
            'anomaly_rate': 'Anomaly Rate (%)',
            'deadline_miss_rate': 'Deadline Miss Rate (%)',
        }
        
        for idx, (metric_key, metric_data) in enumerate(sorted(available_metrics.items())):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            ax.set_facecolor('white')
            
            # Prepare data for box plot
            data_to_plot = []
            labels = []
            
            for phase in ['before', 'during', 'after']:
                values = metric_data[phase]
                if values:
                    data_to_plot.append(values)
                    labels.append(phase.title())
            
            if not data_to_plot:
                continue
            
            # Create box plot
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                          showmeans=True, meanline=True)
            
            # Color code: before=blue, during=orange (overhead), after=green
            colors = {'before': '#3498db', 'during': '#e74c3c', 'after': '#2ecc71'}
            for patch, phase_label in zip(bp['boxes'], labels):
                patch.set_facecolor(colors.get(phase_label.lower(), '#95a5a6'))
                patch.set_alpha(0.7)
            
            ax.set_ylabel(metric_names.get(metric_key, metric_key), fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_title(metric_names.get(metric_key, metric_key), fontsize=11, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Configuration Change Overhead Comparison - {label}', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(self.output_dir, label, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, f"config_overhead_{label}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()
    
    def _parse_duration(self, duration_str: str) -> float:
        """Parse duration string (e.g., '30s', '1m') to seconds."""
        if not duration_str:
            return 0.0
        duration_str = str(duration_str).strip()
        if duration_str.endswith('s'):
            return float(duration_str[:-1])
        elif duration_str.endswith('m'):
            return float(duration_str[:-1]) * 60
        elif duration_str.endswith('h'):
            return float(duration_str[:-1]) * 3600
        else:
            # Try to parse as number (assume seconds)
            try:
                return float(duration_str)
            except:
                return 0.0
    
    def _parse_cpu(self, cpu_str: str) -> float:
        """Parse CPU string (e.g., '500m', '1') to cores."""
        if not cpu_str:
            return 0.0
        cpu_str = str(cpu_str).strip()
        if cpu_str.endswith('m'):
            return float(cpu_str[:-1]) / 1000.0
        else:
            try:
                return float(cpu_str)
            except:
                return 0.0
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory string (e.g., '512Mi', '1Gi') to MB."""
        if not memory_str:
            return 0.0
        memory_str = str(memory_str).strip()
        if memory_str.endswith('Mi'):
            return float(memory_str[:-2])
        elif memory_str.endswith('Gi'):
            return float(memory_str[:-2]) * 1024
        elif memory_str.endswith('Ki'):
            return float(memory_str[:-2]) / 1024
        else:
            try:
                return float(memory_str)
            except:
                return 0.0
    
    def _extract_config_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant parameters from a configuration dictionary."""
        params = {}
        
        kind = config.get('kind', '')
        
        # Handle Knative Service resources
        if kind == 'Service':
            # Extract annotations (autoscaling parameters)
            annotations = config.get('spec', {}).get('template', {}).get('metadata', {}).get('annotations', {})
            
            # Autoscaling parameters
            params['max-scale'] = int(annotations.get('autoscaling.knative.dev/max-scale', 0))
            params['min-scale'] = int(annotations.get('autoscaling.knative.dev/min-scale', 0))
            params['target'] = float(annotations.get('autoscaling.knative.dev/target', 0))
            params['metric'] = annotations.get('autoscaling.knative.dev/metric', '')
            params['window'] = self._parse_duration(annotations.get('autoscaling.knative.dev/window', '0s'))
            params['stable-window'] = self._parse_duration(annotations.get('autoscaling.knative.dev/stable-window', '0s'))
            params['scale-down-delay'] = self._parse_duration(annotations.get('autoscaling.knative.dev/scale-down-delay', '0s'))
            params['panic-threshold-percentage'] = float(annotations.get('autoscaling.knative.dev/panic-threshold-percentage', 0))
            params['panic-window-percentage'] = float(annotations.get('autoscaling.knative.dev/panic-window-percentage', 0))
            params['initial-scale'] = int(annotations.get('autoscaling.knative.dev/initial-scale', 0))
            params['activation-scale'] = int(annotations.get('autoscaling.knative.dev/activation-scale', 0))
            
            # Resource parameters
            containers = config.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
            if containers:
                resources = containers[0].get('resources', {})
                limits = resources.get('limits', {})
                requests = resources.get('requests', {})
                
                params['cpu-limit'] = self._parse_cpu(limits.get('cpu', '0'))
                params['cpu-request'] = self._parse_cpu(requests.get('cpu', '0'))
                params['memory-limit'] = self._parse_memory(limits.get('memory', '0'))
                params['memory-request'] = self._parse_memory(requests.get('memory', '0'))
                
                # Extract environment variables
                env_vars = containers[0].get('env', [])
                for env_var in env_vars:
                    if isinstance(env_var, dict) and 'name' in env_var and 'value' in env_var:
                        env_name = env_var['name']
                        env_value = env_var['value']
                        # Convert numeric strings to numbers if possible
                        try:
                            if '.' in env_value:
                                params[env_name.lower()] = float(env_value)
                            else:
                                params[env_name.lower()] = int(env_value)
                        except ValueError:
                            params[env_name.lower()] = env_value
        
        # Handle Deployment resources (e.g., memcached)
        elif kind == 'Deployment':
            # Extract replicas
            params['replicas'] = int(config.get('spec', {}).get('replicas', 0))
            
            # Resource parameters
            containers = config.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
            if containers:
                resources = containers[0].get('resources', {})
                limits = resources.get('limits', {})
                requests = resources.get('requests', {})
                
                params['cpu-limit'] = self._parse_cpu(limits.get('cpu', '0'))
                params['cpu-request'] = self._parse_cpu(requests.get('cpu', '0'))
                params['memory-limit'] = self._parse_memory(limits.get('memory', '0'))
                params['memory-request'] = self._parse_memory(requests.get('memory', '0'))
                
                # Extract environment variables
                env_vars = containers[0].get('env', [])
                for env_var in env_vars:
                    if isinstance(env_var, dict) and 'name' in env_var and 'value' in env_var:
                        env_name = env_var['name']
                        env_value = env_var['value']
                        # Convert numeric strings to numbers if possible
                        try:
                            if '.' in env_value:
                                params[env_name.lower()] = float(env_value)
                            else:
                                params[env_name.lower()] = int(env_value)
                        except ValueError:
                            params[env_name.lower()] = env_value
        
        # Handle ConfigMap (e.g. config-autoscaler)
        elif kind == 'ConfigMap':
            data = config.get('data', {})
            for key, raw in data.items():
                if not isinstance(raw, str):
                    params[key] = raw
                    continue
                # Try numeric
                try:
                    if '.' in raw:
                        params[key] = float(raw)
                    else:
                        params[key] = int(raw)
                except ValueError:
                    pass
                # Try duration (e.g. "30s", "0s")
                if key not in params and raw.endswith('s'):
                    try:
                        params[key] = self._parse_duration(raw)
                    except Exception:
                        params[key] = raw
                if key not in params:
                    # Boolean or string
                    if raw.lower() in ('true', 'false'):
                        params[key] = raw.lower() == 'true'
                    else:
                        params[key] = raw
        
        return params
    
    def _load_configurations(self, label: str, service_name: str) -> List[Tuple[int, Optional[datetime], Dict[str, Any]]]:
        """
        Load all configurations for a service, sorted by sequence number.
        Returns list of (sequence_number, applied_at, config). applied_at comes from
        .meta.json when present (written by ConfigManager.save_all_configs).
        """
        config_dir = os.path.join(self.output_dir, label, "config")
        if not os.path.exists(config_dir):
            return []

        configs = []

        def read_meta(applied_at_path: str) -> Optional[datetime]:
            if not os.path.exists(applied_at_path):
                return None
            try:
                with open(applied_at_path, 'r') as f:
                    meta = json.load(f)
                ts = meta.get("applied_at")
                if ts is None:
                    return None
                parsed_ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                return parsed_ts
            except Exception as e:
                pass  # skip missing or invalid .meta.json
                return None

        # Load base configuration (if exists)
        base_config_path = os.path.join(config_dir, f"{service_name}.yaml")
        if os.path.exists(base_config_path):
            with open(base_config_path, 'r') as f:
                config = yaml.safe_load(f)
            configs.append((-1, read_meta(os.path.join(config_dir, f"{service_name}.meta.json")), config))

        # Load numbered configurations
        pattern = os.path.join(config_dir, f"{service_name}_*.yaml")
        config_files = glob.glob(pattern)

        for config_file in config_files:
            try:
                basename = os.path.basename(config_file)
                match = re.search(r'_(\d+)\.yaml$', basename)
                if match:
                    seq_num = int(match.group(1))
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    meta_path = os.path.join(config_dir, f"{service_name}_{seq_num}.meta.json")
                    configs.append((seq_num, read_meta(meta_path), config))
            except Exception as e:
                pass  # skip unreadable config file
                continue

        configs.sort(key=lambda x: x[0])
        return configs
    
    def plot_configuration_changes(self, label: str, service_name: Optional[str] = None) -> None:
        """
        Plot how configuration parameters change over time for a service or all services.
        
        Args:
            label: Experiment label
            service_name: Specific service name, or None to plot all services
        """
        config_dir = os.path.join(self.output_dir, label, "config")
        if not os.path.exists(config_dir):
            return
        
        # Get list of services to plot
        if service_name:
            services = [service_name]
        else:
            # Find all service configurations
            all_configs = glob.glob(os.path.join(config_dir, "*.yaml"))
            services = set()
            for config_file in all_configs:
                basename = os.path.basename(config_file)
                # Skip base config files and extract service name
                if not basename.endswith('.yaml'):
                    continue
                # Remove sequence number suffix if present
                service = re.sub(r'_\d+\.yaml$', '', basename)
                service = service.replace('.yaml', '')
                services.add(service)
            services = sorted(list(services))
        
        if not services:
            return
        
        for service in services:
            configs = self._load_configurations(label, service)
            if len(configs) < 2:
                continue
            
            self._plot_service_configuration_changes(label, service, configs)
    
    def _plot_service_configuration_changes(self, label: str, service_name: str, configs: List[Tuple[int, Optional[datetime], Dict[str, Any]]]) -> None:
        """Plot configuration parameter changes for a specific service. Uses applied_at for x-axis when present."""
        # Collect all param keys that appear in any config (mix of Knative/Deployment can have different keys per config)
        all_param_keys = set()
        for _seq, _at, config in configs:
            all_param_keys |= set(self._extract_config_parameters(config).keys())

        # Extract parameters from all configurations; pad missing keys so every param has len(configs) values
        param_data = {}
        sequence_numbers = []
        applied_ats: List[Optional[datetime]] = []

        for seq_num, applied_at, config in configs:
            params = self._extract_config_parameters(config)
            sequence_numbers.append(seq_num)
            applied_ats.append(applied_at)
            for param_name in all_param_keys:
                val = params.get(param_name, None)
                # Use NaN for missing or non-numeric so every param has len(configs) values (x/y match)
                if val is None or isinstance(val, str):
                    plot_val = float('nan')
                elif isinstance(val, (int, float)) and (val == val and val != float('inf') and val != float('-inf')):
                    plot_val = val
                else:
                    plot_val = float('nan')
                param_data.setdefault(param_name, []).append(plot_val)

        # Use timestamps on x-axis when at least one entry has applied_at; fill missing with first snapshot time
        timestamps_found = sum(1 for t in applied_ats if t is not None)
        use_time_axis = timestamps_found > 0 and len(applied_ats) > 0

        if use_time_axis:
            # Fallback for configs without .meta.json: use first snapshot time (experiment start)
            try:
                snapshots = self._get_sorted_snapshots(label)
                fallback_time = self._get_snapshot_time(snapshots[0]) if snapshots else min(t for t in applied_ats if t is not None)
            except Exception:
                fallback_time = min(t for t in applied_ats if t is not None)
            x_values = [applied_ats[i] if applied_ats[i] is not None else fallback_time for i in range(len(applied_ats))]
        else:
            x_values = sequence_numbers


        # Filter out parameters that never change
        changing_params = {k: v for k, v in param_data.items() if len(set(v)) > 1}

        if not changing_params:
            return

        # Create subplots - organize into groups
        num_params = len(changing_params)
        if num_params == 0:
            return

        # Group parameters by category (for reference, not currently used but may be useful later)
        autoscaling_params = [p for p in changing_params.keys() if p in [
            'max-scale', 'min-scale', 'target', 'window', 'stable-window',
            'scale-down-delay', 'panic-threshold-percentage', 'panic-window-percentage',
            'initial-scale', 'activation-scale', 'metric'
        ]]
        resource_params = [p for p in changing_params.keys() if p in [
            'cpu-limit', 'cpu-request', 'memory-limit', 'memory-request', 'replicas'
        ]]
        env_params = [p for p in changing_params.keys() if p not in autoscaling_params and p not in resource_params]

        # Create figure with subplots
        fig = plt.figure(figsize=(16, max(6, len(changing_params) * 1.5)))
        fig.patch.set_facecolor('white')

        gs = fig.add_gridspec(len(changing_params), 1, hspace=0.3)

        # When using time axis, overlay adaptation-phase timeline (e.g. stabilization) for correlation
        phase_timeline = None
        if use_time_axis:
            phase_timeline = self._get_adaptation_phase_timeline(label)

        for idx, (param_name, param_values) in enumerate(sorted(changing_params.items())):
            ax = fig.add_subplot(gs[idx, 0])
            ax.set_facecolor('white')

            # Plot parameter values (x-axis = time when available, else sequence number)
            ax.plot(x_values, param_values, 'o-', linewidth=2, markersize=6, label=param_name)

            # Overlay adaptation-phase background (stabilization) when we have time axis and snapshot timeline
            if use_time_axis and phase_timeline is not None:
                phase_ts, phase_subphases = phase_timeline
                self._add_stabilization_background(ax, phase_ts, phase_subphases)

            # Formatting
            ax.set_ylabel(self._format_param_name(param_name), fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')

            # Format x-axis only for bottom plot
            if idx == len(changing_params) - 1:
                if use_time_axis:
                    self._format_time_axis(ax, x_values)
                    ax.set_xlabel('Time (applied_at)', fontsize=10)
                else:
                    ax.set_xlabel('Configuration sequence', fontsize=10)
            else:
                ax.set_xticklabels([])

        plt.suptitle(f'Configuration Parameter Changes - {service_name} ({label})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(self.output_dir, label, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        safe_service = self._sanitize_filename(service_name)
        out_path = os.path.join(plots_dir, f"config_changes_{label}_{safe_service}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()
    
    def _format_param_name(self, param_name: str) -> str:
        """Format parameter name for display."""
        replacements = {
            'max-scale': 'Max Scale',
            'min-scale': 'Min Scale',
            'target': 'Target',
            'window': 'Window (s)',
            'stable-window': 'Stable Window (s)',
            'scale-down-delay': 'Scale Down Delay (s)',
            'panic-threshold-percentage': 'Panic Threshold (%)',
            'panic-window-percentage': 'Panic Window (%)',
            'initial-scale': 'Initial Scale',
            'activation-scale': 'Activation Scale',
            'metric': 'Metric',
            'cpu-limit': 'CPU Limit (cores)',
            'cpu-request': 'CPU Request (cores)',
            'memory-limit': 'Memory Limit (MB)',
            'memory-request': 'Memory Request (MB)',
            'replicas': 'Replicas',
            'memcached_cache_size': 'Memcached Cache Size (MB)',
            'memcached_threads': 'Memcached Threads',
        }
        # If not in replacements, try to format intelligently
        if param_name not in replacements:
            # Handle environment variable names (uppercase with underscores)
            if '_' in param_name:
                return param_name.replace('_', ' ').title()
            # Handle kebab-case
            return param_name.replace('-', ' ').title()
        return replacements.get(param_name, param_name.replace('-', ' ').title())



    def plot_configuration_summary(self, label: str) -> None:
        """
        Create a summary plot showing which parameters changed for which services.
        
        Args:
            label: Experiment label
        """
        config_dir = os.path.join(self.output_dir, label, "config")
        if not os.path.exists(config_dir):
            return
        
        # Find all services
        all_configs = glob.glob(os.path.join(config_dir, "*.yaml"))
        services = set()
        for config_file in all_configs:
            basename = os.path.basename(config_file)
            if not basename.endswith('.yaml'):
                continue
            service = re.sub(r'_\d+\.yaml$', '', basename)
            service = service.replace('.yaml', '')
            services.add(service)
        services = sorted(list(services))
        
        if not services:
            return
        
        # Collect parameter changes for each service
        service_changes = {}
        all_params = set()
        
        for service in services:
            configs = self._load_configurations(label, service)
            if len(configs) < 2:
                continue
            
            changing_params = set()
            prev_params = None
            
            for seq_num, _applied_at, config in configs:
                params = self._extract_config_parameters(config)
                if prev_params is not None:
                    for param_name, param_value in params.items():
                        if param_name in prev_params and prev_params[param_name] != param_value:
                            changing_params.add(param_name)
                            all_params.add(param_name)
                prev_params = params
            
            if changing_params:
                service_changes[service] = changing_params
        
        if not service_changes:
            return
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(6, len(services) * 0.5)))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Prepare data for heatmap
        all_params = sorted(list(all_params))
        data = []
        for service in services:
            if service in service_changes:
                row = [1 if param in service_changes[service] else 0 for param in all_params]
                data.append(row)
            else:
                data.append([0] * len(all_params))
        
        # Create heatmap
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(all_params)))
        ax.set_yticks(np.arange(len(services)))
        ax.set_xticklabels([self._format_param_name(p) for p in all_params], rotation=45, ha='right')
        ax.set_yticklabels(services)
        
        # Add text annotations
        for i in range(len(services)):
            for j in range(len(all_params)):
                if data[i][j] == 1:
                    ax.text(j, i, '●', ha='center', va='center', color='darkred', fontsize=12)
        
        ax.set_title(f'Configuration Parameter Changes Summary - {label}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Parameters', fontsize=12)
        ax.set_ylabel('Services', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(self.output_dir, label, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, f"config_summary_{label}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

    def print_configuration_change_stats(self, label: str) -> None:
        """
        For this experiment label, list per-service parameters that changed across
        saved configurations and print the number of configuration changes.
        """
        config_dir = os.path.join(self.output_dir, label, "config")
        if not os.path.exists(config_dir):
            print(f"Config directory not found: {config_dir}")
            return

        all_configs = glob.glob(os.path.join(config_dir, "*.yaml"))
        services = set()
        for config_file in all_configs:
            basename = os.path.basename(config_file)
            if not basename.endswith(".yaml"):
                continue
            service = re.sub(r"_\d+\.yaml$", "", basename)
            service = service.replace(".yaml", "")
            services.add(service)
        services = sorted(services)

        service_changes = {}
        num_config_updates = 0

        for service in services:
            configs = self._load_configurations(label, service)
            if len(configs) < 2:
                continue
            changing_params = set()
            prev_params = None
            for _seq_num, _applied_at, config in configs:
                params = self._extract_config_parameters(config)
                if prev_params is not None:
                    for param_name, param_value in params.items():
                        if param_name in prev_params and prev_params[param_name] != param_value:
                            changing_params.add(param_name)
                prev_params = params
            if changing_params:
                service_changes[service] = sorted(changing_params)
            num_config_updates += max(0, len(configs) - 1)

        total_changed_params = sum(len(p) for p in service_changes.values())

        print(f"\n--- Configuration change statistics (label={label}) ---")
        print(f"Number of configuration changes (config updates): {num_config_updates}")
        print(f"Total distinct parameters that changed (across all services): {total_changed_params}")
        print("\nPer-service parameters that changed:")
        for service in services:
            params = service_changes.get(service, [])
            if params:
                print(f"  {service}: {params}")
        if not service_changes:
            print("  (none)")
        print("---\n")


def main():
    import argparse

    # When run as script, default output_dir to project root / output (so it works from ctl/ or ctl/util/)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)  # ctl when run from ctl/util/plot.py
    _default_output = os.path.join(_project_root, "output")
    plotter = SnapshotPlotter(output_dir=_default_output)

    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True)
    args = parser.parse_args()

    label = args.label
    try:
        plotter.plot_all_timeseries(label)
        plotter.print_configuration_change_stats(label)
    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
