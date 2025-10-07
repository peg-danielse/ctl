#!/usr/bin/env python3
"""
Configuration Analysis Utilities for analyzing configuration changes and their impact.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from datetime import datetime

class ConfigurationAnalyzer:
    """
    Analyzes configuration changes and their impact on performance.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the configuration analyzer.
        
        Args:
            output_dir: Base output directory for configuration logs
        """
        self.output_dir = output_dir
    
    def analyze_configuration_impact(self, label: str) -> Dict[str, Any]:
        """
        Analyze the impact of configuration changes on performance.
        
        Args:
            label: Test run label
            
        Returns:
            Analysis results dictionary
        """
        print(f"🔍 Analyzing configuration impact for label: {label}")
        
        # Load configuration changes
        config_changes = self._load_config_changes(label)
        if not config_changes:
            print(f"❌ No configuration changes found for label: {label}")
            return {}
        
        # Load snapshot data
        snapshots = self._load_snapshot_data(label)
        
        # Analyze each service
        analysis_results = {
            "label": label,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        for service_name, changes in config_changes.items():
            service_analysis = self._analyze_service_config_changes(
                service_name, changes, snapshots
            )
            analysis_results["services"][service_name] = service_analysis
        
        # Generate summary statistics
        analysis_results["summary"] = self._generate_summary_stats(analysis_results["services"])
        
        return analysis_results
    
    def _load_config_changes(self, label: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load configuration changes for a label.
        
        Args:
            label: Test run label
            
        Returns:
            Dictionary mapping service names to their configuration changes
        """
        config_logs_dir = os.path.join(self.output_dir, label, "config_logs")
        
        if not os.path.exists(config_logs_dir):
            return {}
        
        config_changes = {}
        
        for filename in os.listdir(config_logs_dir):
            if filename.endswith('_config_history.jsonl'):
                service_name = filename.replace('_config_history.jsonl', '')
                log_file = os.path.join(config_logs_dir, filename)
                
                changes = []
                with open(log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            changes.append(json.loads(line.strip()))
                
                config_changes[service_name] = changes
        
        return config_changes
    
    def _load_snapshot_data(self, label: str) -> List[Dict[str, Any]]:
        """
        Load snapshot data for a label.
        
        Args:
            label: Test run label
            
        Returns:
            List of snapshot data
        """
        snapshot_file = os.path.join(self.output_dir, label, "snapshots", f"snapshot_log_{label}_continuous_0.jsonl")
        
        if not os.path.exists(snapshot_file):
            return []
        
        snapshots = []
        with open(snapshot_file, 'r') as f:
            for line in f:
                if line.strip():
                    snapshot = json.loads(line.strip())
                    if snapshot.get('type') == 'metric_snapshot':
                        snapshots.append(snapshot)
        
        return snapshots
    
    def _analyze_service_config_changes(self, 
                                      service_name: str, 
                                      changes: List[Dict[str, Any]], 
                                      snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze configuration changes for a specific service.
        
        Args:
            service_name: Name of the service
            changes: List of configuration changes
            snapshots: List of all snapshots
            
        Returns:
            Service analysis results
        """
        analysis = {
            "service_name": service_name,
            "total_changes": len(changes),
            "change_analysis": [],
            "performance_trends": {}
        }
        
        # Analyze each configuration change
        for i, change in enumerate(changes):
            change_analysis = {
                "change_index": i,
                "timestamp": change.get("timestamp"),
                "reason": change.get("reason"),
                "changes": change.get("changes", {}),
                "performance_before": None,
                "performance_after": None,
                "performance_impact": {}
            }
            
            # Get performance metrics if available
            if change.get("performance_metrics"):
                change_analysis["performance_after"] = change["performance_metrics"]
            
            # Try to find performance before this change
            if i > 0:
                prev_change = changes[i-1]
                if prev_change.get("performance_metrics"):
                    change_analysis["performance_before"] = prev_change["performance_metrics"]
            
            # Calculate performance impact
            if (change_analysis["performance_before"] and 
                change_analysis["performance_after"]):
                impact = self._calculate_performance_impact(
                    change_analysis["performance_before"],
                    change_analysis["performance_after"]
                )
                change_analysis["performance_impact"] = impact
            
            analysis["change_analysis"].append(change_analysis)
        
        # Calculate overall performance trends
        analysis["performance_trends"] = self._calculate_performance_trends(
            service_name, changes, snapshots
        )
        
        return analysis
    
    def _calculate_performance_impact(self, 
                                    before: Dict[str, Any], 
                                    after: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the impact of a configuration change on performance.
        
        Args:
            before: Performance metrics before the change
            after: Performance metrics after the change
            
        Returns:
            Performance impact analysis
        """
        impact = {}
        
        # Response time impact
        if "response_time_avg" in before and "response_time_avg" in after:
            before_rt = before["response_time_avg"]
            after_rt = after["response_time_avg"]
            if before_rt > 0:
                impact["response_time_change_pct"] = ((after_rt - before_rt) / before_rt) * 100
                impact["response_time_improvement"] = after_rt < before_rt
        
        # Error rate impact
        if "error_rate" in before and "error_rate" in after:
            before_err = before["error_rate"]
            after_err = after["error_rate"]
            impact["error_rate_change_pct"] = ((after_err - before_err) / max(before_err, 0.001)) * 100
            impact["error_rate_improvement"] = after_err < before_err
        
        # Request throughput impact
        if "total_requests" in before and "total_requests" in after:
            before_req = before["total_requests"]
            after_req = after["total_requests"]
            if before_req > 0:
                impact["throughput_change_pct"] = ((after_req - before_req) / before_req) * 100
                impact["throughput_improvement"] = after_req > before_req
        
        # Anomaly count impact
        if "anomaly_count" in before and "anomaly_count" in after:
            before_anom = before["anomaly_count"]
            after_anom = after["anomaly_count"]
            impact["anomaly_count_change"] = after_anom - before_anom
            impact["anomaly_improvement"] = after_anom < before_anom
        
        return impact
    
    def _calculate_performance_trends(self, 
                                    service_name: str, 
                                    changes: List[Dict[str, Any]], 
                                    snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall performance trends for a service.
        
        Args:
            service_name: Name of the service
            changes: List of configuration changes
            snapshots: List of all snapshots
            
        Returns:
            Performance trends analysis
        """
        trends = {
            "response_time_trend": "stable",
            "error_rate_trend": "stable",
            "throughput_trend": "stable",
            "anomaly_trend": "stable"
        }
        
        # Extract performance metrics over time
        performance_data = []
        for change in changes:
            if change.get("performance_metrics"):
                perf = change["performance_metrics"]
                performance_data.append({
                    "timestamp": change["timestamp"],
                    "response_time_avg": perf.get("response_time_avg", 0),
                    "error_rate": perf.get("error_rate", 0),
                    "total_requests": perf.get("total_requests", 0),
                    "anomaly_count": perf.get("anomaly_count", 0)
                })
        
        if len(performance_data) < 2:
            return trends
        
        # Calculate trends (simple linear trend analysis)
        df = pd.DataFrame(performance_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Response time trend
        if len(df) > 1:
            rt_slope = (df['response_time_avg'].iloc[-1] - df['response_time_avg'].iloc[0]) / len(df)
            trends["response_time_trend"] = "improving" if rt_slope < -10 else "degrading" if rt_slope > 10 else "stable"
            
            # Error rate trend
            err_slope = (df['error_rate'].iloc[-1] - df['error_rate'].iloc[0]) / len(df)
            trends["error_rate_trend"] = "improving" if err_slope < -0.01 else "degrading" if err_slope > 0.01 else "stable"
            
            # Throughput trend
            thr_slope = (df['total_requests'].iloc[-1] - df['total_requests'].iloc[0]) / len(df)
            trends["throughput_trend"] = "improving" if thr_slope > 1 else "degrading" if thr_slope < -1 else "stable"
            
            # Anomaly trend
            anom_slope = (df['anomaly_count'].iloc[-1] - df['anomaly_count'].iloc[0]) / len(df)
            trends["anomaly_trend"] = "improving" if anom_slope < -0.1 else "degrading" if anom_slope > 0.1 else "stable"
        
        return trends
    
    def _generate_summary_stats(self, services_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics for all services.
        
        Args:
            services_analysis: Analysis results for all services
            
        Returns:
            Summary statistics
        """
        summary = {
            "total_services": len(services_analysis),
            "total_config_changes": sum(s["total_changes"] for s in services_analysis.values()),
            "services_with_improvements": 0,
            "services_with_degradations": 0,
            "most_changed_service": None,
            "best_performing_service": None
        }
        
        if not services_analysis:
            return summary
        
        # Count services with improvements/degradations
        for service_name, analysis in services_analysis.items():
            trends = analysis.get("performance_trends", {})
            if any(trend == "improving" for trend in trends.values()):
                summary["services_with_improvements"] += 1
            if any(trend == "degrading" for trend in trends.values()):
                summary["services_with_degradations"] += 1
        
        # Find most changed service
        most_changes = max(services_analysis.items(), key=lambda x: x[1]["total_changes"])
        summary["most_changed_service"] = {
            "name": most_changes[0],
            "changes": most_changes[1]["total_changes"]
        }
        
        return summary
    
    def generate_configuration_analysis_plots(self, label: str):
        """
        Generate plots for configuration analysis.
        
        Args:
            label: Test run label
        """
        print(f"📊 Generating configuration analysis plots for label: {label}")
        
        # Load analysis data
        analysis = self.analyze_configuration_impact(label)
        if not analysis:
            print(f"❌ No analysis data available for label: {label}")
            return
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, label, "config_analysis")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate service-specific plots
        for service_name, service_analysis in analysis["services"].items():
            self._plot_service_config_analysis(service_name, service_analysis, plots_dir)
        
        # Generate summary plots
        self._plot_configuration_summary(analysis, plots_dir)
        
        print(f"✅ Configuration analysis plots saved to: {plots_dir}")
    
    def _plot_service_config_analysis(self, 
                                    service_name: str, 
                                    analysis: Dict[str, Any], 
                                    plots_dir: str):
        """
        Generate plots for a specific service's configuration analysis.
        
        Args:
            service_name: Name of the service
            analysis: Service analysis data
            plots_dir: Directory to save plots
        """
        if not analysis["change_analysis"]:
            return
        
        # Extract data for plotting
        timestamps = []
        response_times = []
        error_rates = []
        anomaly_counts = []
        
        for change in analysis["change_analysis"]:
            if change.get("performance_after"):
                perf = change["performance_after"]
                timestamps.append(change["timestamp"])
                response_times.append(perf.get("response_time_avg", 0))
                error_rates.append(perf.get("error_rate", 0) * 100)  # Convert to percentage
                anomaly_counts.append(perf.get("anomaly_count", 0))
        
        if not timestamps:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Configuration Analysis: {service_name}', fontsize=16)
        
        # Response time over time
        axes[0, 0].plot(range(len(response_times)), response_times, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Response Time Over Time')
        axes[0, 0].set_xlabel('Configuration Change')
        axes[0, 0].set_ylabel('Average Response Time (ms)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error rate over time
        axes[0, 1].plot(range(len(error_rates)), error_rates, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Error Rate Over Time')
        axes[0, 1].set_xlabel('Configuration Change')
        axes[0, 1].set_ylabel('Error Rate (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Anomaly count over time
        axes[1, 0].plot(range(len(anomaly_counts)), anomaly_counts, 'g-o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Anomaly Count Over Time')
        axes[1, 0].set_xlabel('Configuration Change')
        axes[1, 0].set_ylabel('Number of Anomalies')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance trends summary
        trends = analysis.get("performance_trends", {})
        trend_labels = list(trends.keys())
        trend_values = [1 if trends[t] == "improving" else -1 if trends[t] == "degrading" else 0 for t in trend_labels]
        
        colors = ['green' if v == 1 else 'red' if v == -1 else 'gray' for v in trend_values]
        axes[1, 1].bar(range(len(trend_labels)), trend_values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Performance Trends')
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Trend (1=Improving, 0=Stable, -1=Degrading)')
        axes[1, 1].set_xticks(range(len(trend_labels)))
        axes[1, 1].set_xticklabels([t.replace('_trend', '') for t in trend_labels], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(plots_dir, f"{service_name}_config_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_configuration_summary(self, analysis: Dict[str, Any], plots_dir: str):
        """
        Generate summary plots for all services.
        
        Args:
            analysis: Complete analysis data
            plots_dir: Directory to save plots
        """
        services = list(analysis["services"].keys())
        total_changes = [analysis["services"][s]["total_changes"] for s in services]
        
        # Create summary figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Configuration Change Summary: {analysis["label"]}', fontsize=16)
        
        # Total changes per service
        axes[0].bar(services, total_changes, color='skyblue', alpha=0.7)
        axes[0].set_title('Configuration Changes per Service')
        axes[0].set_xlabel('Service')
        axes[0].set_ylabel('Number of Changes')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Performance trends summary
        improving_services = []
        degrading_services = []
        stable_services = []
        
        for service_name, service_analysis in analysis["services"].items():
            trends = service_analysis.get("performance_trends", {})
            improving_count = sum(1 for t in trends.values() if t == "improving")
            degrading_count = sum(1 for t in trends.values() if t == "degrading")
            
            if improving_count > degrading_count:
                improving_services.append(service_name)
            elif degrading_count > improving_count:
                degrading_services.append(service_name)
            else:
                stable_services.append(service_name)
        
        trend_counts = [len(improving_services), len(stable_services), len(degrading_services)]
        trend_labels = ['Improving', 'Stable', 'Degrading']
        trend_colors = ['green', 'gray', 'red']
        
        axes[1].bar(trend_labels, trend_counts, color=trend_colors, alpha=0.7)
        axes[1].set_title('Service Performance Trends')
        axes[1].set_xlabel('Trend Category')
        axes[1].set_ylabel('Number of Services')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(plots_dir, "configuration_summary.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()


def create_config_analyzer(output_dir: str = "output") -> ConfigurationAnalyzer:
    """
    Create a new ConfigurationAnalyzer instance.
    
    Args:
        output_dir: Base output directory for configuration logs
        
    Returns:
        ConfigurationAnalyzer instance
    """
    return ConfigurationAnalyzer(output_dir)
