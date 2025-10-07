#!/usr/bin/env python3
"""
Configuration Logger for tracking service configuration changes.
"""

import os
import json
import datetime
from typing import Dict, List, Any, Optional

class ConfigurationLogger:
    """
    Logs configuration changes for each service to track adaptation history.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the configuration logger.
        
        Args:
            output_dir: Base output directory for configuration logs
        """
        self.output_dir = output_dir
        self.service_logs = {}  # Track active log files per service
        
    def log_configuration_change(self, 
                                service_name: str, 
                                old_config: Dict[str, Any], 
                                new_config: Dict[str, Any], 
                                label: str,
                                reason: str = "anomaly_detection",
                                performance_metrics: Optional[Dict[str, Any]] = None):
        """
        Log a configuration change for a specific service.
        
        Args:
            service_name: Name of the service being updated
            old_config: Previous configuration
            new_config: New configuration being applied
            label: Test run label
            reason: Reason for the configuration change
            performance_metrics: Current performance metrics (optional)
        """
        try:
            # Create service-specific log directory
            service_log_dir = os.path.join(self.output_dir, label, "config_logs")
            os.makedirs(service_log_dir, exist_ok=True)
            
            # Service-specific log file
            log_file = os.path.join(service_log_dir, f"{service_name}_config_history.jsonl")
            
            # Create log entry
            log_entry = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "service_name": service_name,
                "label": label,
                "reason": reason,
                "old_config": old_config,
                "new_config": new_config,
                "changes": self._calculate_config_changes(old_config, new_config),
                "performance_metrics": performance_metrics or {}
            }
            
            # Append to log file
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # Also maintain in-memory tracking for quick access
            if service_name not in self.service_logs:
                self.service_logs[service_name] = []
            self.service_logs[service_name].append(log_entry)
            
            print(f"📝 Configuration change logged for {service_name}")
            print(f"   Changes: {log_entry['changes']}")
            if performance_metrics:
                print(f"   Performance: {self._format_performance_summary(performance_metrics)}")
            
        except Exception as e:
            print(f"❌ Error logging configuration change for {service_name}: {e}")
    
    def _calculate_config_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the differences between old and new configurations.
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
            
        Returns:
            Dictionary of changes made
        """
        changes = {}
        
        # Check for added parameters
        for key, value in new_config.items():
            if key not in old_config:
                changes[f"added_{key}"] = value
            elif old_config[key] != value:
                changes[f"changed_{key}"] = {
                    "old": old_config[key],
                    "new": value
                }
        
        # Check for removed parameters
        for key in old_config:
            if key not in new_config:
                changes[f"removed_{key}"] = old_config[key]
        
        return changes
    
    def _format_performance_summary(self, metrics: Dict[str, Any]) -> str:
        """
        Format performance metrics for logging.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Formatted string summary
        """
        summary_parts = []
        
        if 'response_time_avg' in metrics:
            summary_parts.append(f"avg_rt={metrics['response_time_avg']:.1f}ms")
        if 'total_requests' in metrics:
            summary_parts.append(f"requests={metrics['total_requests']}")
        if 'error_rate' in metrics:
            summary_parts.append(f"error_rate={metrics['error_rate']:.2%}")
        if 'anomaly_count' in metrics:
            summary_parts.append(f"anomalies={metrics['anomaly_count']}")
        
        return ", ".join(summary_parts)
    
    def get_service_config_history(self, service_name: str, label: str) -> List[Dict[str, Any]]:
        """
        Get configuration history for a specific service.
        
        Args:
            service_name: Name of the service
            label: Test run label
            
        Returns:
            List of configuration change entries
        """
        try:
            log_file = os.path.join(self.output_dir, label, "config_logs", f"{service_name}_config_history.jsonl")
            
            if not os.path.exists(log_file):
                return []
            
            history = []
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line.strip()))
            
            return history
            
        except Exception as e:
            print(f"❌ Error reading configuration history for {service_name}: {e}")
            return []
    
    def get_all_config_changes(self, label: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get configuration history for all services.
        
        Args:
            label: Test run label
            
        Returns:
            Dictionary mapping service names to their configuration histories
        """
        config_logs_dir = os.path.join(self.output_dir, label, "config_logs")
        
        if not os.path.exists(config_logs_dir):
            return {}
        
        all_changes = {}
        
        for filename in os.listdir(config_logs_dir):
            if filename.endswith('_config_history.jsonl'):
                service_name = filename.replace('_config_history.jsonl', '')
                all_changes[service_name] = self.get_service_config_history(service_name, label)
        
        return all_changes
    
    def create_config_summary(self, label: str) -> Dict[str, Any]:
        """
        Create a summary of all configuration changes for a test run.
        
        Args:
            label: Test run label
            
        Returns:
            Summary dictionary with statistics and trends
        """
        all_changes = self.get_all_config_changes(label)
        
        summary = {
            "label": label,
            "generated_at": datetime.datetime.utcnow().isoformat(),
            "total_services": len(all_changes),
            "total_changes": sum(len(changes) for changes in all_changes.values()),
            "services": {}
        }
        
        for service_name, changes in all_changes.items():
            service_summary = {
                "total_changes": len(changes),
                "first_change": changes[0]["timestamp"] if changes else None,
                "last_change": changes[-1]["timestamp"] if changes else None,
                "change_reasons": {},
                "config_parameters_changed": set()
            }
            
            for change in changes:
                reason = change.get("reason", "unknown")
                service_summary["change_reasons"][reason] = service_summary["change_reasons"].get(reason, 0) + 1
                
                # Track which parameters were changed
                for change_key in change.get("changes", {}).keys():
                    if change_key.startswith("changed_"):
                        param = change_key.replace("changed_", "")
                        service_summary["config_parameters_changed"].add(param)
            
            service_summary["config_parameters_changed"] = list(service_summary["config_parameters_changed"])
            summary["services"][service_name] = service_summary
        
        return summary
    
    def save_config_summary(self, label: str):
        """
        Save configuration summary to a JSON file.
        
        Args:
            label: Test run label
        """
        try:
            summary = self.create_config_summary(label)
            
            summary_file = os.path.join(self.output_dir, label, "config_logs", "configuration_summary.json")
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📊 Configuration summary saved to {summary_file}")
            
        except Exception as e:
            print(f"❌ Error saving configuration summary: {e}")
    
    def create_configuration_evolution_report(self, label: str):
        """
        Create a comprehensive configuration evolution report for the experiment.
        
        Args:
            label: Test run label
            
        Returns:
            Dictionary containing detailed configuration evolution data
        """
        try:
            print(f"📊 Creating configuration evolution report for {label}...")
            
            all_changes = self.get_all_config_changes(label)
            
            evolution_report = {
                "experiment_label": label,
                "generated_at": datetime.datetime.utcnow().isoformat(),
                "total_services": len(all_changes),
                "total_configuration_changes": sum(len(changes) for changes in all_changes.values()),
                "experiment_duration": self._calculate_experiment_duration(all_changes),
                "services": {}
            }
            
            for service_name, changes in all_changes.items():
                service_evolution = self._analyze_service_configuration_evolution(service_name, changes)
                evolution_report["services"][service_name] = service_evolution
            
            # Add cross-service analysis
            evolution_report["cross_service_analysis"] = self._analyze_cross_service_patterns(all_changes)
            
            return evolution_report
            
        except Exception as e:
            print(f"❌ Error creating configuration evolution report: {e}")
            return {}
    
    def _calculate_experiment_duration(self, all_changes: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """
        Calculate the duration of the experiment based on configuration changes.
        
        Args:
            all_changes: All configuration changes across services
            
        Returns:
            Dictionary with start, end, and duration information
        """
        all_timestamps = []
        
        for changes in all_changes.values():
            for change in changes:
                all_timestamps.append(change["timestamp"])
        
        if not all_timestamps:
            return {"start": None, "end": None, "duration_minutes": 0}
        
        all_timestamps.sort()
        start_time = datetime.datetime.fromisoformat(all_timestamps[0].replace('Z', '+00:00'))
        end_time = datetime.datetime.fromisoformat(all_timestamps[-1].replace('Z', '+00:00'))
        duration = (end_time - start_time).total_seconds() / 60  # minutes
        
        return {
            "start": all_timestamps[0],
            "end": all_timestamps[-1],
            "duration_minutes": round(duration, 2)
        }
    
    def _analyze_service_configuration_evolution(self, service_name: str, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the configuration evolution for a specific service.
        
        Args:
            service_name: Name of the service
            changes: List of configuration changes for the service
            
        Returns:
            Dictionary with detailed evolution analysis
        """
        if not changes:
            return {"total_changes": 0, "evolution": []}
        
        evolution = {
            "service_name": service_name,
            "total_changes": len(changes),
            "first_change": changes[0]["timestamp"],
            "last_change": changes[-1]["timestamp"],
            "change_frequency_minutes": self._calculate_change_frequency(changes),
            "evolution_timeline": [],
            "parameter_evolution": {},
            "performance_correlation": {},
            "change_patterns": {}
        }
        
        # Analyze each configuration change
        for i, change in enumerate(changes):
            change_analysis = {
                "change_number": i + 1,
                "timestamp": change["timestamp"],
                "reason": change.get("reason", "unknown"),
                "changes_made": change.get("changes", {}),
                "performance_metrics": change.get("performance_metrics", {}),
                "configuration_snapshot": change.get("new_config", {})
            }
            
            evolution["evolution_timeline"].append(change_analysis)
            
            # Track parameter evolution
            for param, change_info in change.get("changes", {}).items():
                if param.startswith("changed_"):
                    param_name = param.replace("changed_", "")
                    if param_name not in evolution["parameter_evolution"]:
                        evolution["parameter_evolution"][param_name] = []
                    
                    evolution["parameter_evolution"][param_name].append({
                        "change_number": i + 1,
                        "timestamp": change["timestamp"],
                        "old_value": change_info.get("old"),
                        "new_value": change_info.get("new"),
                        "reason": change.get("reason", "unknown")
                    })
        
        # Analyze change patterns
        evolution["change_patterns"] = self._analyze_change_patterns(changes)
        
        return evolution
    
    def _calculate_change_frequency(self, changes: List[Dict[str, Any]]) -> float:
        """
        Calculate the average time between configuration changes.
        
        Args:
            changes: List of configuration changes
            
        Returns:
            Average time between changes in minutes
        """
        if len(changes) < 2:
            return 0.0
        
        timestamps = [change["timestamp"] for change in changes]
        timestamps.sort()
        
        total_duration = 0
        for i in range(1, len(timestamps)):
            start = datetime.datetime.fromisoformat(timestamps[i-1].replace('Z', '+00:00'))
            end = datetime.datetime.fromisoformat(timestamps[i].replace('Z', '+00:00'))
            total_duration += (end - start).total_seconds()
        
        return round(total_duration / (len(timestamps) - 1) / 60, 2)  # minutes
    
    def _analyze_change_patterns(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in configuration changes.
        
        Args:
            changes: List of configuration changes
            
        Returns:
            Dictionary with pattern analysis
        """
        patterns = {
            "most_changed_parameters": {},
            "change_reasons": {},
            "configuration_stability": {},
            "rollback_instances": 0
        }
        
        # Count parameter changes
        for change in changes:
            for param, change_info in change.get("changes", {}).items():
                if param.startswith("changed_"):
                    param_name = param.replace("changed_", "")
                    patterns["most_changed_parameters"][param_name] = patterns["most_changed_parameters"].get(param_name, 0) + 1
            
            # Count change reasons
            reason = change.get("reason", "unknown")
            patterns["change_reasons"][reason] = patterns["change_reasons"].get(reason, 0) + 1
        
        # Sort by frequency
        patterns["most_changed_parameters"] = dict(sorted(
            patterns["most_changed_parameters"].items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return patterns
    
    def _analyze_cross_service_patterns(self, all_changes: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze patterns across all services.
        
        Args:
            all_changes: All configuration changes across services
            
        Returns:
            Dictionary with cross-service analysis
        """
        cross_service = {
            "services_with_most_changes": {},
            "common_change_times": {},
            "configuration_cascade_events": [],
            "overall_stability_metrics": {}
        }
        
        # Count changes per service
        for service_name, changes in all_changes.items():
            cross_service["services_with_most_changes"][service_name] = len(changes)
        
        # Sort by change count
        cross_service["services_with_most_changes"] = dict(sorted(
            cross_service["services_with_most_changes"].items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        # Calculate overall stability metrics
        total_changes = sum(len(changes) for changes in all_changes.values())
        total_services = len(all_changes)
        
        cross_service["overall_stability_metrics"] = {
            "total_configuration_changes": total_changes,
            "average_changes_per_service": round(total_changes / total_services, 2) if total_services > 0 else 0,
            "services_with_changes": total_services,
            "most_active_service": max(all_changes.keys(), key=lambda x: len(all_changes[x])) if all_changes else None
        }
        
        return cross_service
    
    def save_configuration_evolution_report(self, label: str):
        """
        Save the configuration evolution report to files.
        
        Args:
            label: Test run label
        """
        try:
            # Create dedicated configuration tracking directory
            config_tracking_dir = os.path.join(self.output_dir, label, "configuration_tracking")
            os.makedirs(config_tracking_dir, exist_ok=True)
            
            # Generate the evolution report
            evolution_report = self.create_configuration_evolution_report(label)
            
            if not evolution_report:
                print("❌ No evolution report generated")
                return
            
            # Save main evolution report
            evolution_file = os.path.join(config_tracking_dir, "configuration_evolution_report.json")
            with open(evolution_file, 'w') as f:
                json.dump(evolution_report, f, indent=2, default=str)
            
            print(f"📊 Configuration evolution report saved to {evolution_file}")
            
            # Save individual service evolution files
            for service_name, service_evolution in evolution_report.get("services", {}).items():
                service_file = os.path.join(config_tracking_dir, f"{service_name}_evolution.json")
                with open(service_file, 'w') as f:
                    json.dump(service_evolution, f, indent=2, default=str)
                
                print(f"📝 {service_name} evolution saved to {service_file}")
            
            # Save configuration snapshots for each service
            self._save_configuration_snapshots(label, config_tracking_dir)
            
            # Generate a human-readable summary
            self._generate_human_readable_summary(label, config_tracking_dir, evolution_report)
            
        except Exception as e:
            print(f"❌ Error saving configuration evolution report: {e}")
    
    def _save_configuration_snapshots(self, label: str, config_tracking_dir: str):
        """
        Save configuration snapshots for each service at each change point.
        
        Args:
            label: Test run label
            config_tracking_dir: Directory to save snapshots
        """
        try:
            all_changes = self.get_all_config_changes(label)
            
            for service_name, changes in all_changes.items():
                service_snapshots_dir = os.path.join(config_tracking_dir, f"{service_name}_snapshots")
                os.makedirs(service_snapshots_dir, exist_ok=True)
                
                for i, change in enumerate(changes):
                    snapshot_data = {
                        "change_number": i + 1,
                        "timestamp": change["timestamp"],
                        "reason": change.get("reason", "unknown"),
                        "configuration": change.get("new_config", {}),
                        "changes_made": change.get("changes", {}),
                        "performance_metrics": change.get("performance_metrics", {})
                    }
                    
                    snapshot_file = os.path.join(service_snapshots_dir, f"snapshot_{i+1:03d}.json")
                    with open(snapshot_file, 'w') as f:
                        json.dump(snapshot_data, f, indent=2, default=str)
                
                print(f"📸 {len(changes)} configuration snapshots saved for {service_name}")
                
        except Exception as e:
            print(f"❌ Error saving configuration snapshots: {e}")
    
    def _generate_human_readable_summary(self, label: str, config_tracking_dir: str, evolution_report: Dict[str, Any]):
        """
        Generate a human-readable summary of the configuration evolution.
        
        Args:
            label: Test run label
            config_tracking_dir: Directory to save the summary
            evolution_report: The evolution report data
        """
        try:
            summary_file = os.path.join(config_tracking_dir, "configuration_evolution_summary.txt")
            
            with open(summary_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write(f"CONFIGURATION EVOLUTION SUMMARY - {label.upper()}\n")
                f.write("=" * 80 + "\n\n")
                
                # Experiment overview
                f.write("EXPERIMENT OVERVIEW:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Label: {evolution_report.get('experiment_label', 'N/A')}\n")
                f.write(f"Generated: {evolution_report.get('generated_at', 'N/A')}\n")
                f.write(f"Total Services: {evolution_report.get('total_services', 0)}\n")
                f.write(f"Total Configuration Changes: {evolution_report.get('total_configuration_changes', 0)}\n")
                
                duration = evolution_report.get('experiment_duration', {})
                f.write(f"Duration: {duration.get('duration_minutes', 0):.1f} minutes\n")
                f.write(f"Start: {duration.get('start', 'N/A')}\n")
                f.write(f"End: {duration.get('end', 'N/A')}\n\n")
                
                # Cross-service analysis
                cross_service = evolution_report.get('cross_service_analysis', {})
                f.write("CROSS-SERVICE ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                
                stability_metrics = cross_service.get('overall_stability_metrics', {})
                f.write(f"Average Changes per Service: {stability_metrics.get('average_changes_per_service', 0)}\n")
                f.write(f"Most Active Service: {stability_metrics.get('most_active_service', 'N/A')}\n\n")
                
                f.write("Services by Change Count:\n")
                for service, count in cross_service.get('services_with_most_changes', {}).items():
                    f.write(f"  {service}: {count} changes\n")
                f.write("\n")
                
                # Individual service details
                f.write("SERVICE DETAILS:\n")
                f.write("-" * 40 + "\n")
                
                for service_name, service_evolution in evolution_report.get('services', {}).items():
                    f.write(f"\n{service_name.upper()}:\n")
                    f.write(f"  Total Changes: {service_evolution.get('total_changes', 0)}\n")
                    f.write(f"  Change Frequency: {service_evolution.get('change_frequency_minutes', 0)} minutes\n")
                    f.write(f"  First Change: {service_evolution.get('first_change', 'N/A')}\n")
                    f.write(f"  Last Change: {service_evolution.get('last_change', 'N/A')}\n")
                    
                    # Most changed parameters
                    patterns = service_evolution.get('change_patterns', {})
                    most_changed = patterns.get('most_changed_parameters', {})
                    if most_changed:
                        f.write(f"  Most Changed Parameters:\n")
                        for param, count in list(most_changed.items())[:5]:  # Top 5
                            f.write(f"    {param}: {count} times\n")
                    
                    # Change reasons
                    reasons = patterns.get('change_reasons', {})
                    if reasons:
                        f.write(f"  Change Reasons:\n")
                        for reason, count in reasons.items():
                            f.write(f"    {reason}: {count} times\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("END OF CONFIGURATION EVOLUTION SUMMARY\n")
                f.write("=" * 80 + "\n")
            
            print(f"📄 Human-readable summary saved to {summary_file}")
            
        except Exception as e:
            print(f"❌ Error generating human-readable summary: {e}")


def create_config_logger(output_dir: str = "output") -> ConfigurationLogger:
    """
    Create a new ConfigurationLogger instance.
    
    Args:
        output_dir: Base output directory for configuration logs
        
    Returns:
        ConfigurationLogger instance
    """
    return ConfigurationLogger(output_dir)
