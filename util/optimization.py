"""
Optimization utilities for configuration evolution and performance analysis.
Handles the evolution of configurations based on performance metrics.
"""

import math
import yaml
from pprint import pprint

from config import PATH
from util.sequence import load_generated_configurations, get_generation_content_perf, append_generation, get_existing_seq


def clean_float(value):
    """
    Clean and convert a value to float, handling edge cases.
    
    Args:
        value: Value to convert
        
    Returns:
        float: Cleaned float value
    """
    try:
        val = float(value)
        return val if not math.isnan(val) else float(1000)
    except (ValueError, TypeError):
        return float('inf')


def analyze_metrics_for_optimization(metrics_data):
    """
    Analyze current metrics to determine if optimization is needed.
    
    Args:
        metrics_data: Dictionary containing current metrics from continuous monitoring
        
    Returns:
        bool: True if optimization is needed, False otherwise
    """
    if not metrics_data or 'metrics' not in metrics_data:
        return False
    
    metrics = metrics_data['metrics']
    
    # Simple heuristic: check if any service has high CPU usage or pod count issues
    for config_name, config_metrics in metrics.items():
        cpu_usage = config_metrics.get('container_cpu_usage_seconds_total', 0)
        actual_pods = config_metrics.get('autoscaler_actual_pods', 0)
        desired_pods = config_metrics.get('autoscaler_desired_pods', 0)
        
        # Trigger optimization if:
        # 1. CPU usage is high (>0.8 cores)
        # 2. Actual pods are significantly different from desired pods
        if cpu_usage > 0.8 or abs(actual_pods - desired_pods) > 2:
            print(f"Optimization trigger for {config_name}: CPU={cpu_usage}, actual_pods={actual_pods}, desired_pods={desired_pods}")
            return True
    
    return False


def evolve_configurations(label, loop):
    """
    Evolve configurations based on performance metrics.
    Selects the best performing configurations and applies them.
    
    Args:
        label: Experiment label
        loop: Loop number
        
    Returns:
        dict: Best performing configurations
    """
    # Load generated configurations
    configs = load_generated_configurations(label, loop)
    configs = [(i, yaml.safe_load_all(sol)) for i, sol in configs]

    # Load performance data
    trails = get_generation_content_perf(PATH + f"/output/{label}/{label}_{loop}_performance.json")
    perfs = [(i, yaml.safe_load(p)) for i, p in trails]
    perfs = [(i, p) for i, p in perfs if "Audit-Id" not in p]

    # Filter configurations with minimum throughput
    perfs = [(i, p) for i, p in perfs if 300 <= p.get('max_throughput', 0)]

    perf_dict = dict(perfs)
    
    # Filter configs to include only those with corresponding performance data
    merged = [
        (i, config, perf_dict[i])
        for i, config in configs
        if i in perf_dict
    ]

    # Sort configurations by performance metrics (lower is better for most metrics)
    sorted_data = sorted(
        merged,
        key=lambda item: (
            clean_float(item[2].get('total_anomaly_count')),
            clean_float(item[2].get('changed_service_anomaly_count')),
            clean_float(item[2].get('total_errors')),
            clean_float(item[2].get('service_max_cpu_usage')),
            clean_float(item[2].get('total_max_cpu_usage')),
        )
    )

    # Select best configurations
    best = {}
    for p in reversed(sorted_data):
        name = []
        docs = []
        for d in p[1]:
            name.append(d.get('metadata', {}).get('name'))
            docs.append(d)
        
        for n in name:
            best[n] = (p[0], docs, p[2])

    # Apply best configurations
    for service_name, (s, c, p) in best.items():
        fp = f"{PATH}/output/{label}/config/{service_name}.yaml"
        print(f"Update configuration for {service_name} \n performance: {p} \n file: {fp}")
        for d in c:
            if service_name == d.get('metadata', {}).get('name'):
                append_generation(fp, max(get_existing_seq(fp)) + 1, yaml.dump(d) + "\n---\n" + str(p))

    return best


def select_best_configurations(performance_data, criteria=None):
    """
    Select the best configurations based on performance criteria.
    
    Args:
        performance_data: List of (config_id, performance_metrics) tuples
        criteria: Dictionary of criteria weights (optional)
        
    Returns:
        list: Sorted list of best configurations
    """
    if criteria is None:
        criteria = {
            'total_anomaly_count': 1.0,
            'changed_service_anomaly_count': 1.0,
            'total_errors': 1.0,
            'service_max_cpu_usage': 0.5,
            'total_max_cpu_usage': 0.3
        }
    
    def score_config(config_data):
        """Calculate a score for a configuration based on criteria."""
        config_id, metrics = config_data
        score = 0.0
        
        for metric, weight in criteria.items():
            value = clean_float(metrics.get(metric, 0))
            score += value * weight
        
        return score
    
    # Sort by score (lower is better)
    return sorted(performance_data, key=score_config)
