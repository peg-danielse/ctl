#!/usr/bin/env python3
"""
Script to analyze and display summary of the periodic snapshots.
"""

import json
import sys
import os

def analyze_snapshots():
    """
    Analyze the periodic snapshots and display a summary.
    """
    snapshot_file = "output/brocken/periodic_snapshots_test.json"
    
    if not os.path.exists(snapshot_file):
        print(f"❌ Snapshot file not found: {snapshot_file}")
        return
    
    print("=" * 80)
    print("PERIODIC SNAPSHOTS ANALYSIS")
    print("=" * 80)
    
    # Load snapshots
    with open(snapshot_file, 'r') as f:
        snapshots = json.load(f)
    
    print(f"\n📊 OVERVIEW")
    print(f"   Total snapshots: {len(snapshots)}")
    
    # Analyze each snapshot
    print(f"\n📈 SNAPSHOT DETAILS")
    for i, snapshot in enumerate(snapshots, 1):
        window_info = snapshot.get('window_info', {})
        patterns = snapshot.get('patterns', {})
        services = snapshot.get('services', {})
        
        print(f"\n   Window {i}:")
        print(f"     Time: {window_info.get('start_time', 'N/A')} to {window_info.get('end_time', 'N/A')}")
        print(f"     Duration: {window_info.get('duration_seconds', 0):.0f}s")
        print(f"     Traces: {window_info.get('trace_count', 0)}")
        print(f"     Requests: {snapshot.get('total_requests', 0)}")
        print(f"     Mean Response Time: {snapshot.get('mean_response_time', 0)/1000:.1f}ms")
        print(f"     P90 Response Time: {snapshot.get('p90_response_time', 0)/1000:.1f}ms")
        print(f"     Miss Rate: {snapshot.get('deadline_miss_rate', 0):.1f}%")
        print(f"     CPU Utilization: {snapshot.get('cpu_utilization', 0):.2f}")
        print(f"     Anomalies: {snapshot.get('anomaly_count', 0)}")
        print(f"     Anomaly Rate: {snapshot.get('anomaly_rate', 0):.2f}%")
        print(f"     Patterns: {len(patterns)}")
        print(f"     Phase: {snapshot.get('phase', 'N/A')}")
        print(f"     Subphase: {snapshot.get('subphase', 'N/A')}")
        
        # Show top patterns
        if patterns:
            sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['deadline_miss_rate'], reverse=True)
            top_pattern = sorted_patterns[0]
            print(f"     Top Pattern: {top_pattern[0]} ({top_pattern[1]['deadline_miss_rate']:.1f}% miss rate)")
        
        # Show top anomalous services
        if services:
            sorted_services = sorted(services.items(), key=lambda x: x[1]['anomaly_count'], reverse=True)
            if sorted_services:
                top_service = sorted_services[0]
                print(f"     Top Anomalous Service: {top_service[0]} ({top_service[1]['anomaly_count']} anomalies)")
    
    # Aggregate analysis
    print(f"\n🔍 AGGREGATE ANALYSIS")
    
    total_requests = sum(s.get('total_requests', 0) for s in snapshots)
    total_anomalies = sum(s.get('anomaly_count', 0) for s in snapshots)
    
    if snapshots:
        avg_response_time = sum(s.get('mean_response_time', 0) for s in snapshots) / len(snapshots)
        avg_p90_response_time = sum(s.get('p90_response_time', 0) for s in snapshots) / len(snapshots)
        avg_miss_rate = sum(s.get('deadline_miss_rate', 0) for s in snapshots) / len(snapshots)
        avg_cpu_utilization = sum(s.get('cpu_utilization', 0) for s in snapshots) / len(snapshots)
        avg_anomaly_rate = sum(s.get('anomaly_rate', 0) for s in snapshots) / len(snapshots)
        
        print(f"   Total Requests: {total_requests:,}")
        print(f"   Total Anomalies: {total_anomalies}")
        print(f"   Average Response Time: {avg_response_time/1000:.1f}ms")
        print(f"   Average P90 Response Time: {avg_p90_response_time/1000:.1f}ms")
        print(f"   Average Miss Rate: {avg_miss_rate:.1f}%")
        print(f"   Average CPU Utilization: {avg_cpu_utilization:.2f}")
        print(f"   Average Anomaly Rate: {avg_anomaly_rate:.2f}%")
        print(f"   Overall Anomaly Rate: {total_anomalies/total_requests*100:.2f}%")
    
    # Pattern analysis across all windows
    print(f"\n🎯 PATTERN ANALYSIS")
    all_patterns = {}
    for snapshot in snapshots:
        for pattern, stats in snapshot.get('patterns', {}).items():
            if pattern not in all_patterns:
                all_patterns[pattern] = {'total_requests': 0, 'total_misses': 0, 'windows': 0}
            all_patterns[pattern]['total_requests'] += stats.get('total_requests', 0)
            all_patterns[pattern]['total_misses'] += stats.get('miss_count', 0)
            all_patterns[pattern]['windows'] += 1
    
    if all_patterns:
        sorted_patterns = sorted(all_patterns.items(), key=lambda x: x[1]['total_requests'], reverse=True)
        print(f"   Patterns found: {len(all_patterns)}")
        for pattern, stats in sorted_patterns:
            overall_miss_rate = (stats['total_misses'] / stats['total_requests'] * 100) if stats['total_requests'] > 0 else 0
            print(f"     {pattern}: {stats['total_requests']} requests, {overall_miss_rate:.1f}% miss rate, {stats['windows']} windows")
    
    # Service anomaly analysis
    print(f"\n🚨 SERVICE ANOMALY ANALYSIS")
    all_service_anomalies = {}
    for snapshot in snapshots:
        service_counts = snapshot.get('shap_anomaly_detection', {}).get('service_anomaly_counts', {})
        for service, count in service_counts.items():
            all_service_anomalies[service] = all_service_anomalies.get(service, 0) + count
    
    if all_service_anomalies:
        sorted_services = sorted(all_service_anomalies.items(), key=lambda x: x[1], reverse=True)
        print(f"   Services with anomalies: {len(all_service_anomalies)}")
        for service, count in sorted_services:
            print(f"     {service}: {count} anomalies")
    else:
        print("   No service anomalies detected")
    
    # Time series analysis
    print(f"\n⏰ TIME SERIES ANALYSIS")
    response_times = []
    miss_rates = []
    anomaly_counts = []
    
    for snapshot in snapshots:
        overall_stats = snapshot.get('overall_pattern_stats', {})
        shap_results = snapshot.get('shap_anomaly_detection', {})
        
        response_times.append(overall_stats.get('mean_response_time', 0) / 1000)  # Convert to ms
        miss_rates.append(overall_stats.get('miss_rate', 0))
        anomaly_counts.append(shap_results.get('anomaly_count', 0))
    
    if response_times:
        print(f"   Response Time Trend: {response_times[0]:.1f}ms → {response_times[-1]:.1f}ms")
        print(f"   Miss Rate Trend: {miss_rates[0]:.1f}% → {miss_rates[-1]:.1f}%")
        print(f"   Anomaly Count Trend: {anomaly_counts[0]} → {anomaly_counts[-1]}")
        
        # Calculate trends
        response_trend = "📈 Increasing" if response_times[-1] > response_times[0] else "📉 Decreasing" if response_times[-1] < response_times[0] else "➡️ Stable"
        miss_trend = "📈 Increasing" if miss_rates[-1] > miss_rates[0] else "📉 Decreasing" if miss_rates[-1] < miss_rates[0] else "➡️ Stable"
        anomaly_trend = "📈 Increasing" if anomaly_counts[-1] > anomaly_counts[0] else "📉 Decreasing" if anomaly_counts[-1] < anomaly_counts[0] else "➡️ Stable"
        
        print(f"   Response Time Trend: {response_trend}")
        print(f"   Miss Rate Trend: {miss_trend}")
        print(f"   Anomaly Count Trend: {anomaly_trend}")
    
    print(f"\n✅ Analysis completed!")
    print(f"📁 Full data available in: {snapshot_file}")

if __name__ == "__main__":
    analyze_snapshots()
