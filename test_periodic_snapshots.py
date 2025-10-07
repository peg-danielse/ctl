#!/usr/bin/env python3
"""
Test script to create periodic snapshots from brocken data.
Uses the project's data import methods and creates snapshots at regular intervals.
"""

import json
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
sys.path.append('.')

from util.analysis import read_traces, read_metrics, read_response, metric_snapshot

def create_periodic_snapshots():
    """
    Create periodic snapshots from brocken data.
    """
    print("=" * 80)
    print("PERIODIC SNAPSHOT TEST - BROCKEN DATA")
    print("=" * 80)
    
    # Load data using project methods
    print("\n1. Loading data using project methods...")
    
    try:
        # Load traces
        trace_df = read_traces('brocken', 'brocken')
        print(f"   ✓ Traces loaded: {len(trace_df)} rows")
        
        # Load metrics
        metric_dfs = read_metrics('brocken', 'brocken')
        print(f"   ✓ Metrics loaded: {len(metric_dfs)} services")
        
        # Load response data
        response_df = read_response('brocken', 'brocken')
        print(f"   ✓ Response loaded: {len(response_df)} rows")
        
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return
    
    # Analyze time range
    print("\n2. Analyzing time range...")
    
    if trace_df.empty:
        print("   ✗ No trace data available")
        return
    
    # Convert startTime to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(trace_df['startTime']):
        trace_df['startTime'] = pd.to_datetime(trace_df['startTime'])
    
    start_time = trace_df['startTime'].min()
    end_time = trace_df['startTime'].max()
    total_duration = end_time - start_time
    
    print(f"   Start time: {start_time}")
    print(f"   End time: {end_time}")
    print(f"   Total duration: {total_duration}")
    
    # Create time windows for snapshots
    print("\n3. Creating time windows for snapshots...")
    
    # Create 5 time windows of 2 minutes each, overlapping by 30 seconds
    window_duration = timedelta(minutes=2)
    overlap = timedelta(seconds=30)
    num_windows = 5
    
    time_windows = []
    current_start = start_time
    
    for i in range(num_windows):
        current_end = current_start + window_duration
        if current_end > end_time:
            current_end = end_time
        
        time_windows.append({
            'window_id': i + 1,
            'start_time': current_start,
            'end_time': current_end,
            'duration': current_end - current_start
        })
        
        print(f"   Window {i+1}: {current_start} to {current_end} ({current_end - current_start})")
        
        # Move to next window with overlap
        current_start = current_end - overlap
        if current_start >= end_time:
            break
    
    # Create snapshots for each time window
    print("\n4. Creating snapshots for each time window...")
    
    snapshots = []
    
    for window in time_windows:
        print(f"\n   Processing Window {window['window_id']}...")
        
        # Filter trace data for this time window
        window_traces = trace_df[
            (trace_df['startTime'] >= window['start_time']) & 
            (trace_df['startTime'] <= window['end_time'])
        ].copy()
        
        print(f"     Traces in window: {len(window_traces)}")
        
        if window_traces.empty:
            print(f"     ⚠ No traces in this window, skipping...")
            continue
        
        # Create snapshot using metric_snapshot function
        try:
            timestamp, duration, snapshot = metric_snapshot(
                f"brocken_window_{window['window_id']}", 
                window_traces, 
                metric_dfs,
                phase="test", subphase="periodic_test"
            )
            
            # Add window metadata
            snapshot['window_info'] = {
                'window_id': window['window_id'],
                'start_time': window['start_time'].isoformat(),
                'end_time': window['end_time'].isoformat(),
                'duration_seconds': window['duration'].total_seconds(),
                'trace_count': len(window_traces)
            }
            
            snapshots.append(snapshot)
            
            # Print summary
            print(f"     ✓ Snapshot created:")
            print(f"       - Total requests: {snapshot['total_requests']}")
            print(f"       - Mean response time: {snapshot['mean_response_time']/1000:.1f}ms")
            print(f"       - P90 response time: {snapshot['p90_response_time']/1000:.1f}ms")
            print(f"       - Miss rate: {snapshot['deadline_miss_rate']:.1f}%")
            print(f"       - CPU utilization: {snapshot['cpu_utilization']:.2f}")
            print(f"       - Anomalies detected: {snapshot['anomaly_count']}")
            print(f"       - Anomaly rate: {snapshot['anomaly_rate']:.2f}%")
            print(f"       - Patterns analyzed: {len(snapshot['patterns'])}")
            
            # Show top patterns
            if snapshot['patterns']:
                sorted_patterns = sorted(snapshot['patterns'].items(), 
                                       key=lambda x: x[1]['deadline_miss_rate'], reverse=True)
                top_pattern = sorted_patterns[0]
                print(f"       - Top pattern: {top_pattern[0]} ({top_pattern[1]['deadline_miss_rate']:.1f}% miss rate)")
            
            # Show top services with anomalies
            if snapshot['services']:
                sorted_services = sorted(snapshot['services'].items(), 
                                       key=lambda x: x[1]['anomaly_count'], reverse=True)
                if sorted_services:
                    top_service = sorted_services[0]
                    print(f"       - Top anomalous service: {top_service[0]} ({top_service[1]['anomaly_count']} anomalies)")
            
        except Exception as e:
            print(f"     ✗ Error creating snapshot: {e}")
            continue
    
    # Save snapshots to file
    print(f"\n5. Saving {len(snapshots)} snapshots to file...")
    
    output_file = "output/brocken/periodic_snapshots_test.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(snapshots, f, indent=2, default=str)
        
        print(f"   ✓ Snapshots saved to: {output_file}")
        
        # Print file size
        file_size = os.path.getsize(output_file)
        print(f"   ✓ File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
    except Exception as e:
        print(f"   ✗ Error saving snapshots: {e}")
        return
    
    # Print summary statistics
    print(f"\n6. Summary Statistics:")
    print(f"   Total snapshots created: {len(snapshots)}")
    
    if snapshots:
        total_requests = sum(s['total_requests'] for s in snapshots)
        total_anomalies = sum(s['anomaly_count'] for s in snapshots)
        avg_response_time = sum(s['mean_response_time'] for s in snapshots) / len(snapshots)
        avg_miss_rate = sum(s['deadline_miss_rate'] for s in snapshots) / len(snapshots)
        avg_cpu_utilization = sum(s['cpu_utilization'] for s in snapshots) / len(snapshots)
        avg_anomaly_rate = sum(s['anomaly_rate'] for s in snapshots) / len(snapshots)
        
        print(f"   Total requests across all windows: {total_requests:,}")
        print(f"   Total anomalies detected: {total_anomalies}")
        print(f"   Average response time: {avg_response_time/1000:.1f}ms")
        print(f"   Average miss rate: {avg_miss_rate:.1f}%")
        print(f"   Average CPU utilization: {avg_cpu_utilization:.2f}")
        print(f"   Average anomaly rate: {avg_anomaly_rate:.2f}%")
        
        # Show pattern analysis across all windows
        all_patterns = {}
        for snapshot in snapshots:
            for pattern, stats in snapshot['patterns'].items():
                if pattern not in all_patterns:
                    all_patterns[pattern] = {'total_requests': 0, 'total_misses': 0, 'windows': 0}
                # We need to estimate total_requests and miss_count from the pattern data
                # Since we don't have these in the new structure, we'll use the pattern stats
                all_patterns[pattern]['windows'] += 1
        
        if all_patterns:
            print(f"\n   Pattern Analysis (across all windows):")
            sorted_patterns = sorted(all_patterns.items(), key=lambda x: x[1]['windows'], reverse=True)
            for pattern, stats in sorted_patterns[:5]:
                print(f"     {pattern}: {stats['windows']} windows")
    
    print(f"\n✅ Test completed successfully!")
    print(f"📁 Output file: {output_file}")

if __name__ == "__main__":
    create_periodic_snapshots()
