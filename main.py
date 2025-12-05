import argparse
import glob
import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import queue
import traceback

from datetime import datetime, timedelta, timezone

import pandas as pd
import yaml

from util.config_manager import ConfigManager
from util.data_retrieval import DataCollector
from util.llm_client import ChatManager, generate_prompt, task_score_configuration
from util.plot import SnapshotPlotter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('threading_the_needle.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def monitor(duration: int, ratio: int, label: str, anomaly_queue: queue.Queue, phase_queue: queue.Queue):
    duration_seconds = duration
    interval_seconds = ratio
    phase = "initialization"
    subphase = "collection_start"
    snapshot_json = f"./output/{label}/data/snapshots.json"

    collector = DataCollector.get_instance()

    start_overall = datetime.now().astimezone(timezone.utc)
    deadline = start_overall + timedelta(seconds=duration_seconds)
    current_start = start_overall

    while current_start < deadline:
        planned_end = current_start + timedelta(seconds=interval_seconds)
        # Ensure we don't run past the deadline
        end_time = min(planned_end, deadline)

        # Calculate progress
        elapsed_minutes = (current_start - start_overall).total_seconds() / 60
        remaining_minutes = (deadline - current_start).total_seconds() / 60
        total_minutes = duration_seconds / 60
        
        logger.info(f"Collecting window: {current_start} -> {end_time} | Progress: {remaining_minutes:.1f}min left / {total_minutes:.1f}min total ({elapsed_minutes:.1f}min elapsed)")

        # If the planned window end is in the future, sleep until then for stable windows
        now = datetime.now().astimezone(timezone.utc)
        if end_time > now:
            sleep_seconds = (end_time - now).total_seconds()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        logger.debug(f"after sleep: {datetime.now().astimezone(timezone.utc)}")

        # Finalize end_time after sleep to align with planned window
        end_time = min(datetime.now().astimezone(timezone.utc), planned_end, deadline)

        try:
            if not phase_queue.empty():
                phase, subphase = phase_queue.get(timeout=0.2)
            
            # logger.debug(f"phase: {phase}, subphase: {subphase}")
            snapshot, s_trace_df, s_individual_metrics_dfs, s_anomaly_results = collector.log_all_data_to_csv(current_start, end_time, label, snapshot_json, phase, subphase)

            for service_name, anomalies_for_service in s_anomaly_results['anomalies_by_service'].items():
                for anomaly in anomalies_for_service:
                    # logger.debug(f"Anomaly: {anomaly}")

                    start_ts = anomaly['timestamp']
                    end_ts = start_ts + pd.to_timedelta(anomaly['duration_seconds'], unit='s')

                    trace_window = s_trace_df[(s_trace_df['startTime'] >= start_ts) &
                                            (s_trace_df['startTime'] <= end_ts)]

                    metrics_df = s_individual_metrics_dfs.get(service_name)
                    if metrics_df is not None and not metrics_df.empty:
                        start_epoch = start_ts.timestamp()
                        end_epoch = end_ts.timestamp()
                        metrics_window = metrics_df[(metrics_df['index'] >= start_epoch) &
                                                    (metrics_df['index'] <= end_epoch)]
                    else:
                        metrics_window = metrics_df

                    # Use a unique counter to ensure no comparison issues in priority queue
                    unique_priority = (anomaly.get('duration_seconds'), time.time(), {
                        "service_name": service_name,
                        "trace": trace_window,
                        "metrics": metrics_window,
                        "anomaly": anomaly,        # keep as-is to match current consumer usage
                        "anomaly_index": anomaly.get('index'),
                        "anomalies": [anomaly],    # optional if your consumer expects this
                        "knobs": {}                # fill if you have knob info
                    })
                    anomaly_queue.put(unique_priority)         
           
            if os.path.exists(f"./output/{label}/data/snapshots.json"):
                plotter = SnapshotPlotter()

                snapshots = json.loads(open(f"./output/{label}/data/snapshots.json", "r").read())
                # print("snapshots: ", len(snapshots))

                if len(snapshots) > 2:
                    plotter.plot_all_timeseries(label)
            else:
                logger.info(f"snapshots.json not READY for {label}, skipping plot")

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error during collection window {current_start} -> {end_time}: {e}")

        # Advance to next window
        current_start = end_time

    logger.info(f"Monitor completed after {duration_seconds / 60} minutes")
    
    files = glob.glob(f"./{label}*.csv")
    os.makedirs(f"./output/{label}/data", exist_ok=True)
    for file in files:
        shutil.move(file, f"./output/{label}/data/")

def simple_hash(s):
    """Generate a simple hash for seed generation."""
    hash_object = hashlib.sha256(s.encode())
    hex_digest = hash_object.hexdigest()
    int_hash = int(hex_digest, 16)
    return int_hash % 10000

def loadtest(duration, label, phase):
    logger.info(f"Pressure test {label}_{phase} for {duration} seconds")

    try:
        cmd = [
            "locust",
            "--processes", "32",
            "-f", "./util/locust/hotel-reservations.py",
            "-H", "http://145.100.135.11:30505",
            "-t", str(duration) + "s", 
            "--csv", label,
            "--headless",
            "--w-user-min", str(1000),
            "--w-user-max", str(4000),
            "--w-mean", str(1200),
            "--w-ls-y", str(2000),
            # "--w-dt", str(20),
            "--seed", str(42), # simple_hash(f"fix")
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the process to complete
        stdout, stderr = process.communicate(timeout=duration) 
        
        # Save stdout and stderr to files
        os.makedirs(f"./output/{label}/data", exist_ok=True)
        with open(f"./output/{label}/data/locust_stdout.txt", "w") as f:
            f.write(stdout)
        with open(f"./output/{label}/data/locust_stderr.txt", "w") as f:
            f.write(stderr)

        # Move experiment files to output directory
        experiment_files = glob.glob(f"./{label}_*")
        os.makedirs(f"./output/{label}/data", exist_ok=True)
        for file in experiment_files:
            shutil.move(file, f"./output/{label}/data/")
        
    except subprocess.TimeoutExpired:
        logger.warning(f"Async load test {label} timed out")
        
        process.kill()
        
    except Exception as e:
        logger.error(f"Error in load test {label}: {e}")

    logger.info(f"LOADTEST: completed after {duration} seconds")


def main():
    parser = argparse.ArgumentParser(
        prog="closing the loop",
        description="Adaptive configuration generator and evaluator for microservices under load."
    )
    parser.add_argument("-l", type=str, default="run", help="Label for this experiment run (used for output folders).")
    parser.add_argument("-t", type=int, default=30, help="Total experiment duration in minutes.")
    parser.add_argument("-dt", type=int, default=20, help="Sampling window in seconds for monitoring.")
    parser.add_argument("-m", type=int, default=10, help="Measurement duration per configuration in minutes.")
    parser.add_argument("-s", type=int, default=5, help="Stabilization time between configurations in minutes.")
    parser.add_argument("-a", type=int, default=16, help="Number of anomalies to process per iteration.")
    parser.add_argument("-llm", type=str, default="openai", choices=["openai", "gemini", "self-hosted"], help="LLM to use for configuration generation.")
    parser.add_argument("--baseline", action="store_true", help="Run an initial baseline phase before adaptation.")
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument("--init", dest="init", action="store_true", help="Wait to accept initial configuration")
    init_group.add_argument("--no-init", dest="init", action="store_false", help="Do not wait for initial configuration acceptance")
    parser.set_defaults(init=True)

    args = parser.parse_args()

    label = args.l # label for the experiment
    duration = args.t * 60 # seconds
    ratio = args.dt # seconds
    
    stabilization_time = args.s # minutes
    measurement_time = args.m # minutes
    
    anomalies_per_iteration = args.a # number of anomalies per iteration

    baseline = args.baseline # boolean to run the baseline experiment

    # Initialize config manager
    config_manager = ConfigManager(label)
    
    if args.init:
        logger.info("Waiting to fully accept initial configuration...")
        time.sleep(2 * 60)
    
    # Initialize chat manager
    chat_manager = ChatManager.get_instance()

    # Create queue for anomaly communication
    anomaly_queue = queue.PriorityQueue()
    phase_queue = queue.Queue()
    
    # Initialize threads
    loadtest_thread = None
    monitoring_thread = None

    # Start baseline phase
    if baseline:
        logger.info(f"starting baseline phase")
        loadtest_thread = threading.Thread(target=loadtest, args=(duration, label+"_baseline", "baseline"))
        loadtest_thread.start()
    
        phase_queue.put(("baseline", "normal"))
        monitoring_thread = threading.Thread(target=monitor, args=(duration, ratio, label+"_baseline", anomaly_queue, phase_queue), daemon=True)
        monitoring_thread.start()

        while loadtest_thread.is_alive() and monitoring_thread.is_alive():
            logger.info(f"waiting for anomalies to be collected: [{anomaly_queue.qsize()}]")
            if anomaly_queue.qsize() < anomalies_per_iteration:
                time.sleep(ratio)
                continue
            
            logger.info(f"collected anomalies: {anomaly_queue.qsize()}")
            
            # Clear the anomaly queue
            with anomaly_queue.mutex:
                anomaly_queue.queue.clear()

        loadtest_thread.join()
        monitoring_thread.join()
        logger.info(f"baseline phase completed")
    
    
    # Start adaptation phase
    logger.info(f"starting adaptation phase")

    # Start pressure test thread
    loadtest_thread = threading.Thread(target=loadtest, args=(duration, label, "adaptation"))
    loadtest_thread.start()

    # Start monitoring thread
    monitoring_thread = threading.Thread(target=monitor, args=(duration, ratio, label, anomaly_queue, phase_queue), daemon=True)
    phase_queue.put(("adaptation", "normal"))
    monitoring_thread.start()

    # Monitor while pressure test runs
    while loadtest_thread.is_alive() and monitoring_thread.is_alive():
        logger.info(f"waiting for anomalies to be collected: [{anomaly_queue.qsize()}]")
        if anomaly_queue.qsize() < anomalies_per_iteration:
            time.sleep(2)
            continue

        logger.info(f"collected anomalies: {anomaly_queue.qsize()}")
        # For each anomaly, generate a configuration using LLM
        
        dirty_store: dict[str, threading.Thread] = {}
        for i in range(anomalies_per_iteration):
            try:
                anomaly = anomaly_queue.get(timeout=0.2)[2] # (priority, timestamp, item) => item
                
                if anomaly["service_name"] in dirty_store:
                    # logger.info(f"dirty_store: {dirty_store}")
                    logger.info(f"already generated configuration for {anomaly['service_name']}")
                    continue

                prompt = generate_prompt(anomaly["service_name"], 
                                        anomaly["trace"], 
                                        anomaly["metrics"], 
                                        anomaly["anomalies"],  
                                        label)

                configuration_updates = chat_manager.generate_configuration(prompt, anomaly["service_name"], args.llm)

                logger.info(f"configuration_updates: {configuration_updates}")

                if configuration_updates is None:
                    continue

                logger.info(f"setting service config for {anomaly['service_name']}")
                for configuration_update in yaml.safe_load_all(configuration_updates):
                    config_manager.set_service_config(configuration_update.get('metadata').get('name'), configuration_update)

                    # score the configuration
                    dirty_store[configuration_update.get('metadata').get('name')] = threading.Thread(target=task_score_configuration, 
                                                                            args=(measurement_time, configuration_update, anomaly["service_name"], label))
            
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error during anomaly {anomaly['anomaly_index']}: {e}")
                continue

        # save all configs
        config_manager.save_all_configs()

        # wait to stabilize the system
        phase_queue.put(("adaptation", "stabilization"))
        time.sleep(stabilization_time * 60) 

        for service_name in dirty_store:
            dirty_store[service_name].start()

        phase_queue.put(("adaptation", "normal"))

        # wait to perform scoring of the configuration
        for service_name in dirty_store:
            dirty_store[service_name].join()

        # Clear the anomaly queue
        with anomaly_queue.mutex:
            anomaly_queue.queue.clear()

    logger.info(f"adaptation phase completed")
    chat_manager.save_all_chats(label)

    # Wait for busy wait to complete
    logger.info("waiting to complete")
    loadtest_thread.join()
    monitoring_thread.join()

if __name__ == "__main__":
    main()
