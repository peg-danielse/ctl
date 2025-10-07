import sys, os, json, glob, re, requests, time, subprocess, heapq, yaml, shutil

from typing import List, Tuple

from datetime import datetime
from functools import wraps
from operator import itemgetter

from config import PATH

def get_existing_seq(file_path):
    if not os.path.exists(file_path):
        return set()
    with open(file_path) as f:
        content = f.read()
    return set(map(int, re.findall(r"# --- START: seq=(.+?) ---", content)))


def append_generation(file_path, seq, content):
    if seq in get_existing_seq(file_path):
        print(f"Skipped {seq}: already exists.")
        return


    with open(file_path, "a") as f:
        f.write(f"# --- START: seq={seq} ---\n")
        f.write(content + "\n")
        f.write(f"# --- END: seq={seq} ---\n\n")

def get_config_content(file_path):
    configurations = []
    with open(file_path, 'r') as f:
        content = f.read()

        matches = re.findall(r'(?s)# --- START: seq=(\d+) ---\s*(.*?)\s*# --- END: seq=\1 ---', content)

        for seq, block in matches:
            conf, perf = block.strip().rsplit('---', 1)
            configurations.append((seq, yaml.safe_load(conf.strip()), perf.strip()))
    return configurations  

def get_generation_content(file_path):
    configurations = []
    with open(file_path, 'r') as f:
        content = f.read()

        matches = re.findall(r'(?s)# --- START: seq=(\d+) ---.*?```yaml(.*?)```.*?# --- END: seq=\1 ---', content)

        for seq, yaml in matches:
            configurations.append((seq, yaml.strip()))

    return configurations

def get_generation_content_perf(file_path):
    configurations = []
    with open(file_path, 'r') as f:
        content = f.read()

        matches = re.findall(r'(?s)# --- START: seq=(\d+) ---.*?({.*?}).*?# --- END: seq=\1 ---', content)

        for seq, yaml in matches:
            configurations.append((seq, yaml.strip()))

    return configurations

def load_generated_configurations(label, loop):
    configurations = []

    for file in glob.glob(PATH + f"/output/{label}/{label}_{loop}_prompts.json"):
        confs = get_generation_content(file)
        configurations = configurations + confs

    return configurations
