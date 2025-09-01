import sys, os, json, glob, re, requests, time, subprocess, heapq, yaml, shutil

from typing import List, Tuple

from kubernetes import client, config

# experimental code, remove security warning.
import warnings
warnings.filterwarnings('ignore', message="Unverified HTTPS request*")

from config import PATH, KUBE_URL, KUBE_API_TOKEN
from .sequence import get_config_content
from ctl import load_yaml_as_dict

def update_globals(update, api_client):
    try:
        v1 = client.CoreV1Api(api_client)
        
        name = update["metadata"]["name"]
        namespace = update["metadata"].get("namespace", "knative-serving")

        # Fetch the existing ConfigMap to get the current resourceVersion
        existing_cm = v1.read_namespaced_config_map(name=name, namespace=namespace)
        update["metadata"]["resourceVersion"] = existing_cm.metadata.resource_version

        # Replace the ConfigMap
        v1.replace_namespaced_config_map(name=name, namespace=namespace, body=update)

        print(f"config '{name}' updated")

        return True
    except Exception as e:
        raise

def update_deployment(update, api_client):
    try:
        api = client.AppsV1Api(api_client)

        name = update["metadata"]["name"]
        namespace = update["metadata"].get("namespace", "default")

        response = api.patch_namespaced_deployment(
            name=name,
            namespace=namespace,
            body=update
        )

        print(f"Deployment '{name}' updated")

        return 
    except Exception as e:
        raise


def update_knative_service(update, api_client):
    try:
        api = client.CustomObjectsApi(api_client)
        
        service_name = update["metadata"]["name"]
        namespace = update["metadata"].get("namespace", "default")

        existing = api.get_namespaced_custom_object(
            group="serving.knative.dev",
            version="v1",
            namespace=namespace,
            plural="services",
            name=service_name
        )

        update["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]

        # Copy immutable annotations to avoid webhook validation error
        existing_annotations = existing["metadata"].get("annotations", {})
        new_annotations = update["metadata"].setdefault("annotations", {})

        # Preserve immutable fields
        immutable_keys = [
            "serving.knative.dev/creator",
            "serving.knative.dev/lastModifier"
        ]

        for key in immutable_keys:
            if key in existing_annotations:
                new_annotations[key] = existing_annotations[key]

        # Replace the Knative Service
        api.replace_namespaced_custom_object(
            group="serving.knative.dev",
            version="v1",
            plural="services",    # this must match the CRD plural name
            namespace=namespace,
            name=service_name,
            body=update
        )


        print(f"Service '{service_name}' updated")

        return True

    except Exception as e:
        raise
        

def apply_yaml_configuration(doc, api_client):
    match (doc["kind"]):
        case "ConfigMap":
            update_globals(doc, api_client)
        case "Service":
            update_knative_service(doc, api_client)
        case "Deployment":
            update_deployment(doc, api_client)
        case _:
            print(doc['kind'], " not supported")
    
    return


def reset_k8s(api_client, path = PATH + "/base_config"):
    files = glob.glob(path + "/*.yaml")
    for f in files:
        config = load_yaml_as_dict(f)
        
        if config is not None:
            apply_yaml_configuration(config, api_client)
        else:
            print(f"Warning: Could not load configuration from {f}")

    print("waiting to fully accept initial configuration...")
    time.sleep(60)

    return


def get_k8s_api_client():
    aConfiguration = client.Configuration()
    aConfiguration.host = KUBE_URL
    aConfiguration.verify_ssl = False
    aConfiguration.api_key = {"authorization": "Bearer " + KUBE_API_TOKEN}

    return client.ApiClient(aConfiguration)
