GENERATE_PROMPT = '''
{knowledge_yaml}

anomaly_report:
  service: {service_name}
  revision: {revision_name}
  timestamp: 2025-08-05T12:13:00Z

  anomaly_type: {anomaly_type}
  duration: {duration}

metric_snapshot:
{snapshot}

current_configuration:
--- service configuration
{kn_knobs}
--- global configuration
{vscale_knobs}

request: |
  Answer fully in yaml
  Provide a revised configuration for the `checkout-service`
  that aims to resolve the latency anomaly while respecting the above constraints using both horizontal and vertical scaling using the provided keys and values.
  Always end answer with the produced configuration.
  

'''

CHECK_PROMPT = '''
You are a Kubernetes autoscaling expert. Your job is to tune Knative autoscaler configurations in response to performance anomalies. The following context includes:

- An anomaly observed in a service
- The previous autoscaler configuration
- The model's suggested changes
- The performance results of applying that configuration under the same workload

Your goal is to analyze whether the configuration resolved the anomaly and suggest improvements if needed.

---
Anomaly:
  metrics_snapshot:
    {snapshot}

applied config:
  {knative_scaling}
  {v_scaling}

result_snapshot:
  {config_performance}

---

Was this configuration change effective? If not, suggest a new configuration with brief reasoning. Otherwise, explain why it worked.
'''

FILE_PROMPT = '''give one short and concise reasoning then answer with the corrected yaml file to mitigate the anomaly: \n
<yaml> \n
{service_file} \n
--- 
{global_file} \n
</yaml>'''

RESULT_PROMPT = '''the configuration produced the performance indicators: \n
<json> \n
{performance} \n
<json>'''
