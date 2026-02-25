GOAL = "Analyse the monitoring data and provide a revised configuration that aims to resolve the anomaly, respecting the constraints and using horizontal and vertical scaling and service/node affinity/anti-affinity using the provided keys and values"

GENERATE_PROMPT = '''
anomaly_report:
  service: {service_name}
  revision: {revision_name}
  
  anomaly_type: {anomaly_type}
  duration: {duration}

CONSTRAINTS (you MUST respect these):
{constraints}

NODE PLACEMENT (which services run on which nodes; use for affinity/co-location decisions):
{node_placement}

ROOT CAUSE (services with restarts or OOM killed — fix these first; e.g. if srv-rate is OOM, increase srv-rate memory, not memcached-rate):
{root_cause}
{root_cause_configs}

metric_snapshot:
```yaml
{snapshot}
```

service configuration
```yaml
{service_config}
```

global configuration:
```yaml
{auto_config}
```
'''

RESULT_PROMPT = '''The configuration produced the following performance indicators:

```yaml
{result}
```
'''


# WIP
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
    {service_config}
    {auto_config}

result_snapshot:
  {config_performance}

---

Was this configuration change effective? If not, suggest a new configuration with brief reasoning. Otherwise, explain why it worked.
'''

# WIP
FILE_PROMPT = '''give one short and concise reasoning then answer with the corrected yaml file to mitigate the anomaly: \n
<yaml> \n
{service_file} \n
--- 
{global_file} \n
</yaml>'''


