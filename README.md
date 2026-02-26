# Closing the Loop (CTL)

Adaptive configuration generator and evaluator for microservices under load. The system monitors a Kubernetes deployment (Prometheus + Jaeger), detects anomalies in trace and metric data, and uses an LLM (OpenAI or Gemini) to propose configuration changes (e.g. Knative autoscaling, resource limits). Configurations are applied to the cluster and scored.

---

## 1. Project structure

```
ctl/
├── main.py                 # Entry point: baseline/adaptation phases, loadtest + monitor threads, anomaly loop
├── config.py               # Env-based config: KUBE_*, OPENAI_*, GEMINI_*, PATH; SPAN_PROCESS_MAP (span → service)
├── prompt.py               # LLM prompt templates (GOAL, GENERATE_PROMPT, RESULT_PROMPT)
│
├── base_configuration/     # Base Kubernetes/Knative YAMLs (deployments, services, autoscaler)
│   ├── config-autoscaler.yaml
│   ├── frontend-deployment.yaml
│   ├── memcached-*-deployment.yaml
│   ├── mongodb-*-deployment.yaml
│   ├── srv-*.yaml          # Service definitions (geo, profile, rate, recommendation, reservation, search, user)
│   └── ...
│
├── knowledge/              # Domain knowledge for the LLM (autoscaling keys, constraints)
│   ├── knative_autoscaling_knowledge.yaml
│   └── knative_autoscaling_knowledge2.yaml   # Used by llm_client (KNOWLEDGE_PATH)
│
├── anomaly_detection/      # Training data for anomaly detection (Isolation Forest)
│   ├── training-set.csv    # Preprocessed trace features (preferred if present)
│   └── training_traces-2026-02-11.json      # Raw Jaeger-style traces (fallback)
│
├── util/
│   ├── config_manager.py   # Loads base_configuration, applies LLM-suggested configs via Kubernetes API
│   ├── data_retrieval.py   # DataCollector: Prometheus metrics, Jaeger traces, pod health
│   ├── analysis.py         # Anomaly detection (Isolation Forest + SHAP), metric_snapshot, read_traces
│   ├── llm_client.py       # ChatManager, generate_configuration, OpenAI/Gemini/self-hosted calls
│   ├── plot.py             # SnapshotPlotter for timeseries plots
│   ├── square.py           # Kubernetes client (KUBE_URL, KUBE_API_TOKEN), apply YAML (ConfigMap/Deployment)
│   └── locust/
│       ├── hotel-reservations.py   # Locust load test (tasks tagged: search_hotel, recommend, reserve, user_login)
│       └── weibull.py              # Load shape utilities
│
└── output/                 # Per-run output (created at runtime; gitignored)
    └── {label}/
        ├── data/           # ctl.log, snapshots.json, *traces.json, *stats*.csv, locust_stdout/stderr, plots
        └── config/        # Copy of base_configuration + applied configs for this run
```

- **base_configuration**: Defines which services exist; only these can be updated by the LLM. ConfigManager seeds `output/{label}/config/` from here.
- **knowledge**: Constrains LLM output (e.g. Knative keys, fixed-replica services, parameter ranges).
- **anomaly_detection**: Training data for the Isolation Forest model used in `util/analysis.py` to detect anomalous traces.
- **util/data_retrieval**: Hardcoded Prometheus and Jaeger base URLs; override by changing constants or passing different URLs into `DataCollector`.
- **util/square**: Uses `config.KUBE_URL` and `config.KUBE_API_TOKEN` for cluster access.

---

## 2. Running the program

### Prerequisites

- Python 3 with dependencies (e.g. `pandas`, `PyYAML`, `requests`, `kubernetes`, `scikit-learn`, `shap`, `python-dotenv`, `matplotlib, locust`).
- Access to a Kubernetes cluster (API + token), Prometheus, and Jaeger (URLs in `config.py` and `util/data_retrieval.py`).
- `.env` with API keys (see below).

### Basic run

```bash
# From the project root (ctl/)
python main.py -l my_run -t 180 -llm gemini
```

- **`-l`** (label): Experiment label; output goes to `output/{label}/` (default: `run`).
- **`-t`**: Total experiment duration in **minutes** (default: 360).
- **`-dt`**: Monitoring window in seconds (default: 60).
- **`-m`**: Measurement duration per configuration in minutes (default: 10).
- **`-s`**: Stabilization time between configurations in minutes (default: 5).
- **`-a`**: Number of anomalies to process per iteration (default: 16).
- **`-llm`**: `openai` | `gemini` | `self-hosted` (default: `openai`).
- **`--baseline`**: Run a baseline phase first (same duration, no config changes).
- **`--tags`**: Locust task tags, e.g. `--tags search_hotel recommend reserve user_login`.
- **`--verbose`**: DEBUG logging.
- **`--init` / `--no-init`**: Wait (or not) for initial configuration acceptance before starting.

Example with baseline and Gemini:

```bash
python main.py -l exp1 -t 120 --baseline -llm gemini -dt 60 -s 5 -a 16 --verbose
```

---

## 3. Applying this method to other Kubernetes systems

To port CTL to another cluster or stack you will need to:

### 3.1 Environment and API keys (`.env`)

Create a `.env` file in the project root (it is gitignored). Required for LLM-based config generation:

```env
# Kubernetes (optional if using kubeconfig; square.py uses token auth)
KUBE_API_TOKEN=your_kubernetes_api_token

# LLM providers (at least one for -llm openai or -llm gemini)
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

- **GEMINI**: Used when you run with `-llm gemini`. If `GEMINI_API_KEY` is not set, the client logs an error.
- **OPENAI**: Used when you run with `-llm openai`. Model is set in `config.py` (e.g. `OPENAI_MODEL = "gpt-4o-mini"`).

`config.py` loads `.env` via `load_dotenv()` and reads `KUBE_API_TOKEN`, `OPENAI_API_KEY`, and `GEMINI_API_KEY`.

### 3.2 Cluster and monitoring endpoints

- **Kubernetes**: In `config.py`, set `KUBE_URL` (cluster API) and ensure `KUBE_API_TOKEN` is in `.env`. `util/square.py` uses these for applying ConfigMaps and Deployments.
- **Prometheus**: In `util/data_retrieval.py`, set `PROMETHEUS_BASE_URL` to your Prometheus base URL (e.g. `http://<host>:<port>`).
- **Jaeger**: In `util/data_retrieval.py`, set `JAEGER_ENDPOINT_FSTRING` and the `base_url` inside `collect_jaeger_traces` to your Jaeger API (e.g. `http://<host>:30550/api/traces?...`).

### 3.3 Service and span mapping

- **config.py**: Update `SPAN_PROCESS_MAP` so that each Jaeger span `operationName` maps to the deployment/service name used in your `base_configuration` and in Prometheus (e.g. `'HTTP GET /hotels' → 'frontend-deployment'`, gRPC/mongo spans → corresponding srv-* or deployment names). This mapping is used to attribute traces and metrics to services.

### 3.4 Base configuration and knowledge

- **base_configuration/**: Replace or edit YAMLs so they match your cluster (same namespaces, resource names, and structure). Only services that appear here can be updated by the LLM.
- **knowledge/**: Adjust `knative_autoscaling_knowledge2.yaml` (or the file pointed to by `KNOWLEDGE_PATH` in `util/llm_client.py`) to match your autoscaling/constraint model (keys, ranges, fixed-replica list).

### 3.5 Load test

- **main.py**: The `loadtest()` function invokes Locust with a fixed host (`-H`) and options; change the host and, if needed, the script path.
- **util/locust/hotel-reservations.py**: Replace or adapt tasks and tags to your application’s endpoints so that the load test matches your services.

---

## 4. Jaeger training trace files

Anomaly detection uses an **Isolation Forest** trained on trace features (span durations, pattern). Training data is loaded in `util/analysis.py` in `get_trace_IsolationForest()`.

### 4.1 Location and format

- **Preferred**: `anomaly_detection/training-set.csv`  
  - Preprocessed CSV with numeric feature columns (e.g. one column per span duration, same schema as the trace dataframes produced by `read_traces` / `convert_trace_data_to_dataframe`).  
  - If this file exists, it is used and the JSON fallback is skipped.

- **Fallback**: `anomaly_detection/training_traces-2026-02-11.json`  
  - Raw Jaeger-style JSON: root key `"data"` containing a list of traces; each trace has `"traceID"` and `"spans"` with `"operationName"` and `"duration"` (microseconds).  
  - Processed by `read_traces()` in `util/analysis.py` into a dataframe (one row per trace, columns per operation, `startTime`, `total`), then the same feature columns as in production are used to fit the Isolation Forest.

### 4.2 How it’s used

- `get_trace_IsolationForest()` is called during anomaly detection. It fits an `IsolationForest(contamination="auto", random_state=42)` on the training features, excluding `total` if present.
- At runtime, live Jaeger traces are converted to the same dataframe format; features are aligned to the training feature list (missing columns filled with 0). Predictions of `-1` are treated as anomalies and passed to the LLM for configuration suggestions.

### 4.3 For a new system

- **Option A**: Export a representative set of “normal” traces from your Jaeger (same service/operation names as in production), save as Jaeger-style JSON, and place it at `anomaly_detection/training_traces-<date>.json`. Update the path in `util/analysis.py` (variable `json_path`) if you use a different filename.
- **Option B**: Build a CSV with the same feature columns as your runtime trace dataframe (e.g. by running `read_traces()` on a JSON export and saving the dataframe), and place it at `anomaly_detection/training-set.csv`. This will be used in preference to the JSON file.

Ensure the training data includes the same span/operation names and structure as the traces you collect in `util/data_retrieval.py`, so that feature columns match at runtime.
