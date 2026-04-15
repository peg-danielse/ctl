
import os
from typing import Optional
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

load_dotenv()

PATH = "."

KUBECONFIG_PATH = os.path.join(PATH, "secrets", "admin.conf")


def _cluster_host_from_kubeconfig() -> Optional[str]:
    """Return API server hostname from kubeconfig (first cluster), or None if missing/invalid."""
    path = os.path.abspath(os.path.expanduser(KUBECONFIG_PATH))
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except OSError:
        return None
    if not data or not isinstance(data.get("clusters"), list) or not data["clusters"]:
        return None
    server = (data["clusters"][0].get("cluster") or {}).get("server")
    if not server or not isinstance(server, str):
        return None
    parsed = urlparse(server)
    host = parsed.hostname
    return host


_CLUSTER_HOST = os.getenv("CLUSTER_HOST") or _cluster_host_from_kubeconfig()
if not _CLUSTER_HOST:
    raise ValueError(
        "CLUSTER_HOST is not set and could not be read from kubeconfig at "
        f"{KUBECONFIG_PATH}. Set CLUSTER_HOST in the environment or provide a valid admin.conf."
    )

CLUSTER_HOST = _CLUSTER_HOST

JAEGER_PORT = int(os.getenv("JAEGER_PORT", "30550"))
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "31207"))
LOCUST_TARGET_PORT = int(os.getenv("LOCUST_TARGET_PORT", "30505"))

JAEGER_BASE_URL = f"http://{CLUSTER_HOST}:{JAEGER_PORT}"
PROMETHEUS_BASE_URL = f"http://{CLUSTER_HOST}:{PROMETHEUS_PORT}"
LOCUST_TARGET_URL = f"http://{CLUSTER_HOST}:{LOCUST_TARGET_PORT}"

# Optional: SSH target when it differs from CLUSTER_HOST (e.g. API via VIP, SSH to a node).
SSH_REMOTE_HOST = os.getenv("SSH_REMOTE_HOST") or CLUSTER_HOST

JAEGER_ENDPOINT_FSTRING = (
    JAEGER_BASE_URL
    + "/api/traces?limit={limit}&lookback={lookback}&service={service}&start={start}"
)


GEN_API_URL = "http://localhost:4242/generate"
# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"  # GPT-5-mini has issues with content generation, using GPT-4o-mini

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"  # Fast and cost-effective model

SPAN_PROCESS_MAP = {
    # HTTP frontend handlers
    'HTTP GET /hotels': 'frontend-deployment',
    'HTTP GET /recommendations': 'frontend-deployment',
    'HTTP POST /user': 'frontend-deployment',
    'HTTP POST /reservation': 'frontend-deployment',

    # New top-level reservation handlers
    # (these are span operationNames, not HTTP paths)
    'MakeReservation': 'srv-reservation',
    'CheckAvailability': 'srv-reservation',

    # Memcached spans
    'memcached_get_profile': 'memcached-profile-deployment',
    'memcached_capacity_get_multi_number': 'memcached-reservation-deployment',
    'memcached_reserve_get_multi_number': 'memcached-reservation-deployment',
    'memcached_get_multi_rate': 'memcached-rate-deployment',
    # New memcached reservation/capacity spans
    'memcached_reservation_get': 'memcached-reservation-deployment',
    'memcached_capacity_get': 'memcached-reservation-deployment',
    'memcached_reservation_set_multi': 'memcached-reservation-deployment',

    # gRPC service spans
    '/profile.Profile/GetProfiles': 'srv-profile',
    '/search.Search/Nearby': 'srv-search',
    '/user.User/CheckUser': 'srv-user',
    '/geo.Geo/Nearby': 'srv-geo',
    '/recommendation.Recommendation/GetRecommendations': 'srv-recommendation',
    '/rate.Rate/GetRates': 'srv-rate',
    '/reservation.Reservation/CheckAvailability': 'srv-reservation',
    '/reservation.Reservation/MakeReservation': 'srv-reservation',

    # MongoDB reservation service spans (new + existing)
    'mongodb_reservation_find': 'mongodb-reservation-deployment',
    'mongodb_capacity_get': 'mongodb-reservation-deployment',
    'mongodb_reservation_insert': 'mongodb-reservation-deployment',
    'mongodb_capacity_get_multi_number': 'mongodb-reservation-deployment',

    # Legacy mongo span names (for older traces, if present)
    'mongo_rate': 'mongodb-rate-deployment',
    'mongo_user': 'mongodb-user-deployment',
    'mongo_profile': 'mongodb-profile-deployment',
    'mongo_geo': 'mongodb-geo-deployment',
    'mongo_reservation': 'mongodb-reservation-deployment',
}