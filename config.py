
import os
from dotenv import load_dotenv

load_dotenv()

PATH = "."

KUBE_URL = "https://localhost:6443" 
KUBE_API_TOKEN = os.getenv("KUBE_API_TOKEN")


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