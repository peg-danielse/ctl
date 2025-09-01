
import os
from dotenv import load_dotenv

load_dotenv()

PATH = "."

KUBE_URL = "https://localhost:6443" 
KUBE_API_TOKEN = os.getenv("KUBE_API_TOKEN")

GEN_API_URL = "http://localhost:4242/generate"

SPAN_PROCESS_MAP = {'HTTP GET /hotels': 'frontend-deployment',
'HTTP GET /recommendations': 'frontend-deployment',
'HTTP POST /user': 'frontend-deployment',
'HTTP POST /reservation': 'frontend-deployment',

'memcached_get_profile': 'memcached-profile-deployment',
'memcached_capacity_get_multi_number': 'memcached-reservation-deployment',
'memcached_reserve_get_multi_number': 'memcached-reservation-deployment',
'memcached_get_multi_rate': 'memcached-rate-deployment',

'/profile.Profile/GetProfiles': 'srv-profile',
'/search.Search/Nearby': 'srv-search',
'/user.User/CheckUser': 'srv-user',
'/geo.Geo/Nearby': 'srv-geo',
'/recommendation.Recommendation/GetRecommendations': 'srv-recommendation',
'/rate.Rate/GetRates': 'srv-rate',
'/reservation.Reservation/CheckAvailability': 'srv-reservation',
'/reservation.Reservation/MakeReservation': 'srv-reservation',
'mongo_rate': 'mongo_rate'}