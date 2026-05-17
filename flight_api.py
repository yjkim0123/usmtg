"""
USMTG Flight API — Amadeus real-price integration

Usage:
  1. Get free key at https://developers.amadeus.com  (Self-Service tier, free sandbox)
  2. Set env vars:  AMADEUS_KEY=<API Key>  AMADEUS_SECRET=<API Secret>
  3. Import and call get_fare(dep, arr, date) instead of the ML estimate

Sandbox returns test data (fixed prices per route).
Production requires upgrading the Amadeus account (still free up to 2000 calls/month).
"""
import os, time, math
import requests

_AMADEUS_TOKEN = None
_TOKEN_EXPIRES = 0

_KEY    = os.getenv('AMADEUS_KEY',    '')
_SECRET = os.getenv('AMADEUS_SECRET', '')

AMADEUS_BASE = 'https://test.api.amadeus.com'   # swap to api.amadeus.com for production


def _get_token():
    global _AMADEUS_TOKEN, _TOKEN_EXPIRES
    if _AMADEUS_TOKEN and time.time() < _TOKEN_EXPIRES - 60:
        return _AMADEUS_TOKEN
    r = requests.post(
        f'{AMADEUS_BASE}/v1/security/oauth2/token',
        data={'grant_type': 'client_credentials',
              'client_id': _KEY, 'client_secret': _SECRET},
        timeout=10
    )
    r.raise_for_status()
    d = r.json()
    _AMADEUS_TOKEN = d['access_token']
    _TOKEN_EXPIRES = time.time() + d['expires_in']
    return _AMADEUS_TOKEN


def get_fare(dep_iata: str, arr_iata: str, date: str, adults: int = 1) -> float | None:
    """
    Fetch cheapest one-way fare via Amadeus Flight Offers Search.

    Args:
        dep_iata: departure IATA code  e.g. 'JFK'
        arr_iata: arrival IATA code    e.g. 'LAX'
        date:     ISO date string      e.g. '2026-06-01'
        adults:   number of passengers

    Returns:
        cheapest fare in USD, or None if unavailable / no key configured.
    """
    if not _KEY:
        return None
    try:
        token = _get_token()
        r = requests.get(
            f'{AMADEUS_BASE}/v2/shopping/flight-offers',
            headers={'Authorization': f'Bearer {token}'},
            params={
                'originLocationCode':      dep_iata,
                'destinationLocationCode': arr_iata,
                'departureDate':           date,
                'adults':                  adults,
                'max':                     5,
                'currencyCode':            'USD',
                'nonStop':                 'false',
            },
            timeout=15
        )
        r.raise_for_status()
        offers = r.json().get('data', [])
        if not offers:
            return None
        prices = [float(o['price']['grandTotal']) for o in offers]
        return round(min(prices))
    except Exception:
        return None


def get_fare_matrix(origins: list[str], destination: str, date: str) -> dict[str, float]:
    """
    Fetch fares from multiple origins to one destination.
    Returns {iata: fare_usd}. Missing routes return None.
    """
    return {orig: get_fare(orig, destination, date) for orig in origins}


def is_configured() -> bool:
    return bool(_KEY and _SECRET)


# ── Integration point in app.py / meeting_finder_v4 ──────────────────────────
# In app.py search endpoint, optionally call:
#
#   from flight_api import get_fare, is_configured
#   if is_configured() and 'travel_date' in data:
#       real_fare = get_fare(dep, arr, data['travel_date'])
#       if real_fare: use real_fare instead of ml fare estimate
#
# The ML fare predictor is used as fallback when no API key is set.
