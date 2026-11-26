from langchain_core.tools import tool
import requests

from dotenv import load_dotenv
load_dotenv()
import time
from urllib.parse import quote




@tool
def geo_location(destination: str):
    """Search for matching locations using a destination name."""

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": destination,
        "count": 3,
        "language": "en",
        "format": "json"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    results = response.json().get("results", [])
    print(results)
    if not results:
        return "No data found"

    return [
        {
            "name": item.get("name"),
            "country": item.get("country"),
            "state": item.get("admin1"),
            "latitude": item.get("latitude"),
            "longitude": item.get("longitude")
        }
        for item in results
    ]


@tool
def get_weather(latitude: float, longitude: float,start_date: str, end_date: str):
    """Get weather information for a destination
    between the travel start and end dates."""
    print("Running weather tool ! ")

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "weather_code"
        ],
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    data = response.json()

    return data.get("daily", {})



AIRPORT_CODES = {
    "hyderabad": "HYD",
    "delhi": "DEL",
    "mumbai": "BOM",
    "bangalore": "BLR",
    "chennai": "MAA",
    "kolkata": "CCU",
    "pune": "PNQ",
    "goa": "GOI"
}


@tool
def get_flights(departure: str, arrival: str):
    """
    Search for flights between two cities.

    Args:
        departure: Departure city name, e.g. Hyderabad.
        arrival: Arrival city name, e.g. Delhi.

    Returns:
        Available flight information including airline,
        flight number, departure, arrival and status.
    """
    print("get_flights tools  ! ")

    departure_code = AIRPORT_CODES.get(departure.strip().lower())
    arrival_code = AIRPORT_CODES.get(arrival.strip().lower())

    if not departure_code:
        return f"Airport code not available for {departure}"

    if not arrival_code:
        return f"Airport code not available for {arrival}"



    url = "https://api.aviationstack.com/v1/flights"

    params = {
        "access_key": "0f42941a893bf7c4c6637163f7403bd2",
        "dep_iata": departure_code,
        "arr_iata": arrival_code,
        "limit": 20
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()

        data = response.json()

        if "error" in data:
            return f"API Error: {data['error']}"

        flights = data.get("data", [])

        if not flights:
            return "No flights found for this route."

        results = []

        for flight in flights:
            results.append({
                "airline": flight.get("airline", {}).get("name"),
                "flight_number": flight.get("flight", {}).get("iata"),
                "departure": flight.get("departure", {}).get("scheduled"),
                "arrival": flight.get("arrival", {}).get("scheduled"),
                "status": flight.get("flight_status")
            })

        return results

    except requests.RequestException as e:
        return f"Flight API request failed: {str(e)}"



@tool
def get_hotel_list(destination: str):
    """
    Search for hotels in the given destination.
    The destination should be a supported TripAdvisor location.
    """
    print("Getting hotel list !")

    location_keys = {
        "delhi": "g304551",
        # Add more verified location keys here
    }

    location_key = location_keys.get(destination.strip().lower())

    if not location_key:
        return f"Location key not available for {destination}"

    url = "https://data.xotelo.com/api/list"

    params = {
        "location_key": location_key,
        "limit": 30,
        "offset": 0,
        "sort": "best_value"
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()

    return data.get("result", {}).get("list", [])


@tool
def get_tourist_places(destination: str) -> list:
    """
    Search for tourist attractions in a destination using Wikipedia.

    Args:
        destination: City or destination name, e.g. Delhi.

    Returns:
        List of tourist places with title, description, summary and URL.
    """
    print("get the tourist places ! ")

    headers = {
        "User-Agent": "TripMateAI/1.0 (travel planning project)"
    }

    search_url = "https://en.wikipedia.org/w/rest.php/v1/search/page"

    params = {
        "q": f"Tourist attractions in {destination}",
        "limit": 10    }

    response = requests.get(
        search_url,
        params=params,
        headers=headers,
        timeout=20
    )
    response.raise_for_status()

    search_results = response.json().get("pages", [])

    all_places = []

    for result in search_results:

        title = result.get("title")

        if not title:
            continue

        summary_url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + quote(title.replace(" ", "_"), safe="_()")
        )

        try:
            response = requests.get(
                summary_url,
                headers=headers,
                timeout=20
            )
            response.raise_for_status()

            summary = response.json()

            all_places.append({
                "title": summary.get("title"),
                "description": summary.get("description"),
                "summary": summary.get("extract"),
                "url": summary.get("content_urls", {})
                               .get("desktop", {})
                               .get("page")
            })

        except requests.RequestException:
            continue

        time.sleep(2)

    return all_places



planning_tools = [
    get_weather,
    get_flights,
    get_hotel_list,
    get_tourist_places
]