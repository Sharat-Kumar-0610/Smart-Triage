import requests
import math


# ----------------------------
# Distance Calculation (Haversine)
# ----------------------------
def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two coordinates in KM
    """
    R = 6371  # Earth radius in KM

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


# ----------------------------
# Routing Score Calculation
# ----------------------------
def compute_routing_score(distance_km, emergency=False):
    """
    Lower distance = higher score
    Emergency cases boost closer hospitals even more
    """

    # Normalize distance score (closer = better)
    distance_score = 1 / (distance_km + 0.1)

    emergency_boost = 1.5 if emergency else 1.0

    return distance_score * emergency_boost


# ----------------------------
# Main Hospital Finder
# ----------------------------
def get_nearest_hospitals(lat, lon, emergency=False):
    """
    Fetch nearby hospitals and rank them intelligently
    """

    if lat is None or lon is None:
        return []

    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": "hospital",
        "format": "json",
        "limit": 10,
        "viewbox": f"{float(lon)-0.05},{float(lat)+0.05},{float(lon)+0.05},{float(lat)-0.05}",
        "bounded": 1
    }

    headers = {
        "User-Agent": "SmartTriageApp"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)

        if response.status_code != 200:
            return []

        data = response.json()

        hospitals = []

        for place in data:
            hospital_lat = place.get("lat")
            hospital_lon = place.get("lon")

            if hospital_lat and hospital_lon:
                distance = calculate_distance(
                    lat, lon,
                    hospital_lat, hospital_lon
                )

                routing_score = compute_routing_score(
                    distance_km=distance,
                    emergency=emergency
                )

                hospitals.append({
                    "name": place.get("display_name"),
                    "latitude": hospital_lat,
                    "longitude": hospital_lon,
                    "distance_km": round(distance, 2),
                    "routing_score": round(routing_score, 4)
                })

        # Sort by routing score (highest first)
        hospitals = sorted(
            hospitals,
            key=lambda x: x["routing_score"],
            reverse=True
        )

        # Return top 5 optimized hospitals
        return hospitals[:5]

    except Exception as e:
        print("Hospital lookup error:", e)
        return []