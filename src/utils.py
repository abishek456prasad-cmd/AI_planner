import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)

    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def filter_places(metadatas, preferences):
    filtered = []

    for i, meta in enumerate(metadatas):

        if "themes" in preferences:
            if not any(t in meta["themes"] for t in preferences["themes"]):
                continue

        if "crowd_level" in preferences:
            if meta["crowd_level"] != preferences["crowd_level"]:
                continue

        filtered.append((i, meta))

    return filtered