import faiss
import pickle
from sentence_transformers import SentenceTransformer
from utils import haversine, filter_places
from llm import extract_preferences, merge_preferences

index = faiss.read_index("../index/sikkim_index.faiss")

with open("../index/sikkim_data.pkl", "rb") as f:
    data = pickle.load(f)

documents = data["documents"]
metadatas = data["metadatas"]

model = SentenceTransformer("all-MiniLM-L6-v2")


def hybrid_search(query, user_location, preferences, k=5):
    filtered = filter_places(metadatas, preferences)

    if not filtered:
        filtered = list(enumerate(metadatas))

    filtered_indices = set(i for i, _ in filtered)

    query_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, len(metadatas))

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx not in filtered_indices:
            continue

        meta = metadatas[idx]

        distance = haversine(
            user_location[0],
            user_location[1],
            meta["lat"],
            meta["lon"]
        )

        distance_score = 1 / (1 + distance)

        theme_score = 0
        if preferences.get("themes"):
            matches = len(set(preferences["themes"]) & set(meta["themes"]))
            theme_score = matches / len(preferences["themes"])

        crowd_score = 0
        if preferences.get("crowd_level"):
            crowd_score = 1 if meta["crowd_level"] == preferences["crowd_level"] else 0

        bonus = 0
        if "offbeat" in meta["themes"]:
            bonus += 0.05
        if meta["crowd_level"] == "high":
            bonus -= 0.05

        final_score = (
            score * 0.4 +
            distance_score * 0.2 +
            theme_score * 0.25 +
            crowd_score * 0.15 +
            bonus
        )

        results.append((final_score, meta))

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:k]


if __name__ == "__main__":
    import folium

    query = "peaceful and famous places near Gangtok"

    explicit = {
        "themes": ["adventure"]
    }

    implicit = extract_preferences(query)
    final_prefs = merge_preferences(explicit, implicit)

    user_location = (27.3389, 88.606)

    results = hybrid_search(query, user_location, final_prefs)

    print("\nTop Recommendations:\n")
    for score, r in results:
        print(f"{r['name']} → {round(score, 3)}")

    m = folium.Map(
        location=[27.33, 88.61],
        zoom_start=10,
        tiles="CartoDB positron"
    )

    coordinates = []

    for i, (score, place) in enumerate(results):
        lat = place["lat"]
        lon = place["lon"]
        name = place["name"]

        coordinates.append([lat, lon])

        if i == 0:
            color = "green"
        elif i < 3:
            color = "blue"
        else:
            color = "red"

        folium.Marker(
            location=[lat, lon],
            popup=f"""
            <b>{name}</b><br>
            Score: {round(score, 3)}<br>
            Crowd: {place['crowd_level']}<br>
            Themes: {", ".join(place['themes'])}
            """,
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(m)

    if coordinates:
        folium.PolyLine(
            locations=coordinates,
            color="purple",
            weight=3,
            opacity=0.7
        ).add_to(m)
        m.fit_bounds(coordinates)

    m.save("../map.html")
    print("✅ Upgraded map saved as map.html")