import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import haversine, filter_places

# Load everything once
index = faiss.read_index("../index/sikkim_index.faiss")

with open("../index/sikkim_data.pkl", "rb") as f:
    data = pickle.load(f)

documents = data["documents"]
metadatas = data["metadatas"]

model = SentenceTransformer("all-MiniLM-L6-v2")


def hybrid_search(query, user_location, preferences, k=5):

    # Step 1: Filter
    filtered = filter_places(metadatas, preferences)

    if not filtered:
        filtered = list(enumerate(metadatas))

    filtered_indices = set(i for i, _ in filtered)

    # Step 2: Query embedding
    query_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, len(metadatas))

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx not in filtered_indices:
            continue

        meta = metadatas[idx]

        # -----------------------------
        # GEO DISTANCE
        # -----------------------------
        distance = haversine(
            user_location[0], user_location[1],
            meta["lat"], meta["lon"]
        )

        distance_score = 1 / (1 + distance)  # closer = higher

        # -----------------------------
        # THEME MATCH SCORE
        # -----------------------------
        theme_score = 0
        if "themes" in preferences:
            matches = len(set(preferences["themes"]) & set(meta["themes"]))
            theme_score = matches / len(preferences["themes"])

        # -----------------------------
        # CROWD SCORE (peacefulness)
        # -----------------------------
        crowd_score = 0
        if "crowd_level" in preferences:
            crowd_score = 1 if meta["crowd_level"] == preferences["crowd_level"] else 0

        # -----------------------------
        # BONUS SIGNALS (smart tweaks)
        # -----------------------------
        bonus = 0

        # Boost offbeat places
        if "offbeat" in meta["themes"]:
            bonus += 0.05

        # Penalize high crowd
        if meta["crowd_level"] == "high":
            bonus -= 0.05

        # -----------------------------
        # FINAL SCORE (balanced)
        # -----------------------------
        final_score = (
            score * 0.4 +           # semantic
            distance_score * 0.2 +  # geo
            theme_score * 0.25 +    # preference
            crowd_score * 0.15 +    # strict match
            bonus                   # small adjustments
        )

        results.append((final_score, meta))

    # Step 3: Sort
    results.sort(reverse=True, key=lambda x: x[0])

    return results[:k]


# 🔥 TEST BLOCK
if __name__ == "__main__":
    query = "peaceful offbeat places near Gangtok"

    user_location = (27.33, 88.61)

    preferences = {
        "themes": ["offbeat", "nature"],
        "crowd_level": "low"
    }

    results = hybrid_search(query, user_location, preferences)

    print("\nTop Recommendations:\n")

    for score, r in results:
        print(f"{r['name']} ({r['region']}) → {round(score, 3)}")