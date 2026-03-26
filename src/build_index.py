import json
import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -----------------------------
# 1. Load Data
# -----------------------------
with open("../data/sikkim_rag_data.json", "r") as f:
    data = json.load(f)

places = data["places"]

# -----------------------------
# 2. Document Builder
# -----------------------------
def create_document(place):
    text = f"""
    {place['name']} in {place['region']} near {place['nearest_hub']}.

    Description: {place['description']}

    Highlights: {', '.join(place['highlights'])}

    Tip: {place['tip']}

    Themes: {', '.join(place['themes'])}
    """

    metadata = {
        "id": place["id"],
        "name": place["name"],
        "region": place["region"],
        "type": place["type"],
        "nearest_hub": place["nearest_hub"],
        "lat": place["coordinates"]["lat"],
        "lon": place["coordinates"]["lng"],
        "altitude_ft": place["altitude_ft"],
        "crowd_level": place["crowd_level"],
        "difficulty": place["difficulty"],
        "themes": place["themes"],
        "permit_required": place["permit_required"]
    }

    return text.strip(), metadata


# -----------------------------
# 3. Load Embedding Model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
metadatas = []
embeddings = []

print("Creating embeddings...")

for place in tqdm(places):
    text, metadata = create_document(place)

    emb = model.encode(text)

    documents.append(text)
    metadatas.append(metadata)
    embeddings.append(emb)

# Convert to numpy
embeddings = np.array(embeddings).astype("float32")

# Normalize (for cosine similarity)
faiss.normalize_L2(embeddings)

# -----------------------------
# 4. Build FAISS Index
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine similarity

index.add(embeddings)

print(f"Total vectors indexed: {index.ntotal}")

# -----------------------------
# 5. Save Index + Metadata
# -----------------------------
os.makedirs("../index", exist_ok=True)

faiss.write_index(index, "../index/sikkim_index.faiss")

with open("../index/sikkim_data.pkl", "wb") as f:
    pickle.dump({
        "documents": documents,
        "metadatas": metadatas
    }, f)

print("✅ Index built and saved successfully!")