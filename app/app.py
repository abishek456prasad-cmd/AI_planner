import streamlit as st
import folium
from streamlit_folium import st_folium
import sys
import os

sys.path.append(os.path.abspath("../src"))

from search import hybrid_search
from llm import extract_preferences, merge_preferences, generate_response

st.set_page_config(page_title="AI Travel Planner", layout="wide")

st.title("🌍 AI Travel Planner - Sikkim")

# -----------------------------
# User Inputs
# -----------------------------
query = st.text_input("What kind of place are you looking for?")

themes = st.multiselect(
    "Select Interests",
    ["nature", "spiritual", "adventure", "cultural", "photography", "offbeat"]
)

crowd = st.selectbox(
    "Crowd Preference",
    ["any", "low", "medium", "high"]
)

difficulty = st.selectbox(
    "Difficulty",
    ["any", "easy", "moderate", "hard"]
)

user_location = (27.33, 88.61)

# -----------------------------
# Session State Init
# -----------------------------
if "results" not in st.session_state:
    st.session_state.results = None

if "query" not in st.session_state:
    st.session_state.query = None

# -----------------------------
# Button Action
# -----------------------------
if st.button("Get Recommendations"):

    if not query:
        st.warning("Please enter a query")
    else:
        explicit = {
            "themes": themes,
            "crowd_level": None if crowd == "any" else crowd,
            "difficulty": None if difficulty == "any" else difficulty
        }

        with st.spinner("🔍 Finding best places for you..."):
            implicit = extract_preferences(query)
            final_prefs = merge_preferences(explicit, implicit)
            results = hybrid_search(query, user_location, final_prefs)

        st.session_state.results = results
        st.session_state.query = query

# -----------------------------
# DISPLAY RESULTS (PERSISTENT)
# -----------------------------
if st.session_state.results:

    results = st.session_state.results
    query = st.session_state.query

    st.subheader("📍 Map View")

    m = folium.Map(
        location=user_location,
        zoom_start=10,
        tiles="CartoDB positron"
    )

    coords = []

    for i, (score, place) in enumerate(results):
        lat = place["lat"]
        lon = place["lon"]

        coords.append([lat, lon])

        if i == 0:
            color = "green"
        elif i < 3:
            color = "blue"
        else:
            color = "red"

        folium.Marker(
            [lat, lon],
            popup=f"""
            <b>{place['name']}</b><br>
            Score: {round(score, 3)}<br>
            Crowd: {place['crowd_level']}<br>
            Themes: {", ".join(place['themes'])}
            """,
            icon=folium.Icon(color=color)
        ).add_to(m)

    if coords:
        folium.PolyLine(coords, color="purple").add_to(m)
        m.fit_bounds(coords)

    st_folium(m, width=900, height=500)

    # -----------------------------
    # Text Results
    # -----------------------------
    st.subheader("📌 Top Recommendations")

    for score, place in results:
        st.write(f"**{place['name']}** ({place['region']}) - Score: {round(score, 3)}")

    # -----------------------------
    # AI Response
    # -----------------------------
    st.subheader("🤖 AI Travel Plan")

    with st.spinner("🧠 Generating travel plan..."):
        response = generate_response(query, results)

    st.write(response)