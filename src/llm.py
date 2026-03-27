from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)


def normalize_preferences(prefs):
    return {
        "themes": prefs.get("themes", []),
        "crowd_level": prefs.get("crowd_level", None),
        "difficulty": prefs.get("difficulty", None)
    }


def extract_preferences(query):
    prompt = f"""
You are an intelligent travel assistant.

Extract structured preferences from this query:

Query: "{query}"

Return ONLY valid JSON (no markdown, no explanation):

{{
  "themes": ["nature", "offbeat"],
  "crowd_level": "low",
  "difficulty": "easy"
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()
    text = re.sub(r"```json|```", "", text).strip()

    try:
        prefs = json.loads(text)
    except Exception:
        prefs = {}

    return normalize_preferences(prefs)


def merge_preferences(explicit, implicit):
    final = explicit.copy()

    for key, value in implicit.items():
        if value is None:
            continue

        if key not in final or final[key] in [None, [], ""]:
            final[key] = value

        elif isinstance(value, list):
            final[key] = list(set(final[key]) | set(value))

    return final


def generate_response(query, results):
    places_text = ""

    for i, (score, place) in enumerate(results, 1):
        places_text += f"""
{i}. {place['name']} ({place['region']})
- Type: {place['type']}
- Themes: {', '.join(place['themes'])}
- Crowd: {place['crowd_level']}
"""

    prompt = f"""
You are an expert travel planner for Sikkim.

User query:
{query}

Top recommended places:
{places_text}

Instructions:
- Explain why each place matches the user's intent
- Highlight unique aspects
- Suggest a realistic 1 day plan
- Keep it concise

Answer:
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content