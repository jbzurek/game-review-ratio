import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


def parse_list(s: str) -> list[str]:
    # splituje "a, b, c" -> ["a","b","c"]
    return [x.strip() for x in s.split(",") if x.strip()]


st.set_page_config(page_title="GameReviewRatio - demo", layout="centered")
st.title("GameReviewRatio - demo")

with st.expander("api", expanded=False):
    st.write("API_URL:", API_URL)

st.subheader("basic")
required_age = st.number_input("required_age", min_value=0, max_value=99, value=0, step=1)
price = st.number_input("price (USD)", min_value=0.0, value=19.99, step=1.0)
dlc_count = st.number_input("dlc_count", min_value=0, value=0, step=1)

metacritic_score = st.number_input("metacritic_score", min_value=0, max_value=100, value=0, step=1)
achievements = st.number_input("achievements", min_value=0, value=0, step=1)
discount = st.number_input("discount (%)", min_value=0, max_value=100, value=0, step=1)

release_year = st.number_input("release_year", min_value=1990, max_value=2035, value=2024, step=1)
release_month = st.number_input("release_month", min_value=1, max_value=12, value=6, step=1)

st.subheader("platforms")
c1, c2, c3 = st.columns(3)
with c1:
    windows = st.checkbox("windows", value=True)
with c2:
    mac = st.checkbox("mac", value=False)
with c3:
    linux = st.checkbox("linux", value=False)

st.subheader("labels (comma separated)")
genres = st.text_input("genres", value="Action, Indie")
categories = st.text_input("categories", value="Single-player")
tags = st.text_input("tags", value="Action, Indie")

developers = st.text_input("developers", value="")
publishers = st.text_input("publishers", value="")

supported_languages = st.text_input("supported_languages", value="English")
full_audio_languages = st.text_input("full_audio_languages", value="English")

payload = {
    "required_age": int(required_age),
    "price": float(price),
    "dlc_count": int(dlc_count),
    "windows": bool(windows),
    "mac": bool(mac),
    "linux": bool(linux),
    "metacritic_score": int(metacritic_score),
    "achievements": int(achievements),
    "discount": int(discount),
    "release_year": int(release_year),
    "release_month": int(release_month),
    "genres": parse_list(genres),
    "categories": parse_list(categories),
    "tags": parse_list(tags),
    "developers": parse_list(developers),
    "publishers": parse_list(publishers),
    "supported_languages": parse_list(supported_languages),
    "full_audio_languages": parse_list(full_audio_languages),
}

if st.button("Predict"):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        st.write("status:", r.status_code)

        try:
            st.json(r.json())
        except Exception:
            st.code(r.text)

    except Exception as e:
        st.error(f"request failed: {e}")

with st.expander("payload", expanded=False):
    st.json(payload)
