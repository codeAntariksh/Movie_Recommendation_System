import os
import pickle
import streamlit as st
import requests
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── API Keys ──────────────────────────────────────────────────────
try:
    from config import OMDB_API_KEY
except ImportError:
    OMDB_API_KEY = st.secrets["OMDB_API_KEY"]

# Free TMDB API key — works on Streamlit Cloud (US servers)
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
PLACEHOLDER  = "https://placehold.co/300x450/0a0a0f/c9a84c?text=No+Poster"

# ── Auto-download models ──────────────────────────────────────────
MODEL_URLS = {
    "movie_list.pkl":  "https://github.com/codeAntariksh/Movie_Recommendation_System/releases/download/v1.0/movie_list.pkl",
    "similarity.pkl":  "https://github.com/codeAntariksh/Movie_Recommendation_System/releases/download/v1.0/similarity.pkl",
}

def download_models():
    for filename, url in MODEL_URLS.items():
        if not os.path.exists(filename):
            with st.spinner(f"Downloading {filename}… (first run only)"):
                r = requests.get(url, timeout=60)
                open(filename, 'wb').write(r.content)

@st.cache_resource
def load_models():
    download_models()
    movies_data     = pickle.load(open('movie_list.pkl', 'rb'))
    similarity_data = pickle.load(open('similarity.pkl', 'rb'))
    return movies_data, similarity_data

# ── Poster fetching — 5 strategies ───────────────────────────────
@st.cache_data(ttl=3600)
def fetch_poster(movie_title, movie_id=None):

    # ── Strategy 1: TMDB by movie_id (best quality, works on Cloud) ──
    if movie_id:
        try:
            url  = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
            data = requests.get(url, timeout=6).json()
            path = data.get("poster_path", "")
            if path:
                return f"https://image.tmdb.org/t/p/w500{path}"
        except Exception:
            pass

    # ── Strategy 2: OMDb direct title ────────────────────────────────
    try:
        url  = f"http://www.omdbapi.com/?t={quote(movie_title)}&type=movie&apikey={OMDB_API_KEY}"
        data = requests.get(url, timeout=5).json()
        if data.get("Response") == "True":
            poster = data.get("Poster", "")
            if poster and poster != "N/A":
                return poster
    except Exception:
        pass

    # ── Strategy 3: OMDb search (fuzzy) ──────────────────────────────
    try:
        url  = f"http://www.omdbapi.com/?s={quote(movie_title)}&type=movie&apikey={OMDB_API_KEY}"
        data = requests.get(url, timeout=5).json()
        if data.get("Response") == "True":
            for result in data.get("Search", []):
                poster = result.get("Poster", "")
                if poster and poster != "N/A":
                    return poster
    except Exception:
        pass

    # ── Strategy 4: Wikipedia thumbnail ──────────────────────────────
    try:
        wiki_url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/"
            f"{quote(movie_title.replace(' ', '_'))}"
        )
        data  = requests.get(wiki_url, timeout=5).json()
        thumb = data.get("thumbnail", {}).get("source", "")
        if thumb:
            return thumb.replace("/50px-", "/400px-").replace("/100px-", "/400px-")
    except Exception:
        pass

    # ── Strategy 5: DuckDuckGo ────────────────────────────────────────
    try:
        ddg_url = (
            f"https://api.duckduckgo.com/?q={quote(movie_title + ' film poster')}"
            f"&format=json&pretty=0&no_html=1&skip_disambig=1"
        )
        data  = requests.get(ddg_url, timeout=5).json()
        image = data.get("Image", "")
        if image and image.strip():
            return f"https://duckduckgo.com{image}" if image.startswith("/") else image
    except Exception:
        pass

    return PLACEHOLDER


# ── Parallel poster fetching ──────────────────────────────────────
def fetch_posters_parallel(titles, movie_ids):
    posters = [""] * len(titles)
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_index = {
            executor.submit(fetch_poster, title, mid): i
            for i, (title, mid) in enumerate(zip(titles, movie_ids))
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                posters[i] = future.result()
            except Exception:
                posters[i] = PLACEHOLDER
    return posters


def recommend(movie, movies, similarity):
    index     = movies[movies['title'] == movie].index[0]
    distances = sorted(
        enumerate(similarity[index]),
        key=lambda x: x[1],
        reverse=True
    )
    top5      = distances[1:6]
    titles    = [movies.iloc[i[0]].title    for i in top5]
    movie_ids = [movies.iloc[i[0]].movie_id for i in top5]  # ← pass TMDB IDs
    posters   = fetch_posters_parallel(titles, movie_ids)
    return titles, posters


# ── Streamlit UI ──────────────────────────────────────────────────
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.header("🎬 Movie Recommender System")

movies, similarity = load_models()

movie_list     = movies['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

if st.button("Show Recommendation"):
    with st.spinner("Finding recommendations…"):
        names, posters = recommend(selected_movie, movies, similarity)

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, name, poster in zip([col1, col2, col3, col4, col5], names, posters):
        with col:
            st.text(name)
            st.image(poster)
