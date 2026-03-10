import pickle
import streamlit as st
import requests
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import OMDB_API_KEY

PLACEHOLDER = "https://placehold.co/300x450/1a1a2e/c9a84c?text=No+Poster"

# ── 1. Cache heavy model loading — only runs ONCE per session ──
@st.cache_resource
def load_models():
    movies_data   = pickle.load(open('movie_list.pkl', 'rb'))
    similarity_data = pickle.load(open('similarity.pkl', 'rb'))
    return movies_data, similarity_data

# ── 2. Cache poster per title — never fetches same movie twice ─
@st.cache_data(ttl=3600)
def fetch_poster(movie_title):
    # Strategy 1: OMDb direct title
    try:
        url  = f"http://www.omdbapi.com/?t={quote(movie_title)}&type=movie&apikey={OMDB_API_KEY}"
        data = requests.get(url, timeout=5).json()
        if data.get("Response") == "True":
            poster = data.get("Poster", "")
            if poster and poster != "N/A":
                return poster
    except Exception:
        pass

    # Strategy 2: OMDb search (fuzzy match)
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

    # Strategy 3: Wikipedia thumbnail
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

    # Strategy 4: DuckDuckGo
    try:
        ddg_url = (
            f"https://api.duckduckgo.com/?q={quote(movie_title + ' film')}"
            f"&format=json&pretty=0&no_html=1&skip_disambig=1"
        )
        data  = requests.get(ddg_url, timeout=5).json()
        image = data.get("Image", "")
        if image and image.strip():
            return f"https://duckduckgo.com{image}" if image.startswith("/") else image
    except Exception:
        pass

    return PLACEHOLDER


# ── 3. Fetch all 5 posters IN PARALLEL ────────────────────────
def fetch_posters_parallel(titles):
    posters = [""] * len(titles)
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_index = {
            executor.submit(fetch_poster, title): i
            for i, title in enumerate(titles)
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
    titles = [movies.iloc[i[0]].title for i in distances[1:6]]

    # ── Fetch all posters simultaneously ──────────────────────
    posters = fetch_posters_parallel(titles)
    return titles, posters


# ── Streamlit UI ──────────────────────────────────────────────
st.header('Movie Recommender System')

# Models load once and stay cached
movies, similarity = load_models()

movie_list     = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    with st.spinner('Finding recommendations…'):
        recommended_movie_names, recommended_movie_posters = recommend(
            selected_movie, movies, similarity
        )

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, name, poster in zip(
        [col1, col2, col3, col4, col5],
        recommended_movie_names,
        recommended_movie_posters
    ):
        with col:
            st.text(name)
            st.image(poster)