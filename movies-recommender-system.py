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

TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
PLACEHOLDER  = "https://placehold.co/300x450/0a0a0f/c9a84c?text=No+Poster"

# ── Model URLs ────────────────────────────────────────────────────
MODEL_URLS = {
    "movie_list.pkl": "https://github.com/codeAntariksh/Movie_Recommendation_System/releases/download/v2.0/movie_list.pkl",
    "similarity.pkl": "https://github.com/codeAntariksh/Movie_Recommendation_System/releases/download/v2.0/similarity.pkl",
}

def download_models():
    for filename, url in MODEL_URLS.items():
        if not os.path.exists(filename):
            with st.spinner(f"Downloading {filename}… (first run only)"):
                r = requests.get(url, timeout=120)
                open(filename, 'wb').write(r.content)

@st.cache_resource
def load_models():
    download_models()
    movies_data     = pickle.load(open('movie_list.pkl', 'rb'))
    similarity_data = pickle.load(open('similarity.pkl', 'rb'))
    return movies_data, similarity_data

# ── Fetch poster + metadata ───────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_movie_info(movie_title, movie_id=None):
    info = {"poster": PLACEHOLDER, "rating": "", "genre": "", "year": ""}

    # Strategy 1: TMDB — poster
    if movie_id:
        try:
            url  = f"https://api.themoviedb.org/3/movie/{int(movie_id)}?api_key={TMDB_API_KEY}"
            data = requests.get(url, timeout=6).json()
            path = data.get("poster_path", "")
            if path:
                info["poster"] = f"https://image.tmdb.org/t/p/w500{path}"
        except Exception:
            pass

    # Strategy 2: OMDb — metadata + poster fallback
    try:
        url  = f"http://www.omdbapi.com/?t={quote(movie_title)}&type=movie&apikey={OMDB_API_KEY}"
        data = requests.get(url, timeout=5).json()
        if data.get("Response") == "True":
            if info["poster"] == PLACEHOLDER:
                p = data.get("Poster", "")
                if p and p != "N/A":
                    info["poster"] = p
            info["rating"] = data.get("imdbRating", "")
            info["year"]   = data.get("Year", "")
            genres         = data.get("Genre", "")
            info["genre"]  = ", ".join(genres.split(", ")[:2]) if genres else ""
    except Exception:
        pass

    # Strategy 3: Wikipedia — poster fallback
    if info["poster"] == PLACEHOLDER:
        try:
            wiki_url = (
                f"https://en.wikipedia.org/api/rest_v1/page/summary/"
                f"{quote(movie_title.replace(' ', '_'))}"
            )
            data  = requests.get(wiki_url, timeout=5).json()
            thumb = data.get("thumbnail", {}).get("source", "")
            if thumb:
                info["poster"] = thumb.replace("/50px-", "/400px-").replace("/100px-", "/400px-")
        except Exception:
            pass

    return info


def fetch_all_parallel(titles, movie_ids):
    results = [None] * len(titles)
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_index = {
            executor.submit(fetch_movie_info, title, mid): i
            for i, (title, mid) in enumerate(zip(titles, movie_ids))
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception:
                results[i] = {"poster": PLACEHOLDER, "rating": "", "genre": "", "year": ""}
    return results


def recommend(movie, movies, similarity):
    # Use positional index to ensure alignment with similarity matrix
    movie_list = movies['title'].tolist()
    index      = movie_list.index(movie)   # 0-based positional, not DataFrame label
    distances  = sorted(
        enumerate(similarity[index]),
        key=lambda x: x[1], reverse=True
    )
    top5      = distances[1:6]
    titles    = [movies.iloc[i[0]].title    for i in top5]
    movie_ids = [movies.iloc[i[0]].movie_id for i in top5]
    infos     = fetch_all_parallel(titles, movie_ids)
    return titles, infos


# ══════════════════════════════════════════════════════════════════
#   PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Syne:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse at 15% 15%, rgba(201,168,76,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 85%, rgba(201,168,76,0.03) 0%, transparent 50%);
}

/* Hide default chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 4rem; max-width: 1200px; }

/* Hero */
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.6rem, 5vw, 4.5rem);
    font-weight: 600;
    color: #f0ead8;
    line-height: 1.1;
    margin-bottom: 32px;
}
.hero-title span { color: #c9a84c; font-style: italic; }

/* Gold divider */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #c9a84c55, transparent);
    margin: 28px 0;
}

/* Section labels */
.section-label {
    font-size: 0.62rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #c9a84c;
    margin-bottom: 8px;
}
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(1.3rem, 2.2vw, 1.8rem);
    font-weight: 400;
    font-style: italic;
    color: #f0ead8;
    margin-bottom: 24px;
}
.section-title strong { font-style: normal; font-weight: 600; color: #c9a84c; }

/* Selectbox */
.stSelectbox label {
    font-size: 0.65rem !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: #c9a84c !important;
}
.stSelectbox > div > div {
    background: #16161f !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 4px !important;
    color: #f0ead8 !important;
}
.stSelectbox > div > div:focus-within {
    border-color: rgba(201,168,76,0.4) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.08) !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #c9a84c, #a07832) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 14px 28px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    box-shadow: 0 8px 30px rgba(201,168,76,0.2) !important;
    margin-top: 8px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 40px rgba(201,168,76,0.3) !important;
}

/* Poster image — constrained size */
[data-testid="stImage"] img {
    border-radius: 6px 6px 0 0 !important;
    width: 100% !important;
    aspect-ratio: 2/3 !important;
    object-fit: cover !important;
    display: block !important;
    transition: transform 0.4s cubic-bezier(.23,1,.32,1) !important;
}

/* Card wrapper */
.card-wrap {
    background: #16161f;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 6px;
    overflow: hidden;
    transition: transform 0.3s cubic-bezier(.23,1,.32,1), box-shadow 0.3s, border-color 0.3s;
    margin-bottom: 8px;
}
.card-wrap:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    border-color: rgba(201,168,76,0.2);
}

/* Card text block */
.card-info { padding: 10px 12px 12px; }
.card-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #f0ead8;
    line-height: 1.3;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.card-meta {
    font-size: 0.65rem;
    color: #4a4558;
    letter-spacing: 0.5px;
    margin-bottom: 7px;
}
.card-rating {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(201,168,76,0.1);
    border: 1px solid rgba(201,168,76,0.25);
    border-radius: 100px;
    padding: 2px 9px;
    font-size: 0.68rem;
    font-weight: 700;
    color: #c9a84c;
    margin-bottom: 7px;
}
.card-genres { display: flex; flex-wrap: wrap; gap: 4px; }
.genre-pill {
    font-size: 0.58rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 100px;
    border: 1px solid rgba(255,255,255,0.07);
    color: #5a5468;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f17 !important;
    border-right: 1px solid rgba(255,255,255,0.04) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #f0ead8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────
movies, similarity = load_models()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 CineMatch")
    st.markdown("---")
    st.markdown(f"**{len(movies):,}** movies indexed")
    st.markdown("**Algorithm:** Cosine Similarity")
    st.markdown("**Features:** Genres · Keywords · Cast · Crew · Plot")
    st.markdown("---")
    st.markdown("**Poster sources**")
    st.markdown("TMDB → OMDb → Wikipedia")
    st.markdown("---")
    st.caption("Built by Antarikshya")

# ── Hero ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">Find your next<br><span>favourite film.</span></div>
""", unsafe_allow_html=True)

# ── Search row ────────────────────────────────────────────────────
col_select, col_btn = st.columns([4, 1])
with col_select:
    selected_movie = st.selectbox(
        "Select a movie",
        movies['title'].values,
        label_visibility="collapsed",
        placeholder="Type or select a movie…"
    )
with col_btn:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    search = st.button("Recommend →")

st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────
if search:
    with st.spinner("Finding recommendations…"):
        names, infos = recommend(selected_movie, movies, similarity)

    st.markdown(f"""
    <div class="section-label">Because you watched</div>
    <div class="section-title">Films similar to <strong>{selected_movie}</strong></div>
    """, unsafe_allow_html=True)

    cols = st.columns(5, gap="medium")
    for col, name, info in zip(cols, names, infos):
        with col:
            # Poster via st.image — Streamlit handles column width correctly
            st.image(info["poster"], use_container_width=True)

            # Rating
            rating_html = ""
            if info["rating"] and info["rating"] != "N/A":
                rating_html = f'<div class="card-rating">★ {info["rating"]}</div>'

            # Genre pills
            genre_html = ""
            if info["genre"]:
                pills = "".join(
                    f'<span class="genre-pill">{g.strip()}</span>'
                    for g in info["genre"].split(",")
                )
                genre_html = f'<div class="card-genres">{pills}</div>'

            # Card info block
            st.markdown(f"""
            <div class="card-info">
                <div class="card-title" title="{name}">{name}</div>
                <div class="card-meta">{info.get("year", "")}</div>
                {rating_html}
                {genre_html}
            </div>
            """, unsafe_allow_html=True)
