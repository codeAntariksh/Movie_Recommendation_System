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
    "movie_list.pkl": "https://github.com/codeAntariksh/Movie_Recommendation_System/releases/download/v1.0/movie_list.pkl",
    "similarity.pkl": "https://github.com/codeAntariksh/Movie_Recommendation_System/releases/download/v1.0/similarity.pkl",
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

# ── Fetch poster + metadata together ─────────────────────────────
@st.cache_data(ttl=3600)
def fetch_movie_info(movie_title, movie_id=None):
    """Returns dict: poster, rating, genre, year"""
    info = {"poster": PLACEHOLDER, "rating": "", "genre": "", "year": ""}

    # Strategy 1: TMDB by movie_id — poster only
    if movie_id:
        try:
            url  = f"https://api.themoviedb.org/3/movie/{int(movie_id)}?api_key={TMDB_API_KEY}"
            data = requests.get(url, timeout=6).json()
            path = data.get("poster_path", "")
            if path:
                info["poster"] = f"https://image.tmdb.org/t/p/w500{path}"
        except Exception:
            pass

    # Strategy 2: OMDb — metadata (rating, genre, year) + poster fallback
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
            # Trim genres to first 2 for display
            genres = data.get("Genre", "")
            info["genre"]  = ", ".join(genres.split(", ")[:2]) if genres else ""
    except Exception:
        pass

    # Strategy 3: Wikipedia poster fallback
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
    index     = movies[movies['title'] == movie].index[0]
    distances = sorted(
        enumerate(similarity[index]),
        key=lambda x: x[1], reverse=True
    )
    top5      = distances[1:6]
    titles    = [movies.iloc[i[0]].title    for i in top5]
    movie_ids = [movies.iloc[i[0]].movie_id for i in top5]
    infos     = fetch_all_parallel(titles, movie_ids)
    return titles, infos


# ══════════════════════════════════════════════════════════════════
#   STREAMLIT UI
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Syne:wght@400;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(201,168,76,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(201,168,76,0.03) 0%, transparent 50%);
}

/* ── Hide Streamlit default chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; }

/* ── Hero title ── */
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.8rem, 5vw, 5rem);
    font-weight: 600;
    color: #f0ead8;
    line-height: 1.1;
    letter-spacing: -0.01em;
    margin-bottom: 6px;
}
.hero-title span { color: #c9a84c; font-style: italic; }
.hero-sub {
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4a4558;
    margin-bottom: 36px;
}

/* ── Divider ── */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #c9a84c44, transparent);
    margin: 32px 0;
}

/* ── Selectbox label ── */
.stSelectbox label {
    font-size: 0.65rem !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: #c9a84c !important;
}

/* ── Selectbox input ── */
.stSelectbox > div > div {
    background: #16161f !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 4px !important;
    color: #f0ead8 !important;
}
.stSelectbox > div > div:focus-within {
    border-color: rgba(201,168,76,0.4) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.1) !important;
}

/* ── Button ── */
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
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 8px 30px rgba(201,168,76,0.2) !important;
    margin-top: 8px !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 40px rgba(201,168,76,0.3) !important;
}

/* ── Section heading ── */
.section-label {
    font-size: 0.62rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #c9a84c;
    margin-bottom: 20px;
}
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(1.4rem, 2.5vw, 2rem);
    font-weight: 400;
    font-style: italic;
    color: #f0ead8;
    margin-bottom: 28px;
}
.section-title strong {
    font-style: normal;
    font-weight: 600;
    color: #c9a84c;
}

/* ── Movie card ── */
.movie-card {
    background: #16161f;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 6px;
    overflow: hidden;
    transition: transform 0.3s cubic-bezier(.23,1,.32,1),
                box-shadow 0.3s ease,
                border-color 0.3s ease;
    height: 100%;
}
.movie-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 24px 60px rgba(0,0,0,0.6);
    border-color: rgba(201,168,76,0.25);
}

/* ── Poster ── */
.poster-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 2/3;
    overflow: hidden;
    background: #0d0d14;
}
.poster-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: transform 0.5s cubic-bezier(.23,1,.32,1);
}
.movie-card:hover .poster-wrap img {
    transform: scale(1.06);
}

/* ── Rating badge ── */
.rating-badge {
    position: absolute;
    top: 10px; right: 10px;
    background: rgba(10,10,15,0.88);
    border: 1px solid rgba(201,168,76,0.5);
    border-radius: 4px;
    padding: 4px 9px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #c9a84c;
    backdrop-filter: blur(8px);
    display: flex;
    align-items: center;
    gap: 4px;
}

/* ── Card body ── */
.card-body {
    padding: 14px 14px 16px;
}
.card-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
    font-weight: 600;
    color: #f0ead8;
    line-height: 1.3;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.card-year {
    font-size: 0.65rem;
    letter-spacing: 1px;
    color: #4a4558;
    margin-bottom: 8px;
}
.card-genres {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
}
.genre-tag {
    font-size: 0.6rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 100px;
    border: 1px solid rgba(255,255,255,0.08);
    color: #6b6575;
    white-space: nowrap;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f17 !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebar"] * { color: #f0ead8 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #c9a84c !important; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────
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
<div class="hero-sub">Content-based · 4,806 movies · Powered by ML</div>
""", unsafe_allow_html=True)

# ── Search ────────────────────────────────────────────────────────
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

    cols = st.columns(5)
    for col, name, info in zip(cols, names, infos):
        with col:
            # Build rating badge HTML
            badge = ""
            if info["rating"] and info["rating"] != "N/A":
                badge = f'<div class="rating-badge">★ {info["rating"]}</div>'

            # Build genre tags HTML
            genre_html = ""
            if info["genre"]:
                tags = "".join(
                    f'<span class="genre-tag">{g.strip()}</span>'
                    for g in info["genre"].split(",")
                )
                genre_html = f'<div class="card-genres">{tags}</div>'

            st.markdown(f"""
            <div class="movie-card">
                <div class="poster-wrap">
                    <img src="{info['poster']}" alt="{name}" loading="lazy"/>
                    {badge}
                </div>
                <div class="card-body">
                    <div class="card-title" title="{name}">{name}</div>
                    <div class="card-year">{info.get('year','')}</div>
                    {genre_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
