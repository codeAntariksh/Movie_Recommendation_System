<div align="center">

# 🎬 CineMatch — Movie Recommender System

### *Discover films you'll love, powered by machine learning*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://movie-recommendation-system-antariksh.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<img src="https://github.com/codeAntariksh/Movie_Recommendation_System/raw/main/assets/demo.png" alt="CineMatch Demo" width="85%" style="border-radius:12px"/>

> Select any movie from a library of **4,806 films** and instantly receive  
> 5 tailored recommendations with posters, powered by cosine similarity.

</div>

---

## ✨ Features

- 🎯 **Content-based filtering** — recommendations based on genres, keywords, cast, crew & plot
- ⚡ **Parallel poster fetching** — all 5 posters load simultaneously via `ThreadPoolExecutor`
- 🧠 **Smart caching** — models load once per session; posters cached for 1 hour
- 🖼️ **Multi-source posters** — TMDB → OMDb → Wikipedia fallback chain
- ⭐ **Rich metadata** — IMDb rating + genre tags displayed on every card
- ☁️ **Cloud deployed** — live on Streamlit Cloud, zero setup for users
- 🔐 **Secure** — API keys stored in Streamlit Secrets, never in code

---

## 🧠 How It Works

```
User selects a movie
        ↓
Look up positional index in preprocessed DataFrame
        ↓
Retrieve precomputed cosine similarity row
        ↓
Sort all 4,806 movies by similarity score
        ↓
Return top 5 matches (excluding the query film)
        ↓
Fetch poster + IMDb rating + genre in parallel from TMDB / OMDb / Wikipedia
        ↓
Display recommendation cards with metadata
```

### ML Pipeline

| Step | Technique | Detail |
|---|---|---|
| Text cleaning | Tokenization + Lowercasing | Overview, genres, keywords, cast, crew → single `tags` string |
| Normalization | Porter Stemmer (NLTK) | `"running"` → `"run"`, `"loved"` → `"love"` |
| Vectorization | `TfidfVectorizer` | Top 5,000 features, English stop words removed |
| Similarity | Cosine Similarity | `sklearn.metrics.pairwise.cosine_similarity` |
| Storage | Pickle | Precomputed matrix serialized for instant loading |

---

## 📁 Repository Structure

```
Movie_Recommendation_System/
│
├── 📄 movies-recommender-system.py   # Main Streamlit application
├── 📓 movie_recommender-system.ipynb # Full ML pipeline notebook
│
├── ⚙️  requirements.txt              # Pinned Python dependencies
├── 🔧 setup.sh                       # Streamlit Cloud server config
├── 🙈 .gitignore                     # Excludes secrets + model files
│
├── 📁 assets/                        # Screenshots for README
│   └── demo.png
│
└── 📄 README.md                      # You are here
```

> **Note:** `movie_list.pkl` and `similarity.pkl` are excluded from the repo  
> and auto-downloaded from [GitHub Releases](../../releases) on first run.

---

## 🚀 Run Locally

### Prerequisites
- Python 3.11+
- Free [OMDb API key](http://www.omdbapi.com/apikey.aspx)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/codeAntariksh/Movie_Recommendation_System.git
cd Movie_Recommendation_System
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API key**

Create `config.py` in the project root:
```python
# never commit this file
OMDB_API_KEY = "your_omdb_key_here"
```

**4. Download model files**

Either re-run the notebook to regenerate them, or download from [Releases](../../releases/tag/v2.0):
```
Place these in the project root: movie_list.pkl and similarity.pkl
```

**5. Launch the app**
```bash
streamlit run movies-recommender-system.py
```

Open → [http://localhost:8501](http://localhost:8501)

---

## ☁️ Deploy Your Own Instance

**1. Fork this repo** — click the **Fork** button at the top right.

**2. Sign in to Streamlit Cloud** — go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub.

**3. Upload model files to GitHub Releases** — create a release tagged `v2.0` and attach both pkl files.

**4. Create a new app on Streamlit Cloud**

| Field | Value |
|---|---|
| Repository | `your-username/Movie_Recommendation_System` |
| Branch | `main` |
| Main file | `movies-recommender-system.py` |
| Python version | `3.11` |

**5. Add your secret** — under **Advanced Settings → Secrets**:
```toml
OMDB_API_KEY = "your_omdb_key_here"
```

**6. Deploy 🚀**

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.32.0 | Web UI framework |
| `scikit-learn` | 1.4.0 | TfidfVectorizer + Cosine Similarity |
| `pandas` | 2.1.4 | Data manipulation |
| `numpy` | 1.26.4 | Numerical operations |
| `nltk` | 3.8.1 | Porter Stemmer |
| `requests` | 2.31.0 | Poster + metadata API calls |

---

## 📊 Dataset

- **Source:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) — Kaggle
- **Size:** 4,806 movies after preprocessing
- **Features used:** `genres` · `keywords` · `cast` (top 3) · `crew` (director) · `overview`
- **Coverage:** Movies up to ~2017

---

## 🔐 Security

| Secret | Local | Streamlit Cloud |
|---|---|---|
| `OMDB_API_KEY` | `config.py` (gitignored) | App Settings → Secrets (encrypted) |
| Model files | Local disk | GitHub Releases (public, read-only) |

---

## 🗺️ Roadmap

- [x] TF-IDF vectorization upgrade
- [x] IMDb rating + genre display on each card
- [x] Custom CSS cinematic dark UI
- [x] Cloud deployment on Streamlit Cloud
- [ ] Expand to 45,000+ movies (MovieLens dataset)
- [ ] Custom HTML/CSS/JS frontend (Flask backend)
- [ ] User-based collaborative filtering

---

## 👤 Author

**Antarikshya** ([@codeAntariksh](https://github.com/codeAntariksh))

If you found this useful, consider giving it a ⭐ — it helps a lot!

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ using Python · Streamlit · scikit-learn · TMDB</sub>
</div>
