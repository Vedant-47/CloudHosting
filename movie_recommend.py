import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")

@st.cache_data
def load_data():
    df = pd.read_csv("sample_10000_movies.csv")

    # Fix missing columns
    if "description" not in df.columns:
        if "overview" in df.columns:
            df["description"] = df["overview"]
        else:
            st.error("CSV must contain a 'description' or 'overview' column!")
            st.stop()

    if "title" not in df.columns:
        st.error("CSV must contain a 'title' column!")
        st.stop()

    df["description"] = df["description"].fillna("")
    return df

df = load_data()

@st.cache_data
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["description"])
    cosine_sim = cosine_similarity(tfidf_matrix)
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = compute_similarity(df)

def get_recommendations(title):
    if title not in indices:
        return ["Movie not found."]

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df["title"].iloc[movie_indices].tolist()

selected_movie = st.selectbox(
    "Choose a movie:",
    df["title"].sample(min(50, len(df))).sort_values().tolist()
)

if st.button("Recommend"):
    recs = get_recommendations(selected_movie)
    st.subheader("Recommendations:")
    for r in recs:
        st.write("- ", r)
