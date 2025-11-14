import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System (10,000+ Movies)")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\Admin\Desktop\internship_projects\movie_recommendation\sample_10000_movies.csv")

df = load_data()

# Compute TF-IDF Matrix
@st.cache_data
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(data.index, index=data['title']).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = compute_similarity(df)

# Get Recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return ["Movie not found."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# UI
selected_movie = st.selectbox("Choose a movie to get similar recommendations:", df['title'].sample(50).sort_values().tolist())

if st.button("Recommend"):
    recommendations = get_recommendations(selected_movie)
    st.subheader("🎥 Recommended Movies:")
    for rec in recommendations:
        st.write(f"- {rec}")
