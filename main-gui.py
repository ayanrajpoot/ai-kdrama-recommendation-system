# Using version 3.0 from trails
# same algorithm as explained in  main-cli.py just added an interface using stream lit.
# for windows
# to run this code go to the project folder in powershell and run
# streamlit run main-gui.py
# For mac os
# python3 -m streamlit run main-gui.py

import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

#funtion for intralist similarity
def compute_intra_list_similarity(top_shows, tfidf_matrix):
    indices = top_shows.index
    sub_matrix = tfidf_matrix[indices]
    sims = cosine_similarity(sub_matrix)
    avg_sim = (sims.sum() - len(sims)) / (len(sims) * (len(sims) - 1))
    return avg_sim


# loading data
@st.cache_data
def load_data():
    df = pd.read_csv('./dataset/Top_5000_popular_drama_details_from_mydramalist.csv')
    df['genres'] = df['genres'].fillna('')
    df['tags'] = df['tags'].fillna('')
    df['content'] = df['content'].fillna('')
    return df


# preprocessing
def preprocess_tags(tags_string):
    return ' '.join([re.sub(r'[/]', ' ', tag.strip()) for tag in tags_string.split(',')])


# main function 
def find_similar_shows(df, target_show_name, knn_neighbors=50, tag_weight=0.60, content_weight=0.10, genre_weight=0.30):
    df['tags_cleaned'] = df['tags'].apply(preprocess_tags)
    target_show = df[df['name'].str.lower() == target_show_name.lower()]

    if target_show.empty:
        st.error(f"❌ Show named '{target_show_name}' not found in dataset.")
        return pd.DataFrame()

    target_index = target_show.index[0]

    #Tag smilarity using KNN
    tag_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tag_matrix = tag_vectorizer.fit_transform(df['tags_cleaned'])
    knn = NearestNeighbors(n_neighbors=min(knn_neighbors + 1, len(df)), metric='cosine')
    knn.fit(tag_matrix)
    distances, indices = knn.kneighbors(tag_matrix[target_index])
    similar_indices = indices[0][1:]
    similar_shows = df.iloc[similar_indices].copy()
    tag_similarities = 1 - distances[0][1:]

    #Content Similarity using Cosine
    content_vectorizer = TfidfVectorizer()
    content_matrix = content_vectorizer.fit_transform(df['content'])
    target_content_vec = content_matrix[target_index]
    content_similarities = cosine_similarity(content_matrix[similar_indices], target_content_vec).flatten()

    # Genre Similarity using Cosine
    genre_vectorizer = TfidfVectorizer()
    genre_matrix = genre_vectorizer.fit_transform(df['genres'])
    target_genre_vec = genre_matrix[target_index]
    genre_similarities = cosine_similarity(genre_matrix[similar_indices], target_genre_vec).flatten()

    #=Combined Weighted Similarity
    total_similarity_index = (
            tag_weight * tag_similarities +
            content_weight * content_similarities +
            genre_weight * genre_similarities
    )
    similar_shows['total_similarity_index'] = total_similarity_index
    top_10 = similar_shows.sort_values(by='total_similarity_index', ascending=False).head(10)

    # return top_10[['name', 'rating', 'genres', 'tags', 'total_similarity_index']]
    return top_10, tag_matrix  # Return both the DataFrame and tag TF-IDF matrix


# stream lit interface 
st.title("🎭 K-Drama Recommender (KNN + Cosine Model)")

df = load_data()

# Dropdown to select show
selected_show = st.selectbox(
    "Choose a drama to get similar recommendations:",
    sorted(df['name'].dropna().unique())
)

# Button to get recommendations
if st.button("Recommend"):
    with st.spinner("Finding similar shows..."):
        results, tag_matrix = find_similar_shows(df, selected_show)
        if not results.empty:
            intra_similarity = compute_intra_list_similarity(results, tag_matrix)
            st.success(f"Top recommendations similar to {selected_show}:")
            st.dataframe(results)
            st.info(f"🔄 Intra-list Similarity Score: {intra_similarity:.4f}")