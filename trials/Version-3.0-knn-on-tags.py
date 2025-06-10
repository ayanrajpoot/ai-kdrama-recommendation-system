# First, I preprocess the dataset by filling in missing values and cleaning up the tags (replacing slashes
# with spaces). Then, I search for the target show in the dataset and retrieve its index. I use TF-IDF to
# vectorize the tags, content, and genres of all the shows, and calculate similarity scores by applying K-Nearest
# Neighbors (KNN) for the tags and cosine similarity for both content and genres. I combine these similarity scores
# using predefined weights for tags, content, and genres to calculate a total similarity score for each show.
# Finally, I return the top 10 most similar shows based on their highest total similarity scores, showing their
# names, genres, tags, and similarity scores.

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('./dataset/Top_5000_popular_drama_details_from_mydramalist.csv')

def preprocess_tags(tags_string):
    # Replace slashes with spaces and strip commas to preserve tag phrases
    return ' '.join([re.sub(r'[/]', ' ', tag.strip()) for tag in tags_string.split(',')])

def find_similar_shows( df, target_show_name, knn_neighbors=40, tag_weight=0.60, content_weight=0.10, genre_weight=0.30):

    # Fill missing values
    df['genres'] = df['genres'].fillna('')
    df['tags'] = df['tags'].fillna('')
    df['content'] = df['content'].fillna('')

    # Preprocess tags
    df['tags_cleaned'] = df['tags'].apply(preprocess_tags)

    # Find the target show
    target_show = df[df['name'].str.lower() == target_show_name.lower()]
    if target_show.empty:
        raise ValueError(f"Show named '{target_show_name}' not found in dataset.")
    target_index = target_show.index[0]

    # Vectorize tags
    tag_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tag_matrix = tag_vectorizer.fit_transform(df['tags_cleaned'])

    # KNN based on tag similarity
    knn = NearestNeighbors(n_neighbors=min(knn_neighbors + 1, len(df)), metric='cosine')
    knn.fit(tag_matrix)
    distances, indices = knn.kneighbors(tag_matrix[target_index])
    similar_indices = indices[0][1:]  # Exclude the show itself
    similar_shows = df.iloc[similar_indices].copy()

    # Content similarity
    content_vectorizer = TfidfVectorizer()
    content_matrix = content_vectorizer.fit_transform(df['content'])
    target_content_vec = content_matrix[target_index]
    content_similarities = cosine_similarity(content_matrix[similar_indices], target_content_vec).flatten()

    # Genre similarity
    genre_vectorizer = TfidfVectorizer()
    genre_matrix = genre_vectorizer.fit_transform(df['genres'])
    target_genre_vec = genre_matrix[target_index]
    genre_similarities = cosine_similarity(genre_matrix[similar_indices], target_genre_vec).flatten()

    # Tag similarity from KNN distances
    tag_similarities = 1 - distances[0][1:]

    # Combine all with weighted similarity
    total_similarity = (
        tag_weight * tag_similarities +
        content_weight * content_similarities +
        genre_weight * genre_similarities
    )
    similar_shows['total_similarity'] = total_similarity

    # Return top 10 similar shows
    top_10 = similar_shows.sort_values(by='total_similarity', ascending=False).head(10)
    return top_10[['name', 'genres', 'tags', 'total_similarity']]

top_similar_shows = find_similar_shows(df, target_show_name="falling into your smile")
print(top_similar_shows)
