# country specific-weight testing

import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('./dataset/Top_5000_popular_drama_details_from_mydramalist.csv')

def preprocess_tags(tags_string):
    # Replace slashes with spaces and strip commas to preserve tag phrases
    if pd.isna(tags_string):
        return ''
    return ' '.join([re.sub(r'[/]', ' ', tag.strip()) for tag in tags_string.split(',')])

# Country-specific weights
country_weights_dict = {
    'China': 1.0,  # Normal weight
    'South Korea': 1.0,  # Higher weight for South Korean shows
    'Japan': 0.1,  # Slightly higher weight for Japanese shows
    'Thailand': 0.05,
    'Taiwan': 0.05,
    'Hong Kong': 0.05,
    'Philippines': 0,
    # Add other countries with custom weights
}

def get_country_similarity(country1, country2):
    # Binary matching with custom weights based on the dictionary
    if pd.isna(country1) or pd.isna(country2):
        return 0
    if country1.lower() == country2.lower():
        return 1
    else:
        return country_weights_dict.get(country2, 0.5)  # Default weight is 0.5 if not found in dictionary

def find_similar_shows(df, target_show_name, knn_neighbors=50, tag_weight=0.4, content_weight=0.2, genre_weight=0.2,
                       country_weight=0.2):
    # Fill missing values
    df['genres'] = df['genres'].fillna('')
    df['tags'] = df['tags'].fillna('')
    df['content'] = df['content'].fillna('')
    df['country'] = df['country'].fillna('')  # Handle missing country data

    # Preprocess tags
    df['tags_cleaned'] = df['tags'].apply(preprocess_tags)

    # Find the target show
    target_show = df[df['name'].str.lower() == target_show_name.lower()]
    if target_show.empty:
        raise ValueError(f"Show named '{target_show_name}' not found in dataset.")
    target_index = target_show.index[0]

    # Step 1: Vectorize tags
    tag_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tag_matrix = tag_vectorizer.fit_transform(df['tags_cleaned'])

    # Step 2: KNN based on tag similarity
    knn = NearestNeighbors(n_neighbors=min(knn_neighbors + 1, len(df)), metric='cosine')
    knn.fit(tag_matrix)
    distances, indices = knn.kneighbors(tag_matrix[target_index])
    similar_indices = indices[0][1:]  # Exclude the show itself
    similar_shows = df.iloc[similar_indices].copy()

    # Step 3: Content similarity
    content_vectorizer = TfidfVectorizer()
    content_matrix = content_vectorizer.fit_transform(df['content'])
    target_content_vec = content_matrix[target_index]
    content_similarities = cosine_similarity(content_matrix[similar_indices], target_content_vec).flatten()

    # Step 4: Genre similarity
    genre_vectorizer = TfidfVectorizer()
    genre_matrix = genre_vectorizer.fit_transform(df['genres'])
    target_genre_vec = genre_matrix[target_index]
    genre_similarities = cosine_similarity(genre_matrix[similar_indices], target_genre_vec).flatten()

    # Step 5: Country similarity (using custom function)
    country_similarities = [
        get_country_similarity(df['country'].iloc[target_index], country) for country in df['country'].iloc[similar_indices]
    ]

    # Step 6: Tag similarity from KNN distances
    tag_similarities = 1 - distances[0][1:]

    # Convert all similarities to numpy arrays
    tag_similarities = np.array(tag_similarities)
    content_similarities = np.array(content_similarities)
    genre_similarities = np.array(genre_similarities)
    country_similarities = np.array(country_similarities)

    # Step 7: Combine all with weighted similarity
    total_similarity = (
            tag_weight * tag_similarities +
            content_weight * content_similarities +
            genre_weight * genre_similarities +
            country_weight * country_similarities
    )

    similar_shows['total_similarity'] = total_similarity

    # Step 8: Return top 10 similar shows
    top_10 = similar_shows.sort_values(by='total_similarity', ascending=False).head(10)
    return top_10[['name', 'genres', 'country', 'total_similarity']]

# Run the function for "Love Scenery"
try:
    top_similar_shows = find_similar_shows(df, target_show_name="what's wrong with secretary kim")
    print(top_similar_shows.to_string())
except Exception as e:
    print(f"Error: {e}")