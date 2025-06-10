# First, I preprocess the dataset by filling any missing values in the relevant columns.
# Then, I use TF-IDF to vectorize the genres, tags, and content, converting the text data into numerical form.
# I apply K-Nearest Neighbors (KNN) on the genre vectors to find similar shows and calculate cosine similarity
# between the target show’s tags and content and those of other shows. I then combine these similarities using
# weighted scores for tags and content to calculate a total similarity score for each show.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset
df = pd.read_csv('./dataset/Top_5000_popular_drama_details_from_mydramalist.csv')


def find_similar_shows(df, target_show_name, knn_neighbors=100, tag_weight=0.7, content_weight=0.3):
    # Ensure columns are filled
    df['genres'] = df['genres'].fillna('')
    df['tags'] = df['tags'].fillna('')
    df['content'] = df['content'].fillna('')

    # Find the target show
    target_show = df[df['name'].str.lower() == target_show_name.lower()]
    if target_show.empty:
        raise ValueError(f"Show named '{target_show_name}' not found in dataset.")

    target_index = target_show.index[0]

    # Vectorize genres for KNN
    genre_vectorizer = TfidfVectorizer()
    genre_tfidf = genre_vectorizer.fit_transform(df['genres'])

    # Apply KNN on genres
    knn = NearestNeighbors(n_neighbors=min(knn_neighbors + 1, len(df)), metric='cosine')
    knn.fit(genre_tfidf)
    distances, indices = knn.kneighbors(genre_tfidf[target_index])

    similar_indices = indices[0][1:]  # Exclude the target show itself
    similar_shows = df.iloc[similar_indices].copy()

    # Vectorize tags and content
    tag_vectorizer = TfidfVectorizer()
    content_vectorizer = TfidfVectorizer()

    tag_matrix = tag_vectorizer.fit_transform(df['tags'])
    content_matrix = content_vectorizer.fit_transform(df['content'])

    #Compute cosine similarity for tags and content
    target_tag_vec = tag_matrix[target_index]
    target_content_vec = content_matrix[target_index]

    tag_similarities = cosine_similarity(tag_matrix[similar_indices], target_tag_vec).flatten()
    content_similarities = cosine_similarity(content_matrix[similar_indices], target_content_vec).flatten()

    # Compute total similarity with weights
    total_similarity = tag_weight * tag_similarities + content_weight * content_similarities

    similar_shows['total_similarity'] = total_similarity

    # Return top 10 results
    top_10 = similar_shows.sort_values(by='total_similarity', ascending=False).head(10)
    return top_10[['name', 'total_similarity']]



top_similar_shows = find_similar_shows(df, target_show_name='hotel del luna')
print(top_similar_shows)
