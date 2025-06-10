
# Summary
# loading a CSV dataset and preprocessing the data to fill missing
#  values in the content, genres, and tags columns.  boosting the weight of tags by duplicating
#  them three times before combining the content, genres, and weighted tags into a single feature for each show.
#  Using the TfidfVectorizer,
#  THen  transforming  the combined text features into numerical vectors and calculates the
#  cosine similarity between shows based on these vectors.
#  The recommend_shows function takes a show name, finds its index in the dataset, calculates its similarity with
#  all other shows, and recommends the top N similar shows
#  sorted by rating and similarity index. The recommended shows are then returned with details such as name, rating,
#  genres, and tags.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV data
df = pd.read_csv('./dataset/Top_5000_popular_drama_details_from_mydramalist.csv')

# 1. Data Preprocessing: Filling  missing values (if any).
df['content'] = df['content'].fillna('')
df['genres'] = df['genres'].fillna('')
df['tags'] = df['tags'].fillna('')

# 2. Increase weightage for tags by duplicating its
tag_weightage_multiplier = 3  # Multiply tags by 3 to give them more weight

# Combine content, genres, and tags, giving tags more weight
df['combined_features'] = df['content'] + " " + df['genres'] + " " + (df['tags'] * tag_weightage_multiplier)

# 3. Text Vectorization using Tfidf (Term Frequency-Inverse Document Frequency)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# 4. Calculate Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 5. Function to recommend similar shows based on a given show name and display the similarity index
def recommend_shows(show_name, top_n=10):
    # Find the index of the given show
    idx = df[df['name'] == show_name].index[0]

    # Get pairwise similarity scores for all shows with the given show
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the shows based on similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top_n similar shows (excluding the show itself)
    sim_scores = sim_scores[1:top_n + 1]

    # Get the indices of the top_n similar shows
    show_indices = [i[0] for i in sim_scores]

    # Get the recommended shows (including rating, similarity index, and other details)
    recommended_shows = df.iloc[show_indices][['name', 'rating', 'genres', 'tags', 'content']]

    # Add similarity score to the DataFrame
    recommended_shows['similarity_index'] = [sim_scores[i][1] for i in range(top_n)]

    # Sort the recommended shows by rating in descending order
    recommended_shows = recommended_shows.sort_values(by='rating', ascending=False)

    return recommended_shows



recommended_shows = recommend_shows("hidden love", top_n=10)
print(recommended_shows)