# I first load the dataset and clean the data by filling missing values and
# normalizing the show names, tags, and genres (making them lowercase and removing extra spaces).
# Then, I applied TF-IDF vectorization separately to the content, genres, and tags columns, and compute
# the cosine similarity for each feature.
# The recommend_shows function allows me to input a show name and  adjust the weights for content, genres, and tags
# to influence the final similarity score. By combining these weighted similarities, I can sort the shows by
# overall similarity and return the top N recommendations,with an option to sort them by rating. For example,
# when I input "Hotel Del Luna," the script prints the top 10 recommended shows, with tags given a higher weight
# in the similarity calculation.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('./dataset/Top_5000_popular_drama_details_from_mydramalist.csv')

# Fill missing values
df['content'] = df['content'].fillna('')
df['genres'] = df['genres'].fillna('')
df['tags'] = df['tags'].fillna('')
df['name'] = df['name'].fillna('')

# Normalize show names for case-insensitive lookup
df['name_lower'] = df['name'].str.lower()

# Normalizing and cleaning tags:
# Converting to lowercase
# Removing extra spaces
# Replacing  ", " or "," with space to treat each tag as a distinct token
df['tags_cleaned'] = df['tags'].str.lower().str.replace(', ', ' ', regex=False).str.replace(',', ' ', regex=False)

# Normalize genres similarly
df['genres_cleaned'] = df['genres'].str.lower().str.replace(', ', ' ', regex=False).str.replace(',', ' ', regex=False)

# Create separate TF-IDF vectorizers
tfidf_vectorizer_content = TfidfVectorizer(stop_words='english')
tfidf_vectorizer_genres = TfidfVectorizer(token_pattern=r"(?u)\b[\w/]+\b")
tfidf_vectorizer_tags = TfidfVectorizer(token_pattern=r"(?u)\b[\w/]+\b")

# Vectorize each feature
content_tfidf = tfidf_vectorizer_content.fit_transform(df['content'])
genres_tfidf = tfidf_vectorizer_genres.fit_transform(df['genres_cleaned'])
tags_tfidf = tfidf_vectorizer_tags.fit_transform(df['tags_cleaned'])

# Compute cosine similarities
cosine_sim_content = cosine_similarity(content_tfidf, content_tfidf)
cosine_sim_genres = cosine_similarity(genres_tfidf, genres_tfidf)
cosine_sim_tags = cosine_similarity(tags_tfidf, tags_tfidf)

# Recommendation function
def recommend_shows(show_name, top_n=10, content_weight=1, genres_weight=1, tags_weight=1):
    show_name = show_name.lower()

    if show_name not in df['name_lower'].values:
        print(f"Show '{show_name}' not found in the dataset.")
        return pd.DataFrame()

    idx = df[df['name_lower'] == show_name].index[0]

    # Get similarity scores
    sim_content = list(enumerate(cosine_sim_content[idx]))
    sim_genres = list(enumerate(cosine_sim_genres[idx]))
    sim_tags = list(enumerate(cosine_sim_tags[idx]))

    # Combine similarities
    weighted_sim = []
    for i in range(len(df)):
        total_similarity = (sim_content[i][1] * content_weight +
                            sim_genres[i][1] * genres_weight +
                            sim_tags[i][1] * tags_weight)
        weighted_sim.append((i, total_similarity))

    # Sorting by similarity
    weighted_sim = sorted(weighted_sim, key=lambda x: x[1], reverse=True)

    # Get top N (excluding itself)
    top_similar_shows = weighted_sim[1:top_n + 1]
    show_indices = [i[0] for i in top_similar_shows]

    # Extract show info
    recommended_shows = df.iloc[show_indices][['name', 'rating', 'genres', 'tags', 'content']].copy()
    recommended_shows['total_similarity'] = [i[1] for i in top_similar_shows]

    # Sorting by rating
    recommended_shows = recommended_shows.sort_values(by='rating', ascending=False)

    return recommended_shows

show = "hotel del luna"
recommended = recommend_shows(show, top_n=10, content_weight=0.15, genres_weight=0.15, tags_weight=0.7)
print(recommended)
