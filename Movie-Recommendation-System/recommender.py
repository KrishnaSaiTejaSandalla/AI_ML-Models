import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("data/movies.csv")

data["combined_features"] = data["genre"] + " " + data["description"]

vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(data["combined_features"])
similarity = cosine_similarity(feature_matrix)

def recommend(movie_title):
    if movie_title not in data["title"].values:
        print("Movie not found!")
        return

    index = data[data["title"] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity[index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print(f"\nMovies similar to {movie_title}:\n")

    for i in sorted_scores[1:4]:
        print(data.iloc[i[0]]["title"])


movie = input("Enter a movie name: ")
recommend(movie)