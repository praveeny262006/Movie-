import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('dataset.csv', sep='\t', header=None, 
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])
    return data

# Preprocess data
def preprocess_data(data):
    data = data.drop_duplicates(['user_id', 'movie_id'])
    user_item_matrix = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    return user_item_matrix

# Train SVD model using scipy
@st.cache_resource
def train_model(user_item_matrix):
    # Convert to numpy array
    matrix = user_item_matrix.values

    # Mean center the data (important for SVD)
    user_ratings_mean = np.mean(matrix, axis=1)
    matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)

    # Perform SVD
    U, sigma, Vt = svds(matrix_demeaned, k=50)
    sigma = np.diag(sigma)

    # Reconstruct predictions
    predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

    return predicted_ratings, user_item_matrix

# Get recommendations
def get_recommendations(predicted_ratings, user_item_matrix, user_id, n_recommendations=5):
    user_idx = np.where(user_item_matrix.index == user_id)[0][0]
    user_ratings = predicted_ratings[user_idx]

    rated_movie_ids = user_item_matrix.columns[user_item_matrix.iloc[user_idx] > 0].tolist()
    recommendations = []

    for i, movie_id in enumerate(user_item_matrix.columns):
        if movie_id not in rated_movie_ids:
            recommendations.append((movie_id, user_ratings[i]))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n_recommendations]

# Streamlit app
def main():
    st.title("🎬 Movie Recommendation System (No Surprise Library)")
    st.write("This system recommends movies using collaborative filtering (SVD without `surprise`).")

    data = load_data()
    user_item_matrix = preprocess_data(data)
    predicted_ratings, user_item_matrix = train_model(user_item_matrix)

    st.sidebar.header("User Preferences")
    user_ids = user_item_matrix.index.tolist()
    selected_user = st.sidebar.selectbox("Select User ID", user_ids)
    n_recommendations = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

    st.subheader(f"Movies Rated by User {selected_user}")
    user_ratings = data[data['user_id'] == selected_user][['movie_id', 'rating']]
    st.dataframe(user_ratings)

    if st.button("Get Recommendations"):
        recommendations = get_recommendations(predicted_ratings, user_item_matrix, selected_user, n_recommendations)
        st.subheader(f"Top {n_recommendations} Recommended Movies")
        for i, (movie_id, pred_rating) in enumerate(recommendations, 1):
            st.write(f"{i}. Movie ID: {movie_id} (Predicted Rating: {pred_rating:.2f})")

    st.sidebar.subheader("Dataset Statistics")
    st.sidebar.write(f"Total Users: {len(user_item_matrix.index)}")
    st.sidebar.write(f"Total Movies: {len(user_item_matrix.columns)}")
    st.sidebar.write(f"Total Ratings: {len(data)}")

if __name__ == "__main__":
    main()
