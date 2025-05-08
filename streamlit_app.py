import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('dataset.csv', sep='\t', header=None, 
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])
    return data

# Preprocess data
def preprocess_data(data):
    # Remove duplicates if any
    data = data.drop_duplicates(['user_id', 'movie_id'])
    
    # Create user-item matrix
    user_item_matrix = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    
    return user_item_matrix

# Train SVD model
@st.cache_resource
def train_model(data):
    # Prepare data for Surprise
    reader = Reader(rating_scale=(1, 5))
    data_surprise = Dataset.load_from_df(data[['user_id', 'movie_id', 'rating']], reader)
    
    # Split into train and test set
    trainset = data_surprise.build_full_trainset()
    
    # Train SVD algorithm
    algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
    algo.fit(trainset)
    
    return algo

# Get top movie recommendations
def get_recommendations(algo, data, user_id, n_recommendations=5):
    # Get list of all movie IDs
    all_movie_ids = data['movie_id'].unique()
    
    # Get movies the user has already rated
    rated_movies = data[data['user_id'] == user_id]['movie_id'].unique()
    
    # Predict ratings for unrated movies
    recommendations = []
    for movie_id in all_movie_ids:
        if movie_id not in rated_movies:
            pred = algo.predict(user_id, movie_id).est
            recommendations.append((movie_id, pred))
    
    # Sort by predicted rating
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N recommendations
    return recommendations[:n_recommendations]

# Streamlit app
def main():
    st.title("🎬 Movie Recommendation System")
    st.write("""
    This system recommends movies based on your past ratings and similar users' preferences.
    """)
    
    # Load data
    data = load_data()
    
    # Preprocess data
    user_item_matrix = preprocess_data(data)
    
    # Train model
    algo = train_model(data)
    
    # Sidebar for user input
    st.sidebar.header("User Preferences")
    
    # Get unique user IDs
    user_ids = data['user_id'].unique()
    selected_user = st.sidebar.selectbox("Select User ID", user_ids)
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider("Number of Recommendations", 1, 10, 5)
    
    # Show user's rated movies
    st.subheader(f"Movies Rated by User {selected_user}")
    user_ratings = data[data['user_id'] == selected_user][['movie_id', 'rating']]
    st.dataframe(user_ratings)
    
    # Get recommendations
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(algo, data, selected_user, n_recommendations)
        
        st.subheader(f"Top {n_recommendations} Recommended Movies")
        for i, (movie_id, pred_rating) in enumerate(recommendations, 1):
            st.write(f"{i}. Movie ID: {movie_id} (Predicted Rating: {pred_rating:.2f})")
    
    # Show some statistics
    st.sidebar.subheader("Dataset Statistics")
    st.sidebar.write(f"Total Users: {len(user_ids)}")
    st.sidebar.write(f"Total Movies: {len(data['movie_id'].unique())}")
    st.sidebar.write(f"Total Ratings: {len(data)}")

if __name__ == "__main__":
    main()
