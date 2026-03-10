import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Movie Recommender AI", page_icon=":)", layout="wide")
similarity_features = [
    'runtime', 'genre_Action', 'genre_Adventure', 'genre_Animation', 
    'genre_Comedy', 'genre_Crime', 'genre_Documentary', 'genre_Drama', 
    'genre_Family', 'genre_Fantasy', 'genre_History', 'genre_Horror', 
    'genre_Music', 'genre_Mystery', 'genre_Romance', 'genre_Science Fiction', 
    'genre_TV Movie', 'genre_Thriller', 'genre_Unknown', 'genre_War', 
    'genre_Western', 'writer_exp_rating', 'director_exp_rating', 
    'cast_exp_rating', 'movie_age', 'popularity_log']
xgb_features = [
    'runtime', 'imdb_rating', 'cast_size', 'financial_status',
    'vote_count_log', 'popularity_log', 'genre_Action', 'genre_Adventure',
    'genre_Animation', 'genre_Comedy', 'genre_Crime', 'genre_Documentary',
    'genre_Drama', 'genre_Family', 'genre_Fantasy', 'genre_History',
    'genre_Horror', 'genre_Music', 'genre_Mystery', 'genre_Romance',
    'genre_Science Fiction', 'genre_TV Movie', 'genre_Thriller',
    'genre_Unknown', 'genre_War', 'genre_Western', 'writer_avg_score',
    'writer_exp_rating', 'director_exp_rating', 'cast_exp_rating',
    'production_companies_exp_rating', 'production_countries_exp_rating',
    'spoken_languages_exp_rating', 'movie_age']

def get_poster_html(title, path):
    base_url = "https://image.tmdb.org/t/p/w200"
    img_url = f"{base_url}{path}" if path and str(path) != 'nan' else "https://via.placeholder.com/200x300?text=No+Poster"
    return f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="{img_url}" style="border-radius: 10px; width: 150px; box-shadow: 2px 2px 10px rgba(0,0,0,0.2);">
        <p style="font-family: Arial; font-size: 14px; font-weight: bold; margin-top: 10px;">{title}</p>
    </div>
    """
@st.cache_resource
def load_assets():
    df = pd.read_csv('movies_data.csv.gz', compression='gzip') 
    xgb_model = joblib.load('xgb_model.pkl')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[similarity_features])
    nn_model = NearestNeighbors(n_neighbors=50, metric='cosine')
    nn_model.fit(scaled_data)
    return df, xgb_model, tfidf_matrix, nn_model, scaler

try:
    df, xgb_model, tfidf_matrix, nn_model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

def get_recommendations(movie_title, min_rating):
    if movie_title not in df['title'].values:
        return pd.DataFrame()
    
    movie_idx = df[df['title'] == movie_title].index[0]
    source_movie_data = df.loc[[movie_idx], similarity_features]
    source_scaled = scaler.transform(source_movie_data)
    
    distances, indices = nn_model.kneighbors(source_scaled)
    candidates = df.iloc[indices[0]].copy()
    candidates['similarity_score'] = 1 - distances[0]

    pos = df.index.get_loc(movie_idx)
    candidate_positions = [df.index.get_loc(idx) for idx in candidates.index]
    plot_sim = cosine_similarity(tfidf_matrix[pos], tfidf_matrix[candidate_positions])
    candidates['plot_sim'] = plot_sim[0]
    candidates['predicted_rating'] = xgb_model.predict(candidates[xgb_features])
    candidates['final_rank'] = (candidates['plot_sim'] + 0.1) * candidates['similarity_score'] * (candidates['predicted_rating'] / 10)
    
    res = candidates[(candidates['vote_average'] >= min_rating) & (candidates['title'] != movie_title)]
    return res.sort_values(by='final_rank', ascending=False).head(5)

st.title("Movie Recommender ")

with st.sidebar:
    st.header("Settings")
    min_rating = st.slider("Minimum Rating", 5.0, 9.0, 6.5, 0.1)

movie_titles = sorted(df['title'].unique())
selected_movie = st.selectbox("Select a movie you liked:", [""] + movie_titles)

if st.button("Get Recommendations"):
    if selected_movie:
        res = get_recommendations(selected_movie, min_rating)
        if res.empty:
            st.warning(f"No results found for '{selected_movie}' above rating {min_rating}.")
        else:
            st.subheader(f"Top Picks for you:")
            cols = st.columns(5)
            for i, (idx, row) in enumerate(res.iterrows()):
                with cols[i]:
                    st.markdown(get_poster_html(row['title'], row['poster_path']), unsafe_allow_html=True)
            st.dataframe(res[['title', 'vote_average', 'predicted_rating']])
    else:
        st.info("Please select a movie from the list.")