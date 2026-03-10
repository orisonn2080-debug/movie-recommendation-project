import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.markdown("""<style>.stApp a:has(svg) {display: none;} </style>""", unsafe_allow_html=True)

st.set_page_config(page_title="Movie Recommender", page_icon="", layout="wide")

similarity_features = ['runtime', 'genre_Action', 'genre_Adventure', 'genre_Animation', 'genre_Comedy', 'genre_Crime', 'genre_Documentary', 'genre_Drama', 'genre_Family', 'genre_Fantasy', 'genre_History', 'genre_Horror', 'genre_Music', 'genre_Mystery', 'genre_Romance', 'genre_Science Fiction', 'genre_TV Movie', 'genre_Thriller', 'genre_Unknown', 'genre_War', 'genre_Western', 'writer_exp_rating', 'director_exp_rating', 'cast_exp_rating', 'movie_age', 'popularity_log']
xgb_features = ['runtime', 'imdb_rating', 'cast_size', 'financial_status', 'vote_count_log', 'popularity_log', 'genre_Action', 'genre_Adventure', 'genre_Animation', 'genre_Comedy', 'genre_Crime', 'genre_Documentary', 'genre_Drama', 'genre_Family', 'genre_Fantasy', 'genre_History', 'genre_Horror', 'genre_Music', 'genre_Mystery', 'genre_Romance', 'genre_Science Fiction', 'genre_TV Movie', 'genre_Thriller', 'genre_Unknown', 'genre_War', 'genre_Western', 'writer_avg_score', 'writer_exp_rating', 'director_exp_rating', 'cast_exp_rating', 'production_companies_exp_rating', 'production_countries_exp_rating', 'spoken_languages_exp_rating', 'movie_age']

def get_poster_html(title, path):
    base_url = "https://image.tmdb.org/t/p/w200"
    img_url = f"{base_url}{path}" if path and str(path) != 'nan' else "https://via.placeholder.com/200x300?text=No+Poster"
    return f'<div style="text-align: center;"><img src="{img_url}" style="width: 150px; border-radius: 10px;"><p><b>{title}</b></p></div>'

@st.cache_resource
def load_assets():
    df_raw = pd.read_csv('movies_data.csv.gz', compression='gzip')
    df = df_raw[df_raw['vote_count'] >= 50].copy().reset_index(drop=True)
    model = joblib.load('xgb_model.pkl')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[similarity_features])
    nn_model = NearestNeighbors(n_neighbors=50, metric='cosine')
    nn_model.fit(scaled_data)
    return df, model, tfidf_matrix, nn_model, scaler

df, xgb_model, tfidf_matrix, nn_model, scaler = load_assets()

def get_recommendations(movie_title, min_rating, num_recommendations):
    if movie_title not in df['title'].values:
        return pd.DataFrame()
    movie_idx = df[df['title'] == movie_title].index[0]
    genre_cols = [col for col in df.columns if col.startswith('genre_')]
    source_genres = df.loc[movie_idx, genre_cols]
    active_genres = source_genres[source_genres == 1].index.tolist()
    source_scaled = scaler.transform(df.loc[[movie_idx], similarity_features])
    distances, indices = nn_model.kneighbors(source_scaled)
    candidates = df.iloc[indices[0]].copy()
    candidates['similarity_score'] = 1 - distances[0]
    if active_genres:
        candidates['genre_match_score'] = candidates[active_genres].sum(axis=1) / len(active_genres)
    else:
        candidates['genre_match_score'] = 0
    plot_sim = cosine_similarity(tfidf_matrix[movie_idx], tfidf_matrix[indices[0]])
    candidates['plot_sim'] = plot_sim[0]
    candidates['predicted_rating'] = np.minimum(xgb_model.predict(candidates[xgb_features]), 10.0)
    candidates['final_rank'] = ((candidates['plot_sim'] * 0.3) + (candidates['similarity_score'] * 0.4) + (candidates['genre_match_score'] * 0.2) + ((candidates['predicted_rating'] / 10) * 0.1))
    res = candidates[(candidates['vote_average'] >= min_rating) & (candidates['title'] != movie_title)]
    return res.sort_values(by='final_rank', ascending=False).head(num_recommendations)

st.title("Movie Recommender")
selected_movie = st.selectbox("Select a movie you liked:", [""] + sorted(df['title'].unique().tolist()))

st.sidebar.header("Settings")
min_rating = st.sidebar.slider("Minimum Rating for recommendations", 5.0, 9.5, 6.5)
num_rec = st.sidebar.slider("Number of recommendations", 3, 20, 5)

if st.button("Get Recommendations"):
    if selected_movie:
        source_row = df[df['title'] == selected_movie].iloc[0]
        st.markdown("---")
        st.subheader(f"You selected: {selected_movie}")
        col_src1, col_src2 = st.columns([1, 3])
        with col_src1:
            st.markdown(get_poster_html(source_row['title'], source_row['poster_path']), unsafe_allow_html=True)
        with col_src2:
            st.write(f"**Actual Rating:** {source_row['vote_average']}")
            st.write(f"**Popularity Score:** {round(source_row['popularity_log'], 2)}")
            st.write(f"**Runtime:** {source_row['runtime']} min")
            st.write(f"**Overview:** {source_row['overview']}")
        st.markdown("---")
        
        res = get_recommendations(selected_movie, min_rating, num_rec)

        if not res.empty:
            st.subheader(f"Top {len(res)} Picks for you:")
            for i in range(0, len(res), 5):
                cols = st.columns(5)
                for j in range(5):
                    if i + j < len(res):
                        row = res.iloc[i + j]
                        with cols[j]:
                            st.markdown(get_poster_html(row['title'], row['poster_path']), unsafe_allow_html=True)
                            release_year = 2026 - int(row['movie_age'])
                            st.caption(f"Year: {release_year}")
            st.write("### Statistical Comparison:")
            st.table(res[['title', 'vote_average', 'predicted_rating', 'similarity_score']])
        else:
            st.warning("No high-quality matches found for this specific filter.")