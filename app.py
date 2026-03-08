import streamlit as st
import pandas as pd
import numpy as np
import joblib # לטעינת המודל השמור
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# --- הגדרות דף ---
st.set_page_config(page_title="Movie Recommender AI", page_icon="🎬", layout="wide")

# --- פונקציית עזר לפוסטרים (Streamlit תומכת ב-HTML בסיסי) ---
def get_poster_html(title, path):
    base_url = "https://image.tmdb.org/t/p/w200"
    img_url = f"{base_url}{path}" if path and str(path) != 'nan' else "https://via.placeholder.com/200x300?text=No+Poster"
    return f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="{img_url}" style="border-radius: 10px; width: 150px; box-shadow: 2px 2px 10px rgba(0,0,0,0.2);">
        <p style="font-family: Arial; font-size: 14px; font-weight: bold; margin-top: 10px;">{title}</p>
    </div>
    """

# --- טעינת נתונים ומודלים (Caching) ---
@st.cache_resource
def load_assets():
    # כאן עליך לטעון את הקבצים שהעלית ל-GitHub
    df = pd.read_csv('df_cleaned_with_overview.csv') # וודא שזה השם
    xgb_model = joblib.load('xgb_model.pkl') # וודא ששמרת את המודל כ-pkl
    tfidf_matrix = joblib.load('tfidf_matrix.pkl') # וודא ששמרת את המטריצה
    return df, xgb_model, tfidf_matrix

# נסה לטעון, אם הקבצים חסרים תוצג הודעה
try:
    df_cleaned, xgb_model, tfidf_matrix = load_assets()
except:
    st.error("Missing data files! Make sure to upload CSV and Model files to GitHub.")
    st.stop()

# --- לוגיקת ההמלצה (הקוד המקורי שלך) ---
def build_recommendation(movie_title, min_rating):
    # (כאן נכנסת כל פונקציית build_ultimate_recommendation שכתבת)
    # ... לוגיקת KNN, TF-IDF ו-XGBoost ...
    # חזרה על הנוסחה המכפלתית: (plot_sim + 0.1) * similarity_score * norm_predicted * age_factor
    pass 

# --- ממשק המשתמש של Streamlit ---
st.title("🎬 AI Movie Recommender")
st.markdown("Find your next favorite movie using Machine Learning.")

# Sidebar להגדרות
with st.sidebar:
    st.header("Settings")
    min_rating = st.slider("⭐ Minimum Rating", 5.0, 9.0, 6.5, 0.1)

# חיפוש סרט
movie_titles = sorted(df_cleaned['title'].unique())
selected_movie = st.selectbox("🔍 Search for a movie you liked:", [""] + movie_titles)

if st.button("Get Recommendations!"):
    if selected_movie == "":
        st.warning("Please select a movie first.")
    else:
        with st.spinner('Thinking...'):
            res = build_recommendation(selected_movie, min_rating)
            
            if res.empty:
                st.info(f"No movies found above {min_rating}. Try lowering the threshold!")
            else:
                st.subheader(f"✨ Recommendations for {selected_movie}:")
                
                # הצגה בטורים (Grid)
                cols = st.columns(5)
                for idx, row in res.reset_index().iterrows():
                    with cols[idx]:
                        # שימוש במילון הפוסטרים
                        p_path = df_cleaned[df_cleaned['title'] == row['title']]['poster_path'].values[0]
                        st.markdown(get_poster_html(row['title'], p_path), unsafe_allow_html=True)
                
                st.write("---")
                st.dataframe(res) # הצגת הטבלה מתחת