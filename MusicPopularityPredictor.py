import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import joblib

# --- Cross-validation import ---
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# --- 1. Load Dataset ---
songs = pd.read_csv('Spotify_data.csv')

# --- Artist Popularity (Sorted) ---
artist_popularity = {
    "Taylor Swift": 98,
    "Drake": 97,
    "Arijit Singh": 95,
    "Diljit Dosanjh": 94,
    "Eminem": 94,
    "AP Dhillon": 93,
    "Kanye West": 93,
    "Badshah": 92,
    "The Weeknd": 91,
    "Shreya Ghoshal": 90,
    "Rihanna": 89,
    "Armaan Malik": 88,
    "Ariana Grande": 88,
    "Honey Singh": 87,
    "BTS": 86,
    "Kendrick Lamar": 85,
    "Shubh": 85,
    "Dhvani Bhanushali": 84,
    "Neha Kakkar": 83,
    "Camila Cabello": 82,
    "King": 82,
    "Selena Gomez": 81,
    "Doja Cat": 80,
    "Ed Sheeran": 79,
    "Billie Eilish": 78,
    "Zayn": 78,
    "Nicki Minaj": 77,
    "Karan Aujla": 76,
    "Shan Vincent De Paul": 75,
    "Marshmello": 74,
    "Divine": 73,
    "Jubin Nautiyal": 72,
    "Adele": 71,
    "Raftaar": 70
}

def get_artist_popularity(artist_name):
    artist_names = [a.strip() for a in artist_name.split(',')]
    scores = [artist_popularity.get(name, 60) for name in artist_names]
    return max(scores)

# --- 2. Define feature columns and target ---
song_feature_cols = [
    'Danceability', 'Energy', 'Valence', 'Speechiness', 'Acousticness',
    'Instrumentalness', 'Tempo', 'Liveness', 'Loudness', 'Key', 'Mode', 'Duration (ms)'
]
target_col = 'Popularity'

songs['ArtistPopularity'] = songs['Artists'].apply(get_artist_popularity)
X = songs[song_feature_cols + ['ArtistPopularity']]
songs = songs.dropna(subset=song_feature_cols + ['ArtistPopularity'] + [target_col])
y = songs[target_col]

# --- 5. Feature scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 6. Model ---
model = XGBRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=0.6,
    random_state=42
)
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
mean_rmse = -scores.mean()
print(f"✅ Cross-Validated RMSE (XGBoost): {mean_rmse:.2f}")
model.fit(X_scaled, y)

# --- Save model ---
joblib.dump(model, "music_popularity_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("💾 Model and scaler saved!")

# --- Clustering ---
kmeans = KMeans(n_clusters=5, random_state=42)
songs['Cluster'] = kmeans.fit_predict(X_scaled)

def hybrid_recommend(user_profile, top_n=10, alpha=0.5):
    user_vec = scaler.transform([user_profile])
    content_scores = cosine_similarity(user_vec, X_scaled)[0]
    user_cluster = kmeans.predict(user_vec)[0]
    collab_scores = (songs['Cluster'] == user_cluster).astype(int)
    hybrid_scores = alpha * content_scores + (1 - alpha) * collab_scores
    top_indices = np.argsort(hybrid_scores)[::-1][:top_n]
    return songs.iloc[top_indices][['Track Name', 'Popularity']]

def predict_popularity(song_features):
    song_scaled = scaler.transform([song_features])
    raw_pred = model.predict(song_scaled)[0]
    return raw_pred

# --- Example Usage ---
liked_song_indices = [0, 1, 2]
user_profile = songs.loc[liked_song_indices, song_feature_cols + ['ArtistPopularity']].mean().values

print("\n🎵 Hybrid Recommendations:")
print(hybrid_recommend(user_profile, top_n=10))

example_song_features = songs.iloc[0][song_feature_cols + ['ArtistPopularity']].values
predicted_pop = predict_popularity(example_song_features)
print(f"\n🔮 Predicted Popularity for Example Song: {predicted_pop:.2f}")
