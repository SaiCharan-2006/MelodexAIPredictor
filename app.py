from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the model and dataset
model_path = os.path.join(os.path.dirname(__file__), "music_popularity_model.pkl")
model = joblib.load(model_path)

# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), "Spotify_data.csv")
df = pd.read_csv(data_path)
# Standardize column names
df.columns = df.columns.str.strip().str.lower()

@app.route("/")
def home():
    return "🎧 Music Popularity Predictor API is running!"

@app.route("/predict_by_name", methods=["POST"])
def predict_by_name():
    data = request.get_json()
    song_name = data.get("song_name", "").strip()

    if not song_name:
        return jsonify({"error": "No song name provided"}), 400

    # Case-insensitive partial match, select only the first match to avoid ambiguity
    matches = df[df['track name'].str.lower().str.contains(song_name.lower(), na=False)].head(1)
    if matches.empty:
        return jsonify({"error": f"Song '{song_name}' not found in dataset"}), 404

    song_row = matches.iloc[0]
    artist_popularity_scores = {
        "Taylor Swift": 95, "Drake": 94, "Eminem": 94, "Ed Sheeran": 93, "Ariana Grande": 92, "BTS": 91,
        "The Weeknd": 90, "Billie Eilish": 89, "Justin Bieber": 88, "Dua Lipa": 87,
        "Bad Bunny": 86, "Olivia Rodrigo": 85, "Post Malone": 84, "Harry Styles": 83,
        "Imagine Dragons": 82, "Marshmello": 81, "Selena Gomez": 80, "Travis Scott": 79,
        "Khalid": 78, "Doja Cat": 77, "Halsey": 76, "Katy Perry": 75, "BLACKPINK": 74,
        "Camila Cabello": 73, "Shawn Mendes": 72, "Zayn": 71, "Anirudh Ravichander": 70,
        "Arijit Singh": 69, "Shreya Ghoshal": 68, "Badshah": 67, "Neha Kakkar": 66,
        "Pritam": 65, "Vishal-Shekhar": 64, "KK": 63, "Sonu Nigam": 62, "Armaan Malik": 61,
        "Jubin Nautiyal": 60, "Honey Singh": 59, "A. R. Rahman": 58
    }
    artist_names = [a.strip() for a in str(song_row['artists']).split(",")]
    artist_popularity = max(
        (artist_popularity_scores.get(artist, 60) for artist in artist_names),
        default=60
    )

    feature_columns = ['danceability', 'energy', 'valence', 'speechiness', 'acousticness',
                       'instrumentalness', 'tempo', 'liveness', 'loudness', 'key', 'mode', 'duration (ms)']

    try:
        features = song_row[feature_columns].values.tolist()
        features.append(artist_popularity)
        features = np.array(features).reshape(1, -1)
    except KeyError as e:
        return jsonify({"error": f"Missing feature column: {str(e)}"}), 500

    predicted_popularity = model.predict(features)[0]

    # Define hit threshold and label
    hit_label = "🔥 This song is a HIT!" if predicted_popularity >= 70 else "🎵 This song is a VIBE!"

    # Return full prediction including artist popularity
    return jsonify({
        "track": song_row['track name'],
        "artist": song_row['artists'],
        "predicted_popularity": int(predicted_popularity),
        "features": {col: float(song_row[col]) if pd.api.types.is_numeric_dtype(type(song_row[col])) else str(song_row[col]) for col in feature_columns},
        "artist_popularity_score": artist_popularity,
        "hit_prediction": hit_label,
    })

if __name__ == "__main__":
    print("📁 Current Working Directory:", os.getcwd())
    app.run(debug=True)