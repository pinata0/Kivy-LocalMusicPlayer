import os
import pandas as pd
import numpy as np
import librosa
from datetime import datetime
from transformers import pipeline
import torch

def main(songs_path, lyrics_path, user_state_params, weights):
    """
    Process songs and lyrics, calculate recommendations, and return a list of recommended song names.
    
    Args:
        songs_path (str): Path to the songs directory.
        lyrics_path (str): Path to the lyrics directory.
        user_state_params (dict): User state parameters (emotion, activity, time_of_day, weather).
        weights (np.ndarray): Weights for recommendation calculation.
    
    Returns:
        list: List of recommended song names.
    """
    def extract_tempo(filepath):
        try:
            y, sr = librosa.load(filepath)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return tempo
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return 0

    def extract_key(filepath):
        try:
            y, sr = librosa.load(filepath)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            pitch_class = np.argmax(np.sum(chroma, axis=1))
            pitch_to_key = {
                0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
                6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"
            }
            return pitch_to_key[pitch_class]
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return "Unknown"

    def extract_energy(filepath):
        try:
            y, sr = librosa.load(filepath)
            rms = librosa.feature.rms(y=y)
            rms_mean = float(np.mean(rms))
            return {'RMS': rms_mean}
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return {'RMS': 0.0}

    def analyze_sentiment(lyrics):
        try:
            device = 0 if torch.cuda.is_available() else -1
            sentiment_pipeline = pipeline("sentiment-analysis", model="beomi/kcbert-base", framework="pt", device=device)
            truncated_lyrics = lyrics[:300]
            result = sentiment_pipeline(truncated_lyrics)[0]
            return 1.0 if result['label'] == 'POSITIVE' else -1.0
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0.0

    def process_songs_and_lyrics(songs_path, lyrics_path):
        songs_data = []
        for song_file in os.listdir(songs_path):
            if song_file.endswith('.mp3'):
                song_name = os.path.splitext(song_file)[0]
                song_file_path = os.path.join(songs_path, song_file)
                tempo = extract_tempo(song_file_path)
                key = extract_key(song_file_path)
                energy = extract_energy(song_file_path).get('RMS', 0.5)
                lyrics_file_path = os.path.join(lyrics_path, song_name + '.txt')
                sentiment = analyze_sentiment(open(lyrics_file_path).read()) if os.path.exists(lyrics_file_path) else 0
                songs_data.append({
                    'Song': song_name,
                    'Tempo': tempo,
                    'Key': key,
                    'Energy': energy,
                    'Sentiment': sentiment
                })

        songs_df = pd.DataFrame(songs_data)
        songs_df['Tempo'] = pd.to_numeric(songs_df['Tempo'], errors='coerce').fillna(0.0)
        songs_df['Key'] = pd.to_numeric(songs_df['Key'], errors='coerce').fillna(0.0)
        songs_df['Energy'] = pd.to_numeric(songs_df['Energy'], errors='coerce').fillna(0.0)
        songs_df['Sentiment'] = pd.to_numeric(songs_df['Sentiment'], errors='coerce').fillna(0.0)
        return songs_df

    def calculate_distance(song_features, user_features, weights):
        song_features = np.array(song_features, dtype=np.float64)
        user_features = np.array(user_features, dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        return np.sqrt(np.sum(weights * (song_features - user_features) ** 2))

    def recommend_songs(songs_df, user_state, weights):
        recommendations = []
        for _, row in songs_df.iterrows():
            try:
                song_features = np.array([row['Tempo'], row['Key'], row['Energy'], row['Sentiment']], dtype=np.float64)
                dist = calculate_distance(song_features, user_state.features, weights)
                recommendations.append((row['Song'], dist))
            except Exception as e:
                print(f"Error processing row: {row}. Error: {e}")
        recommendations = sorted(recommendations, key=lambda x: x[1])
        return [rec[0] for rec in recommendations]  # Return only song names

    class UserState:
        def __init__(self, emotion, activity, time_of_day, weather):
            self.features = np.array([emotion, activity, time_of_day, weather])

    # Create user state
    user_state = UserState(**user_state_params)

    # Process songs and lyrics
    songs_dataset = process_songs_and_lyrics(songs_path, lyrics_path)

    # Get recommended songs
    return recommend_songs(songs_dataset, user_state, weights)


    # 좋아요 버튼 안 만듬
    # def update_weights(weights, feedback, song_features, user_features, learning_rate=0.1):
    #     """조아용 기준 가중치"""
    #     if feedback == "like":
    #         weights -= learning_rate * np.abs(song_features - user_features)
    #     elif feedback == "dislike":
    #         weights += learning_rate * np.abs(song_features - user_features)
    #     return np.clip(weights, 0.1, 1)  # 가중치 제한 (0.1 ~ 1)

def algorithm(songs_path, lyrics_path):

    # Get the current time
    now = datetime.now()

    # Calculate the total minutes in a day
    total_minutes_in_a_day = 24 * 60  # 1440 minutes

    # Calculate the current minutes since midnight
    current_minutes = now.hour * 60 + now.minute

    # Normalize to a range of 0 to 1
    normalized_time_of_day = current_minutes / total_minutes_in_a_day


    # User state parameters
    user_state_params = {
        'emotion': 0.9,     # 감정 (기쁨)
        'activity': 1.0,    # 활동 (운동)
        'time_of_day': round(normalized_time_of_day, 1), # 저녁
        'weather': 0.3      # 날씨 (흐림)
    }

    # Weights for features
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    # Call the algorithm
    recommended_songs = main(songs_path, lyrics_path, user_state_params, weights)

    recommended_songs = [name + '.mp3' for name in recommended_songs]
    # Print recommended songs
    print(recommended_songs)
    return recommended_songs

if __name__ == '__main__':
    songs_path = './data/songs'  # Update with your songs directory
    lyrics_path = './data/lyrics'  # Update with your lyrics directory

    algorithm(songs_path, lyrics_path)