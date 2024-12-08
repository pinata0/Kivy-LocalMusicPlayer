import os
import pandas as pd
import numpy as np
import librosa
from collections import defaultdict
from transformers import pipeline
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import List


def algorithm(songs_path, lyrics_path,
            singers=[1, 2, 3, 2, 3, 3, 1, 2, 3, 3, 4, 4, 1, 4, 4],
            user_state_params={'emotion': 0.9, 'activity': 1.0, 'time_of_day': 0.7, 'weather': 0.3},
            weights=np.array([1.0, 1.0, 1.0, 1.0])):

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    class Query:
        def __init__(self, t: int, p: int, s: int):
            self.t = t
            self.p = p
            self.s = s

        def __lt__(self, other):
            return self.t < other.t

    def dfs(cur: int, adj: List[List[int]], visited: List[bool], in_time: List[int], out_time: List[int], sz: List[int], cnt: List[int]) -> int:
        visited[cur] = True
        cnt[0] += 1
        in_time[cur] = cnt[0]
        sz[cur] = 1
        for nxt in adj[cur]:
            if not visited[nxt]:
                sz[cur] += dfs(nxt, adj, visited, in_time, out_time, sz, cnt)
        out_time[cur] = cnt[0]
        return sz[cur]

    def extract_features(songs_path, lyrics_path):
        songs_data = []
        for song_file in os.listdir(songs_path):
            if song_file.endswith('.mp3'):
                song_name = os.path.splitext(song_file)[0]
                song_file_path = os.path.join(songs_path, song_file)
                try:
                    y, sr = librosa.load(song_file_path)
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                    key = np.argmax(np.sum(chroma, axis=1))
                    rms = librosa.feature.rms(y=y).mean()
                except Exception as e:
                    print(f"Error processing {song_file}: {e}")
                    tempo, key, rms = 0, 0, 0.5

                lyrics_file_path = os.path.join(lyrics_path, song_name + '.txt')
                try:
                    lyrics = open(lyrics_file_path).read()
                    device = 0 if torch.cuda.is_available() else -1
                    sentiment_pipeline = pipeline("sentiment-analysis", model="beomi/kcbert-base", framework="pt",
                                                  device=device)
                    sentiment = sentiment_pipeline(lyrics[:300])[0]
                    sentiment_score = 1.0 if sentiment['label'] == 'POSITIVE' else -1.0
                except Exception as e:
                    print(f"Error processing lyrics for {song_file}: {e}")
                    sentiment_score = 0.0

                songs_data.append({
                    'Song': song_name,
                    'Tempo': tempo,
                    'Key': key,
                    'Energy': rms,
                    'Sentiment': sentiment_score
                })

        songs_df = pd.DataFrame(songs_data)
        return songs_df

    def calculate_similarity_matrix(songs_df, user_state_params):

        num_songs = len(songs_df)
        distance_matrix = np.zeros((num_songs, num_songs))

        # 사용자 상태에 따른 가중치 정의
        weight_tempo = user_state_params['activity']  # 활동량에 따라 템포 가중치
        weight_key = user_state_params['time_of_day']  # 시간대에 따른 키 가중치
        weight_energy = user_state_params['weather']  # 날씨에 따른 에너지 가중치
        weight_sentiment = user_state_params['emotion']  # 감정에 따른 감성 가중치

        # 특성별 가중치 배열 생성
        feature_weights = np.array([weight_tempo, weight_key, weight_energy, weight_sentiment])

        for i in range(num_songs):
            for j in range(i + 1, num_songs):
                feature1 = songs_df.iloc[i][['Tempo', 'Key', 'Energy', 'Sentiment']].values
                feature2 = songs_df.iloc[j][['Tempo', 'Key', 'Energy', 'Sentiment']].values
                # 각 특성의 차이를 가중치와 함께 반영
                weighted_difference = feature_weights * (feature1 - feature2)
                distance = np.linalg.norm(weighted_difference)  # 유클리드 거리 계산
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix

    def generate_edges_from_similarity(distance_matrix):
        mst = minimum_spanning_tree(distance_matrix).toarray().astype(float)
        edges = []
        for i in range(len(mst)):
            for j in range(i + 1, len(mst)):
                if mst[i, j] > 0:
                    edges.append((i + 1, j + 1))
        return edges

    def update(tree: List[int], x: int, v: int, n: int):
        while x <= n:
            tree[x] += v
            x += x & -x

    def range_update(tree: List[int], l: int, r: int, v: int, n: int):
        update(tree, l, v, n)
        update(tree, r + 1, -v, n)

    # 곡 데이터 추출
    songs_df = extract_features(songs_path, lyrics_path)
    if songs_df.empty:
        raise ValueError("The songs dataframe is empty. Check the song data extraction.")

    # edges 생성
    distance_matrix = calculate_similarity_matrix(songs_df, user_state_params)
    edges = generate_edges_from_similarity(distance_matrix)

    # 곡 추천 알고리즘
    n = len(songs_df)
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    in_time = [0] * (n + 1)
    out_time = [0] * (n + 1)
    sz = [0] * (n + 1)
    visited = [False] * (n + 1)
    cnt = [0]

    dfs(1, adj, visited, in_time, out_time, sz, cnt)

    nums = [0] * (n + 1)
    for i in range(1, n + 1):
        nums[singers[i - 1]] += 1

    tree = [0] * (n + 1)
    for i in range(1, n + 1):
        range_update(tree, in_time[i], out_time[i], weights[i % len(weights)], n)

    recommended_songs = [songs_df['Song'][i - 1] + '.mp3' for i in range(1, n + 1)]
    return recommended_songs

if __name__ == "__main__":
        
    # 사용 예시
    songs_path = './data/songs'
    lyrics_path = './data/lyrics'
    singers = [1, 2, 3, 2, 3, 3, 1, 2, 3, 3, 4, 4, 1, 4, 4]
    user_state_params = {'emotion': 0.9, 'activity': 1.0, 'time_of_day': 0.7, 'weather': 0.3}
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    recommended_songs = algorithm(songs_path, lyrics_path, singers, user_state_params, weights)
    print(f"Recommended Songs: {recommended_songs}")
