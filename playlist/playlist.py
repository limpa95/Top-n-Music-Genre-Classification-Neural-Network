from flask import Flask, jsonify, render_template
import os
import json
import time
import spotipy  # type: ignore
from spotipy.oauth2 import SpotifyClientCredentials  # type: ignore

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)

CLIENT_ID = '53ce03981b224f6390e23c33329b67aa'
CLIENT_SECRET = 'cd9881b784e74c94abbec986d0438b06'
redirect_uri = 'http://127.0.0.1:5000'  # Redirect URI for Spotify API


def get_tracks_by_genre(genre, limit=10):
    # Authenticate with the Spotify API
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

    # Search for tracks based on genre
    results = sp.search(q=f'genre:{genre}', type='track', limit=limit)
    tracks = results['tracks']['items']
    return tracks


def read_genre_from_json(file_path):
    '''searches for the genre in the prediction.json file'''
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data.get('genre')
    except FileNotFoundError:
        print(f"File {file_path} not found. Continuing to search...")
        return None


def save_tracks_to_json(tracks, file_path):
    '''saves the playlist to the playlist.json file'''
    filtered_tracks = []
    for track in tracks:
        if track:
            track_info = {
                'name': track.get('name'),
                'artists': [
                    artist.get('name') for artist in track.get('artists', [])
                ]
            }
            filtered_tracks.append(track_info)
    with open(file_path, 'w') as file:
        json.dump(filtered_tracks, file, indent=4)


def update_prediction_json(file_path, genre):
    '''updates the prediction.json file with the new playlist'''
    try:
        with open(file_path, 'r+') as file:
            data = json.load(file)
            data['genre'] = genre
            file.seek(0)  # Reset file pointer to the beginning
            json.dump(data, file, indent=4)
            file.truncate()  # Truncate the file to remove any leftover data
    except FileNotFoundError:
        print(f"File {file_path} not found. Creating a new file...")
        with open(file_path, 'w') as file:
            json.dump({'genre': genre}, file, indent=4)


@app.route('/')
def main():
    input_file = 'prediction.json'
    output_file = 'playlist.json'
    last_modified_time = None

    while True:
        # search for 'prediction.json' file to read the genre
        genre = read_genre_from_json(input_file)
        if not genre:
            print("No genre found in the input file. Continuing to search...")
            continue

        tracks = get_tracks_by_genre(genre)
        print(f'Top {len(tracks)} tracks in the {genre} genre:')
        for track in tracks:
            if track:  # Check if track is not None
                name = track.get('name')
                artist = track.get('artists', [{}])[0].get('name')
                if name and artist:
                    print(f"{name} by {artist}")
        # Save the playlist(track) to a playlist.json file
        save_tracks_to_json(tracks, output_file)
        print(f"Tracks saved to {output_file}")

        # Check if the playlist.json file has been updated
        current_modified_time = os.path.getmtime(output_file)
        if (last_modified_time is None or
                current_modified_time > last_modified_time):
            last_modified_time = current_modified_time
            update_prediction_json(input_file, genre)
            print(f"Prediction file updated with new genre: {genre}")

        # Wait for a specified amount of time before continuing the loop
        time.sleep(60)  # Wait for 60 seconds

    return render_template('index.html')


@app.route('/playlist')
def get_playlist():
    output_file = 'playlist.json'
    try:
        with open(output_file, 'r') as file:
            data = json.load(file)
            return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "Playlist file not found"}), 404


if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app