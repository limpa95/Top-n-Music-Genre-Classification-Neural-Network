from flask import Flask, jsonify
import os
import json
import time
import threading
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)

# Load environment variables from .env file
load_dotenv()

CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
redirect_uri = 'http://127.0.0.1:5000'  # Redirect URI for Spotify API

observer = None  # Initialize observer as None

# Define the paths for the JSON files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTION_FILE = os.path.join(BASE_DIR, 'prediction.json')
PLAYLIST_FILE = os.path.join(BASE_DIR, 'playlist.json')


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
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file {file_path}.")
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
    try:
        with open(file_path, 'w') as file:
            json.dump(filtered_tracks, file, indent=4)
        print(f"Tracks successfully saved to {file_path}")
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")


class PredictionFileHandler(FileSystemEventHandler):
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def on_modified(self, event):
        if event.src_path == self.input_file:
            print(f"Detected modification in {self.input_file}")
            genre = read_genre_from_json(self.input_file)
            if genre:
                print(f"Genre found: {genre}")
                try:
                    tracks = get_tracks_by_genre(genre)
                    save_tracks_to_json(tracks, self.output_file)
                    print(f"Tracks saved to {self.output_file}")
                except Exception as e:
                    print(f"Error fetching tracks from Spotify: {e}")
            else:
                print("No genre found in the prediction file.")


def start_observer():
    global observer  # Use the global observer

    event_handler = PredictionFileHandler(PREDICTION_FILE, PLAYLIST_FILE)
    observer = Observer()

    observer.schedule(event_handler, path=BASE_DIR, recursive=False)
    observer.start()
    print("Observer started")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def reset_observer():
    global observer  # Use the global observer
    if observer:
        observer.stop()
        observer.join()
    start_observer()


@app.route('/')
def main():
    return "Welcome to the Music Genre Classification App"


@app.route('/playlist')
def get_playlist():
    try:
        with open(PLAYLIST_FILE, 'r') as file:
            data = json.load(file)
            return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "Playlist file not found"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding JSON from playlist file"}), \
               500


if __name__ == '__main__':
    observer_thread = threading.Thread(target=start_observer)
    observer_thread.daemon = True
    observer_thread.start()
    app.run(debug=True)  # Run the Flask app
