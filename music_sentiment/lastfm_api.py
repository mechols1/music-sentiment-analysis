import pylast
from dotenv import load_dotenv
import os

def get_lastfm_network():
    """Create and return an authenticated Last.fm network object"""
    load_dotenv()
    
    API_KEY = os.environ.get("LASTFM_API_KEY")
    API_SECRET = os.environ.get("LASTFM_API_SECRET")
    API_USERNAME = os.environ.get("LASTFM_API_USERNAME")
    API_PASSWORD = os.environ.get("LASTFM_API_PASSWORD")
    API_PASSWORD_HASHED = pylast.md5(API_PASSWORD) if API_PASSWORD else None
    
    return pylast.LastFMNetwork(
        api_key=API_KEY,
        api_secret=API_SECRET,
        username=API_USERNAME,
        password_hash=API_PASSWORD_HASHED,
    )

def get_song_tags(network, artist, track, limit=20):
    """Get the tags and weights for a song"""
    try:
        track_obj = network.get_track(artist, track)
        track_tags = track_obj.get_top_tags(limit=limit)
        # Return a list of (name, weight) tuples
        return [(tag.item.get_name(), int(tag.weight)) for tag in track_tags]
    except pylast.WSError as e:
        print(f"Error retrieving tags for {artist} - {track}: {e}")
        return []