# music_sentiment/__init__.py
from .lastfm_api import get_lastfm_network, get_song_tags
from .tag_utils import display_tags_table, process_tag_weights

__version__ = '0.1.0'