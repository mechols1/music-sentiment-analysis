import pandas as pd

from music_sentiment.lastfm_api import get_lastfm_network, get_song_tags

#for data processing and making table for tags and songs

# we will be using tags a lot, so let's create a function for
# formatting the tags into human readable format
def display_tags_table(track_info, limit=30):
    """Display a nicely formatted table of tags and weights"""
    # Create a list of (tag, weight) tuples
    tag_data = [(item.item.get_name(), int(item.weight)) for item in track_info]
    
    # Sort by weight in descending order
    tag_data.sort(key=lambda x: x[1], reverse=True)
    
    # Limit to specified number
    tag_data = tag_data[:limit]
    
    
    # Create a DataFrame for display
    df = pd.DataFrame(tag_data, columns=['Tag', 'Weight'])
    
    # Return the DataFrame (Jupyter will display it nicely)
    return df

def process_tag_weights(song_tags):
 

    """Process tags with weights for text analysis"""
 

    # create a weighted text representation where more important tags appear more frequently
 

    return " ".join([tag[0] for tag in song_tags for _ in range(min(int(tag[1]//10), 10))])

# def tags_dict_to_text(tags_dict):
#     if isinstance(tags_dict, dict):
#         return ' '.join(tags_dict.keys())
#     elif pd.isna(tags_dict):
#         return ''
#     else:
#         return str(tags_dict)
