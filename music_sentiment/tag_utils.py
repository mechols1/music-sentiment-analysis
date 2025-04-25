import pandas as pd

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
    """Process tags to create a dictionary with tag names and weights"""
    # Create a dictionary where keys are tag names and values are weights
    tag_dict = {tag[0]: tag[1] for tag in song_tags}
    # Return the dictionary
    return tag_dict