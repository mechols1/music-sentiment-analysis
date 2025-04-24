
import os
import sys
import pylast
import pandas as pd
from joblib import load

sys.path.append(os.path.abspath('..'))
from music_sentiment.lastfm_api import get_lastfm_network, get_song_tags
from music_sentiment.tag_utils import process_tag_weights

network = get_lastfm_network()

# load the saved model
model_path = '../models/valence_classifier.joblib'
vectorizer_path = '../models/tfidf_vectorizer.joblib'
print(f"Model path: {model_path}, Exists: {os.path.exists(model_path)}")
print(f"Vectorizer path: {vectorizer_path}, Exists: {os.path.exists(vectorizer_path)}")
# test if model and vectorizer exist
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    print("Model or vectorizer file does not exist.")
    print(f"Current working directory: {os.getcwd()}")
    sys.exit()

# load the model and vectorizer
model = load(model_path)
vectorizer = load(vectorizer_path)

valence_categories = {
    0: "Very Sad (0-3)",
    1: "Somewhat Sad (3-5)",
    2: "Somewhat Happy (5-7)",
    3: "Very Happy (7-8.5)"
}

# function to predict valence
def predict_song_valence(artist, title):
    # get tags 
    print(f"Fetching tags for {artist} - {title}...")
    tags = get_song_tags(network, artist, title)
    if not tags:
        print("No tags found.")
        return None
    
    # process tags
    print("Processing tags...")
    tag_text = process_tag_weights(tags)
    
    # transform tags using teh vectorizer
    print("Transforming tags...")
    features = vectorizer.transform([tag_text])
    
    # make prediction
    print("Making prediction...")
    prediction = model.predict(features)[0]
    
    # convert to a numpy
    prediction = int(prediction)
    
    return prediction

def check_database_valence(artist, title, df):
    """Check if a song is in the database and return its valence value"""
    # Try to find the song in the database (case-insensitive search)
    matches = df[(df['artist'].str.lower() == artist.lower()) & 
                (df['track'].str.lower() == title.lower())]
    
    if len(matches) > 0:
        # Return the first match's valence
        valence = matches.iloc[0]['valence_tags']
        emotion = matches.iloc[0]['emotion']
        return valence, emotion
    else:
        return None, None

user_input = input("Enter a song name: ")

# Search for the song using Last.fm's track.search
track_search = network.search_for_track("", user_input)  # Empty string for artist to search globally
results = track_search.get_next_page()

if not results:
    print("No results found.")
    sys.exit()

# Display the top 5 results
print("\nTop 5 search results:")
for i, track in enumerate(results[:5], start=1):
    print(f"{i}. {track.artist.name} - {track.title}")

# Prompt the user to select a song
try:
    selection = int(input("\nSelect a song by entering the corresponding number (1-5): "))
    if 1 <= selection <= len(results[:5]):
        selected_track = results[selection - 1]
        print(f"\nYou selected: {selected_track.artist.name} - {selected_track.title}")
        
        # predict valence for the selected track
        valence_bin = predict_song_valence(selected_track.artist.name, selected_track.title) # prediction will be a valence bin
        
        if valence_bin is not None:
            # map the prediction to the valence category
            print(f"Predicted valence category: {valence_categories[valence_bin]}")
            

            # check if the song is in the database
            df = pd.read_csv('../data/cleaned/muse_cleaned.csv')
            db_valence, db_emotion = check_database_valence(selected_track.artist.name, selected_track.title, df)
            
            if db_valence is not None:
                print(f"Actual values in database:")
                print(f"- Valence score: {db_valence:.2f}")
                print(f"- Emotion: {db_emotion}")
                print(f"Valence catagory: {valence_categories[int(db_valence > 5)]}")
            else:
                print("Song not found in the database.")
            # only get the most important tags
            song_tags = get_song_tags(network, selected_track.artist.name, selected_track.title)
            for tag, weight, in sorted(song_tags, key=lambda x: x[1], reverse=True)[:10]:
                print(f"- {tag}, (weight: {weight})")
        
    else:
        print("Invalid selection. Exiting.")
except ValueError:
    print("Invalid input. Please enter a number.")