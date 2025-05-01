#!/usr/bin/env python

import os
import sys
import pandas as pd
import numpy as np
from joblib import load
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
import argparse

# Add the parent directory to path so we can import the music_sentiment package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from music_sentiment.lastfm_api import get_lastfm_network, get_song_tags
from music_sentiment.tag_utils import process_tag_weights

def load_models_and_data():
    """Load the trained model, vectorizer, and Warriner wordlist"""
    # Check if models exist
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'models', 'warriner_valence_classifier.joblib')
    vectorizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  'models', 'warriner_tfidf_vectorizer.joblib')
    warriner_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'data', 'cleaned', 'warriner_clean.csv')
    
    # Validate paths
    for path, name in [(model_path, "Model"), (vectorizer_path, "Vectorizer"), 
                       (warriner_path, "Warriner wordlist")]:
        if not os.path.exists(path):
            print(f"{name} not found at: {path}")
            return None, None, None
    
    # Load the files
    try:
        model = load(model_path)
        vectorizer = load(vectorizer_path)
        warriner_df = pd.read_csv(warriner_path)
        # Set word as index for faster lookups
        warriner_df.set_index('word', inplace=True)
        return model, vectorizer, warriner_df
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None, None

def extract_warriner_features(tag_text, warriner_df):
    """Extract emotional features from tags using the Warriner wordlist"""
    # Split the tag text into individual words
    words = set(tag_text.lower().split())
    
    # Initialize feature values
    valence_scores = []
    arousal_scores = []
    dominance_scores = []
    matched_words = []
    
    # Match words with the Warriner lexicon
    for word in words:
        if word in warriner_df.index:
            matched_words.append(word)
            valence_scores.append(warriner_df.loc[word, 'valence_score'])
            arousal_scores.append(warriner_df.loc[word, 'arousal_score'])
            dominance_scores.append(warriner_df.loc[word, 'dominance_score'])
    
    # Calculate feature dictionary
    features = {
        'warriner_valence_mean': np.mean(valence_scores) if valence_scores else 0.0,
        'warriner_arousal_mean': np.mean(arousal_scores) if arousal_scores else 0.0,
        'warriner_dominance_mean': np.mean(dominance_scores) if dominance_scores else 0.0,
        'warriner_coverage': len(matched_words) / len(words) if words else 0.0
    }
    
    return features

def predict_song_valence(artist, title, model, vectorizer, warriner_df, network):
    """Predict the valence bin of a song using Last.fm tags and Warriner features"""
    # Define valence categories
    valence_categories = {
        0: "Very Sad (0-2.5)",
        1: "Somewhat Sad (2.5-5)",
        2: "Somewhat Happy (5-7.5)",
        3: "Very Happy (7.5-10)"
    }
    
    print(f"\nFetching tags for {artist} - {title}...")
    # Get tags from Last.fm
    tags = get_song_tags(network, artist, title)
    if not tags:
        print("No tags found for this song.")
        return None
    
    # Process tags into text representation
    tag_text = process_tag_weights(tags)
    
    print("Extracting features for prediction...")
    # Extract TF-IDF features
    tfidf_features = vectorizer.transform([tag_text])
    
    # Extract Warriner emotional features
    warriner_feature_dict = extract_warriner_features(tag_text, warriner_df)
    
    # Convert warriner features to array for combining with TF-IDF
    warriner_features = np.array([
        warriner_feature_dict['warriner_valence_mean'],
        warriner_feature_dict['warriner_arousal_mean'],
        warriner_feature_dict['warriner_dominance_mean'],
        warriner_feature_dict['warriner_coverage']
    ]).reshape(1, -1)
    
    # Scale the warriner features
    scaler = StandardScaler()
    warriner_features_scaled = scaler.fit_transform(warriner_features)
    
    # Combine features (TF-IDF + Warriner)
    combined_features = hstack([tfidf_features, warriner_features_scaled])
    
    # Make prediction
    print("Predicting valence category...")
    try:
        prediction = model.predict(combined_features)[0]
        
        # Debug information
        print(f"Raw prediction: {prediction}, Type: {type(prediction)}")
        
        # Handle different prediction types
        if isinstance(prediction, (str, np.str_)):
            # If prediction is a string like 'happy' or 'sad'
            if prediction == 'happy' or prediction == np.str_('happy'):
                prediction_bin = 2  # Somewhat Happy
                prediction_label = 'happy'
            else:
                prediction_bin = 1  # Somewhat Sad
                prediction_label = 'sad'
        else:
            # If prediction is already numeric
            try:
                prediction_bin = int(prediction)
                prediction_label = "happy" if prediction_bin >= 2 else "sad"
            except:
                # Fallback if conversion fails
                if warriner_feature_dict['warriner_valence_mean'] > 5.0:
                    prediction_bin = 2
                    prediction_label = 'happy'
                else:
                    prediction_bin = 1
                    prediction_label = 'sad'
        
        # Get the categorical description
        category = valence_categories[prediction_bin]
        
        # Return results
        return {
            'prediction_bin': prediction_bin,
            'prediction_label': prediction_label,
            'category': category,
            'warriner_features': warriner_feature_dict,
            'matched_tags': tags
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(f"Prediction value: {prediction}, Type: {type(prediction)}")
        
        # Fallback prediction based on Warriner valence
        fallback_bin = 2 if warriner_feature_dict['warriner_valence_mean'] > 5.0 else 1
        fallback_label = 'happy' if fallback_bin == 2 else 'sad'
        fallback_category = valence_categories[fallback_bin]
        
        print(f"Using fallback prediction based on Warriner valence score")
        
        return {
            'prediction_bin': fallback_bin,
            'prediction_label': fallback_label,
            'category': fallback_category,
            'warriner_features': warriner_feature_dict,
            'matched_tags': tags
        }

def check_database_valence(artist, title, db_path=None):
    """Check if a song is in the database and return its valence value"""
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'data', 'cleaned', 'muse_cleaned.csv')
    
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        return None, None
    
    try:
        df = pd.read_csv(db_path)
        # Try to find the song in the database (case-insensitive search)
        matches = df[(df['artist'].str.lower() == artist.lower()) & 
                    (df['track'].str.lower() == title.lower())]
        
        if len(matches) > 0:
            # Return the first match's valence and emotion
            valence = matches.iloc[0]['valence_tags']
            emotion = matches.iloc[0]['emotion']
            return valence, emotion
        else:
            return None, None
    except Exception as e:
        print(f"Error checking database: {e}")
        return None, None

def search_song(query, network):
    """Search for a song using Last.fm's track.search"""
    track_search = network.search_for_track("", query)  # Empty string for artist to search globally
    results = track_search.get_next_page()
    
    if not results:
        print("No results found.")
        return None
    
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
            return selected_track
        else:
            print("Invalid selection.")
            return None
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None

def main():
    """Main function to run the song search and valence prediction"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict the valence of a song using Last.fm tags and Warriner wordlist')
    parser.add_argument('--song', type=str, help='Song name to search for')
    args = parser.parse_args()
    
    # Load models and data
    model, vectorizer, warriner_df = load_models_and_data()
    if model is None or vectorizer is None or warriner_df is None:
        print("Failed to load required files. Exiting.")
        sys.exit(1)
    
    # Get Last.fm network
    network = get_lastfm_network()
    
    # Get song query from command line or prompt
    song_query = args.song if args.song else input("Enter a song name: ")
    
    # Search for the song
    selected_track = search_song(song_query, network)
    if not selected_track:
        sys.exit(1)
    
    # Predict valence for the selected track
    result = predict_song_valence(
        selected_track.artist.name, 
        selected_track.title, 
        model, 
        vectorizer, 
        warriner_df, 
        network
    )
    
    if result:
        print(f"\nPrediction Results:")
        print(f"Predicted valence category: {result['category']}")
        print(f"Prediction label: {result['prediction_label']}")
        
        # Check if the song is in the database for comparison
        actual_valence, actual_emotion = check_database_valence(selected_track.artist.name, selected_track.title)
        
        if actual_valence is not None:
            print(f"\nActual values in database:")
            print(f"- Valence score: {actual_valence:.2f}")
            print(f"- Emotion: {actual_emotion}")
            
            # Determine actual category based on the same bins
            bins = [0, 2.5, 5, 7.5, 10]
            for i in range(len(bins) - 1):
                if bins[i] <= actual_valence < bins[i+1]:
                    actual_bin = i
                    break
            else:
                actual_bin = 3  # For the edge case of valence = 10
                
            valence_categories = {
                0: "Very Sad (0-2.5)",
                1: "Somewhat Sad (2.5-5)",
                2: "Somewhat Happy (5-7.5)",
                3: "Very Happy (7.5-10)"
            }
            print(f"Actual valence category: {valence_categories[actual_bin]}")
        else:
            print("\nSong not found in the reference database.")
        
        # Print Warriner emotional feature information
        print("\nWarriner Emotional Features:")
        print(f"- Valence score: {result['warriner_features']['warriner_valence_mean']:.2f}")
        print(f"- Arousal score: {result['warriner_features']['warriner_arousal_mean']:.2f}")
        print(f"- Dominance score: {result['warriner_features']['warriner_dominance_mean']:.2f}")
        print(f"- Word coverage: {result['warriner_features']['warriner_coverage'] * 100:.1f}%")
        
        # Print top tags for reference
        print("\nTop tags for this song:")
        for tag, weight in sorted(result['matched_tags'], key=lambda x: x[1], reverse=True)[:10]:
            print(f"- {tag} (weight: {weight})")

if __name__ == "__main__":
    main()