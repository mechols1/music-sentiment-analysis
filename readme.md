# Music Sentiment Analysis

This project implements a music sentiment classifier using Naive Bayes and Warriner's lexicon. The classifier categorizes songs based on valence scores derived from Last.fm tags and the muSE dataset.

## About

The project uses the muSE (Music Sentiment Expression) dataset, which provides various audio features including valence tags. Valence in music represents the musical positiveness conveyed by a track. 

The implementation uses a binning approach on valence scores to create categorical labels:
- Valence 0-2.5: Very Sad (0)
- Valence 2.5-5: Somewhat Sad (1)
- Valence 5-7.5: Somewhat Happy (2)
- Valence 7.5-10: Very Happy (3)

These categorized labels train a Complement Naive Bayes classifier for sentiment prediction.

## Current Progress

- ✅ Data loading and initial exploration completed
- ✅ Implemented valence binning approach with 4 categories
- ✅ Initial distribution analysis shows 57,799 "happy" songs vs 32,202 "sad" songs
- ✅ Successfully integrated Last.fm API (pylast) for retrieving song tags
- ✅ Created helper functions to format and display tag data
- ✅ Implemented Complement Naive Bayes classifier
- ✅ Integrated Warriner lexicon for emotional feature extraction
- ✅ Model accuracy reached 66.22% for the 4-category classification
- ✅ Binary classification (happy/sad) reached 74.5% accuracy

## Current Approach

The current implementation:
1. Bins the valence scores into four categories for a more nuanced sentiment analysis
2. Retrieves tags from Last.fm API for a sample of songs
3. Processes tags as text features with TF-IDF vectorization
4. Combines tag features with emotional scores from Warriner's lexicon
5. Implements a Complement Naive Bayes classifier, which is better suited for imbalanced datasets

## Areas for Improvement

Despite the improved accuracy, there are still areas for enhancement:

1. **Further Refine Warriner Integration**: The current implementation successfully integrates Warriner's lexicon, but more sophisticated matching algorithms could improve word coverage.

2. **Increase Training Data Size**: The current implementation uses only a sample of songs. Expanding to use more songs from the dataset could improve accuracy.

3. **Cross-Validation**: Implement proper cross-validation to ensure model robustness.

4. **Consider Genre-Specific Models**: Different music genres may have different indicators of sentiment, so genre-specific models could be more accurate.

5. **Advanced ML Models**: Experiment with more sophisticated classifiers like SVM, Random Forest, or neural networks.

## Dataset Analysis

The muSE dataset contains 90,001 songs with the following features:
- Basic metadata: track name, artist, lastfm_url
- Seeds: Emotion tags used to collect the songs
- Emotion metrics:
  - valence_tags: Musical positiveness (ranges from ~0.2 to ~8.5)
  - arousal_tags: Energy/intensity (ranges from ~0.1 to ~7.3)
  - dominance_tags: Power/strength conveyed (ranges from ~0.2 to ~7.4)
- Additional information: mbid, spotify_id, genre

Data analysis reveals:
- No missing values in core features (track, artist, emotion metrics)
- Some missing values in optional identifiers (mbid: 28,784, spotify_id: 28,371)
- 6,639 songs missing genre information

## Warriner Lexicon Integration

The project incorporates the Warriner lexicon, which provides emotional scores for English words:
- Valence: Pleasantness (ranges from 1.26 to 8.53)
- Arousal: Intensity (ranges from 1.6 to 7.79)
- Dominance: Control (ranges from 1.68 to 7.9)

These scores are used to extract additional features from song tags, capturing emotional content that might be missed by simple TF-IDF vectorization.

## Next Steps

1. **Improve Feature Representation**:
   - Refine tag weight incorporation into features
   - Experiment with different text vectorization methods

2. **Expand Training Data**:
   - Process more songs from the dataset to increase training sample size

3. **Model Optimization**:
   - Try different Naive Bayes variants (Multinomial, Gaussian)
   - Experiment with other classifiers (SVM, Random Forest)

4. **Evaluation Framework**:
   - Implement proper cross-validation
   - Add more evaluation metrics (precision, recall, F1 score by class)

5. **Deploy Simple Demo**:
   - Create a simple web interface for testing the classifier

6. **Organize Files**:
   - Better separation of concerns, possibly putting the machine learning algorithm operations in another file

## Technologies

### Current Technologies
- **Python 3.11+**: Core programming language
- **pandas**: For data manipulation and analysis
- **NumPy**: For numerical operations
- **scikit-learn**: For implementing the Complement Naive Bayes classifier
- **Jupyter Notebook**: For interactive development and data exploration
- **pylast**: Python interface to the Last.fm API
- **python-dotenv**: For environment variable management
- **matplotlib/seaborn**: For data visualization
- **joblib**: For model persistence
- **scipy**: For sparse matrix operations
- **tqdm**: For progress tracking

### Planned Technologies
- **NLTK/spaCy**: For potential text processing of tags and other text features
- **librosa**: For audio feature extraction (if raw audio files are used)
- **Flask/FastAPI**: For potential API development to serve the model
- **pytest**: For unit testing

## Setup

### Prerequisites

- Python 3.11+
- pip
- virtualenv (recommended)
- Last.fm API credentials (for tag retrieval functionality)

### Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/music-sentiment-analysis.git
   cd music-sentiment-analysis
   ```

2. Create and activate a virtual environment
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages
   ```
   pip install -r requirements.txt
   ```

4. Set up Last.fm API access
   - Create a `.env` file in the project root with your Last.fm API credentials:
   ```
   LASTFM_API_KEY=your_api_key
   LASTFM_API_SECRET=your_api_secret
   LASTFM_API_USERNAME=your_username
   LASTFM_API_PASSWORD=your_password
   ```

5. Download the muSE dataset
   - The raw data file (`muse_v3.csv`) should be placed in the `data/raw/` directory
   - Note: You may need to request access to the dataset if you don't already have it

### Using the Song Predictor

To predict the sentiment of a song:
```
python scripts/song-search-w.py
```

This script will:
1. Ask you to enter a song name
2. Search for matches on Last.fm
3. Ask you to select the correct song
4. Retrieve tags and predict the song's sentiment
5. Show additional information like Warriner emotional scores

### Running the Jupyter Notebooks

To explore the data cleaning process:
```
jupyter notebook notebooks/data-cleaning.ipynb
```

To explore Last.fm tag retrieval:
```
jupyter notebook notebooks/lastfm_music_tags.ipynb
```

To explore the Warriner wordlist cleaning:
```
jupyter notebook notebooks/warriner_cleaning.ipynb
```

To run the classification model exploration:
```
jupyter notebook notebooks/exploration.ipynb
```

## Project Structure

```
music-sentiment-analysis/
├── data/
│   ├── raw/              # Raw data files
│   │   └── muse_v3.csv   # muSE dataset
│   ├── cleaned/          # Processed data files
│   │   ├── muse_cleaned.csv
│   │   ├── muse_with_tags_valence.csv
│   │   └── warriner_clean.csv
│   ├── cache/            # Cache for API calls
│   └── wordlists/        # Lexicons
│       └── warriner.csv  # Warriner emotional lexicon
├── models/               # Saved models
│   ├── valence_classifier.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── warriner_valence_classifier.joblib
│   └── warriner_tfidf_vectorizer.joblib
├── music_sentiment/      # Main package
│   ├── __init__.py
│   ├── lastfm_api.py     # Last.fm API functions
│   └── tag_utils.py      # Tag processing utilities
├── notebooks/            # Jupyter notebooks
│   ├── data-cleaning.ipynb
│   ├── lastfm_music_tags.ipynb
│   ├── warriner_cleaning.ipynb
│   └── exploration.ipynb
├── scripts/              # Command-line tools
│   ├── song-search.py    # Basic song sentiment predictor
│   └── song-search-w.py  # Advanced predictor with Warriner
├── venv/                 # Virtual environment
├── .env                  # API credentials (not committed to repo)
├── .gitignore            # Git ignore file
├── requirements.txt      # Project dependencies
└── README.md             # This file
