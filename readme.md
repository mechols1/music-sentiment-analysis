# Music Sentiment Analysis

This project implements a binary sentiment classifier for music using Naive Bayes. The classifier categorizes songs as either "happy" or "sad" based on valence scores from the muSE dataset.

## About

The project uses the muSE (Music Sentiment Expression) dataset, which provides various audio features including valence tags. Valence in music represents the musical positiveness conveyed by a track. 

Currently, the implementation uses a simple threshold-based approach on valence scores to create binary labels:
- Valence > 5.0: Classified as "happy"
- Valence ≤ 5.0: Classified as "sad"

These binary labels are then used to train a Naive Bayes classifier for sentiment prediction.

## Current Progress

- ✅ Data loading and initial exploration completed
- ✅ Implemented binary sentiment labeling based on valence threshold (5.0)
- ✅ Initial distribution analysis shows 57,799 "happy" songs vs 32,202 "sad" songs
- ✅ Successfully integrated Last.fm API (pylast) for retrieving song tags
- ✅ Created helper functions to format and display tag data
- 🔄 Working on feature extraction from Last.fm tags for classifier input

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

## Technologies

### Current Technologies
- **Python 3.10+**: Core programming language
- **pandas**: For data manipulation and analysis
- **NumPy**: For numerical operations
- **Jupyter Notebook**: For interactive development and data exploration
- **pylast**: Python interface to the Last.fm API
- **python-dotenv**: For environment variable management

### Planned Technologies
- **scikit-learn**: For implementing the Naive Bayes classifier and other ML algorithms
- **matplotlib/seaborn**: For data visualization and model performance analysis
- **NLTK/spaCy**: For potential text processing if lyrics are incorporated
- **librosa**: For audio feature extraction (if raw audio files are used)
- **Flask/FastAPI**: For potential API development to serve the model
- **pytest**: For unit testing

## Setup

### Prerequisites

- Python 3.10+
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

### Running the Jupyter Notebooks

To explore the data cleaning process:
```
jupyter notebook notebooks/data-cleaning.ipynb
```

To explore Last.fm tag retrieval:
```
jupyter notebook notebooks/lastfm_music_tags.ipynb
```

## Project Structure

```
music-sentiment-analysis/
├── data/
│   ├── raw/             # Raw data files
│   │   └── muse_v3.csv  # muSE dataset
│   └── processed/       # Processed data files
├── notebooks/
│   ├── data-cleaning.ipynb      # Data preprocessing
│   └── lastfm_music_tags.ipynb  # Last.fm API tag retrieval
├── venv/                # Virtual environment
├── .env                 # API credentials (not committed to repo)
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## Next Steps

- Complete Naive Bayes classifier implementation using tag features
- Create a feature matrix combining valence scores and tag information
- Implement cross-validation for model evaluation
- Experiment with different classification algorithms (SVM, Random Forest)
- Incorporate additional features beyond valence (arousal, dominance)
- Consider genre-specific thresholds for more accurate labeling
- Expand beyond binary classification to multi-class sentiment analysis
- Evaluate model performance using various metrics (accuracy, F1-score, ROC)
- Potentially incorporate audio features if available
- Create a simple web interface for demonstration purposes