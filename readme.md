# Music Sentiment Analysis

This project implements a binary sentiment classifier for music using Naive Bayes. The classifier categorizes songs as either "happy" or "sad" based on valence scores from the muSE dataset.

## About

The project uses the muSE (Music Sentiment Expression) dataset, which provides various audio features including valence tags. Valence in music represents the musical positiveness conveyed by a track. 

The implementation uses a threshold-based approach on valence scores to create binary labels:
- Valence > 5.0: Classified as "happy"
- Valence ≤ 5.0: Classified as "sad"

These binary labels are then used to train a Complement Naive Bayes classifier for sentiment prediction.

## Current Progress

- ✅ Data loading and initial exploration completed
- ✅ Implemented binary sentiment labeling based on valence threshold (5.0)
- ✅ Initial distribution analysis shows 57,799 "happy" songs vs 32,202 "sad" songs
- ✅ Successfully integrated Last.fm API (pylast) for retrieving song tags
- ✅ Created helper functions to format and display tag data
- ✅ Implemented basic Complement Naive Bayes classifier
- ❌ Current model accuracy is around 45% - needs improvement

## Current Approach

The current implementation:
1. Uses a simple valence threshold (5.0) to create binary sentiment labels
2. Retrieves tags from Last.fm API for a sample of songs
3. Processes tags as text features without considering tag weights
4. Implements a Complement Naive Bayes classifier, which is typically better suited for imbalanced datasets

## Areas for Improvement

The current model's performance (45% accuracy) indicates significant room for improvement:

1. **Incorporate Tag Weights**: Currently, tag weights from Last.fm aren't used in the feature representation. Including these weights could improve model performance.

2. **Increase Training Data Size**: The current implementation uses only a small sample of songs. Expanding to use more songs from the dataset could improve accuracy.

3. **Combine Features**: Integrating Last.fm tags with audio features from the dataset could provide a more comprehensive representation of each song.

4. **Refine Valence Threshold**: The current fixed threshold of 5.0 may not be optimal. Experimenting with different thresholds or using a more sophisticated approach could improve label quality.

5. **Feature Engineering**: Improving feature extraction and selection from Last.fm tags, potentially using TF-IDF or more advanced NLP techniques.

6. **Cross-Validation**: Implement proper cross-validation to ensure model robustness.

7. **Consider Genre-Specific Models**: Different music genres may have different indicators of sentiment, so genre-specific models could be more accurate.

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

## Next Steps

1. **Improve Feature Representation**:
   - Implement tag weight incorporation into features
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
6. **Organize files**:
   - Need a better seperation of concerns, possibly putting the machine learnign algorithm operation in another file.

## Technologies

### Current Technologies
- **Python 3.10+**: Core programming language
- **pandas**: For data manipulation and analysis
- **NumPy**: For numerical operations
- **Jupyter Notebook**: For interactive development and data exploration
- **pylast**: Python interface to the Last.fm API
- **python-dotenv**: For environment variable management
- **scikit-learn**: For implementing the Complement Naive Bayes classifier

### Planned Technologies
- **matplotlib/seaborn**: For data visualization and model performance analysis
- **NLTK/spaCy**: For potential text processing of tags and other text features
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

To run the classification model:
```
jupyter notebook notebooks/music_sentiment_classifier.ipynb
```

## Project Structure

```
music-sentiment-analysis/
├── data/
│   ├── raw/              # Raw data files
│   │   └── muse_v3.csv   # muSE dataset
│   └── processed/        # Processed data files
│       └── muse_cleaned.csv
│       └── muse_with_tags_valence.csv
├── notebooks/
│   ├── data-cleaning.ipynb      # Data preprocessing
│   ├── lastfm_music_tags.ipynb  # Last.fm API tag retrieval
├── venv/                # Virtual environment
├── .env                 # API credentials (not committed to repo)
├── .gitignore           # Git ignore file
└── README.md            # This file
```