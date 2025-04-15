# Music Sentiment Analysis

This project implements a binary sentiment classifier for music using Naive Bayes. The classifier categorizes songs as either "happy" or "sad" based on valence scores from the muSE dataset.

## About

The project uses the muSE (Music Sentiment Expression) dataset, which provides various audio features including valence tags. Valence in music represents the musical positiveness conveyed by a track. 

Currently, the implementation uses a simple threshold-based approach on valence scores to create binary labels:
- Valence > 5.0: Classified as "happy"
- Valence ≤ 5.0: Classified as "sad"

These binary labels are then used to train a Naive Bayes classifier for sentiment prediction.

## Setup

### Prerequisites

- Python 3.7+
- pip
- virtualenv (recommended)

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

4. Download the muSE dataset
   - The raw data file (`muse_v3.csv`) should be placed in the `data/raw/` directory
   - Note: You may need to request access to the dataset if you don't already have it

### Running the Jupyter Notebooks

To explore the data cleaning process:
```
jupyter notebook notebooks/data-cleaning.ipynb
```

## Project Structure

```
music-sentiment-analysis/
├── data/
│   ├── raw/             # Raw data files
│   │   └── muse_v3.csv  # muSE dataset
│   └── processed/       # Processed data files
├── notebooks/
│   └── data-cleaning.ipynb  # Data preprocessing notebook
├── venv/                # Virtual environment
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## Future Work

- Implement more sophisticated feature extraction
- Experiment with different classification algorithms
- Expand beyond binary classification to multi-class sentiment analysis
- Train naive-bayes model
- Evaluate model performance using various metrics