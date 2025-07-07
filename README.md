# Document Classification API

This project implements a RESTful API for the automatic classification of text documents into 20 different categories, based on the "20 Newsgroups" dataset. The API is developed with Flask and utilizes traditional Machine Learning models (SVM) for classification. A simple web interface is also provided for easy testing.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Model Training](#model-training)
  - [API Launch](#api-launch)
  - [API Testing](#api-testing)
- [Model Performance](#model-performance)
- [Future Improvements](#future-improvements)
- [License](#license)

## Introduction

The goal of this project is to build a document classification system capable of assigning a text document to one of 20 predefined categories from the "20 Newsgroups" dataset. The process includes text preprocessing, feature extraction (TF-IDF), training a classification model (optimized SVM), and deploying this model via a Flask API.

## Features

-   **Text Preprocessing**: Cleaning, tokenization, stop-word removal, and stemming.
-   **Feature Extraction**: Uses the TF-IDF (Term Frequency-Inverse Document Frequency) approach to convert text into numerical vectors.
-   **Modeling**: Training and evaluation of several classification models (Naive Bayes, SVM, Random Forest, simple Neural Network).
-   **Hyperparameter Optimization**: Uses `GridSearchCV` to find the best hyperparameters for the SVM model.
-   **RESTful API**: A Flask API (`/classify`) to receive text and return the predicted category with a confidence score.
-   **Simple Web Interface**: An `index.html` page for easy interaction with the API.
-   **Model Saving and Loading**: Uses `joblib` and `pickle` to persist the trained model and vectorizer.
-   **Dependency Management**: `requirements.txt` file for easy installation of necessary libraries.

## Project Structure

document_classifier_api/
├── models/
│   ├── classification_model.pkl    # Trained SVM model
│   ├── tfidf_vectorizer.pkl        # Trained TF-IDF vectorizer
│   └── classes.pkl                 # Class (category) names
├── src/
│   ├── app.py                      # Flask API application
│   └── train_model.py              # Model training and evaluation script
├── static/
│   └── index.html                  # Simple web interface for API testing
└── requirements.txt                # List of Python dependencies

## Installation

To install and run this project locally, follow the steps below:

1.  **Clone this repository** (once you've pushed it to GitHub):
    ```bash
    git clone https://github.com/Mariechanne/document-classification-api.git
    cd document-classification-api
    ```
    *(Note: Currently, you are already working in the local folder, so this step is for a future user or for yourself if you re-clone the project. )*

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download necessary NLTK resources**:
    Open an interactive Python session in your terminal and run:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab') # Required for word_tokenize
    exit()
    ```

## Usage

### Model Training

The `train_model.py` script contains all the steps for data exploration, preprocessing, model training, and evaluation. It also saves the trained model and vectorizer.

To run the training script (this may take some time):
```bash
python src/train_model.py
```

Note: This script is designed to be run once to generate the .pkl files required by the API. If you already have the .pkl files in the models/ folder, you do not need to re-run it.

### API Launch

To start the Flask application:
```bash
python src/app.py
```
The API will be accessible at http://127.0.0.1:5000/.

### API Testing

Once the API is running:

- Via the Web Interface (recommended ):
Open your browser and go to http://127.0.0.1:5000/. Enter text in the provided area and click "Classify".

- Via curl (command line ):
Open a new terminal (leave the API running in the other) and execute:
```bash
# For Windows PowerShell
Invoke-RestMethod -Uri http://127.0.0.1:5000/classify -Method Post -Headers @{"Content-Type"="application/json"} -Body '{"text": "This is a document about computer hardware and software."}'

# For Linux/macOS or Git Bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "This is a document about computer hardware and software."}' http://127.0.0.1:5000/classify
```

## Model Performance

The optimized SVM model showed the following performance on the test set:
- Accuracy: ~0.6971
- F1-score (macro avg ): ~0.69
These metrics indicate a good classification capability for a 20-class problem with textual data.

## Future Improvements

- Advanced Optimization: Explore other hyperparameter optimization techniques or ensemble models.
- Deep Learning Models: Integrate and evaluate more advanced Deep Learning models like Transformers (BERT, CamemBERT) to potentially improve accuracy.
- Class Imbalance Handling: Apply techniques like SMOTE to improve performance on under-represented categories.
- Production Monitoring: Implement a monitoring system to track model performance in real-time and detect data drift.
- Multilingual Support: Extend the classifier to support other languages (requires new datasets and language-specific preprocessing tools).

## License

This project is licensed under the  MIT License. See the LICENSE file for more details.
