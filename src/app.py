from flask import Flask, request, jsonify, send_from_directory
import joblib
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# --- Initialisation de l'application Flask ---
app = Flask(__name__, static_folder='../static') 

# --- Chargement des ressources (modèle, vectorizer, classes) ---
MODEL_PATH = 'models/classification_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
CLASSES_PATH = 'models/classes.pkl'

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    with open(CLASSES_PATH, 'rb') as f:
        classes = pickle.load(f)
    print("Modèle, vectorizer et classes chargés avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement des ressources : {e}")
    model = None
    vectorizer = None
    classes = None

# Dictionnaire de renommage des catégories pour plus de clarté
category_display_names = {
    'alt.atheism': 'Atheism',
    'comp.graphics': 'Computer Graphics',
    'comp.os.ms-windows.misc': 'MS Windows OS (Misc)',
    'comp.sys.ibm.pc.hardware': 'IBM PC Hardware',
    'comp.sys.mac.hardware': 'Macintosh Hardware',
    'comp.windows.x': 'X Window System',
    'misc.forsale': 'Miscellaneous For Sale',
    'rec.autos': 'Automobiles',
    'rec.motorcycles': 'Motorcycles',
    'rec.sport.baseball': 'Baseball',
    'rec.sport.hockey': 'Hockey',
    'sci.crypt': 'Cryptography',
    'sci.electronics': 'Electronics',
    'sci.med': 'Medicine',
    'sci.space': 'Space',
    'soc.religion.christian': 'Christianity',
    'talk.politics.guns': 'Politics (Guns)',
    'talk.politics.mideast': 'Politics (Middle East)',
    'talk.politics.misc': 'Politics (Misc)',
    'talk.religion.misc': 'Religion (Misc)'
}

# --- Fonction de prétraitement (doit être la même que celle utilisée pour l'entraînement) ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return ""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

# --- Nouvelle route pour servir la page HTML ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# --- Point d'API pour la classification ---
@app.route('/classify', methods=['POST'])
def classify_document():
    if not model or not vectorizer or not classes:
        return jsonify({"error": "Model not loaded. API not ready."}), 500

    data = request.get_json(force=True)
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request."}), 400

    document = data['text']

    # 1. Preprocessing the document
    processed_document = preprocess_text(document)

    # Handle case where document becomes empty after preprocessing
    if not processed_document:
        return jsonify({"category": "unknown", "confidence": 0.0, "message": "Document empty after preprocessing."}), 200

    # 2. Vectorization of the document
    document_vector = vectorizer.transform([processed_document])

    # 3. Prediction of the category
    prediction_index = model.predict(document_vector)[0]
    original_category_name = classes[prediction_index]
    
    # Get the display name for the category, default to original if not found
    display_category_name = category_display_names.get(original_category_name, original_category_name)

    # 4. Calculate confidence
    confidence = 0.0
    if hasattr(model, 'predict_proba'):
        confidence = model.predict_proba(document_vector).max()
    else:
        confidence = 1.0

    return jsonify({
        "category": display_category_name, # Return the more explicit name
        "confidence": float(confidence)
    })

# --- Run the Flask application ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
