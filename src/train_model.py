# %% Imports
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # Ajout de CountVectorizer et TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV # Ajout de GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier # Ajout de RandomForestClassifier
import tensorflow as tf # Ajout de tensorflow
from tensorflow.keras.models import Sequential # Ajout de Sequential
from tensorflow.keras.layers import Dense, Dropout # Ajout de Dense, Dropout
from tensorflow.keras.optimizers import Adam # Ajout de Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import pickle

# %% Phase 1 : Préparation et Collecte des Données
# Télécharger le dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Afficher quelques informations de base
print(f"Nombre total de documents : {len(newsgroups.data)}")
print(f"Nombre de catégories : {len(newsgroups.target_names)}")
print(f"Noms des catégories : {newsgroups.target_names}")

# %% Phase 2 : Exploration et Analyse des Données - Distribution des catégories
# Compter la distribution des documents par catégorie
category_counts = Counter(newsgroups.target)
category_names = [newsgroups.target_names[i] for i in category_counts.keys()]
category_values = list(category_counts.values())

# Créer un DataFrame pour faciliter la visualisation avec seaborn
df_distribution = pd.DataFrame({
    'Category': category_names,
    'Count': category_values
})
df_distribution = df_distribution.sort_values(by='Count', ascending=False)

# Visualiser la distribution
plt.figure(figsize=(12, 8))
sns.barplot(x='Count', y='Category', data=df_distribution, palette='viridis')
plt.title('Distribution des documents par catégorie')
plt.xlabel('Nombre de documents')
plt.ylabel('Catégorie')
plt.tight_layout()
plt.show()

print("\nDistribution des documents par catégorie:")
for category, count in df_distribution.values:
    print(f"- {category}: {count} documents")

# %% Phase 2 : Exploration et Analyse des Données - Longueur des documents
# Calcul de la longueur des documents
document_lengths = [len(doc.split()) for doc in newsgroups.data]

# Statistiques descriptives sur la longueur des documents
print(f"\nLongueur moyenne des documents : {np.mean(document_lengths):.2f} mots")
print(f"Longueur médiane des documents : {np.median(document_lengths):.2f} mots")
print(f"Longueur minimale des documents : {np.min(document_lengths)} mots")
print(f"Longueur maximale des documents : {np.max(document_lengths)} mots")

# Visualisation de la distribution des longueurs
plt.figure(figsize=(10, 6))
sns.histplot(document_lengths, bins=50, kde=True, color='skyblue')
plt.title('Distribution de la longueur des documents')
plt.xlabel('Nombre de mots')
plt.ylabel('Fréquence')
plt.show()

# %% Phase 3 : Prétraitement des Données - Fonction de prétraitement
def preprocess_text(text):
    # 1. Conversion en minuscules
    text = text.lower()

    # 2. Suppression des caractères spéciaux (conserve seulement lettres et espaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()

    # Gérer les documents vides après nettoyage
    if not text:
        return ""

    # 4. Tokenisation
    tokens = word_tokenize(text)

    # 5. Suppression des mots vides (stopwords) - Utilisation de l'anglais
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # 6. Racinisation (stemming)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)

# %% Phase 3 : Prétraitement des Données - Application du prétraitement
# --- Test de la fonction de prétraitement ---
sample_doc_index = 0
original_text = newsgroups.data[sample_doc_index]
preprocessed_text = preprocess_text(original_text)

print(f"\n--- Exemple de Prétraitement ---")
print(f"Texte original (extrait) : {original_text[:500]}...")
print(f"Texte prétraité (extrait) : {preprocessed_text[:500]}...")

# Appliquer le prétraitement à tous les documents
print("\nApplication du prétraitement à tous les documents (cela peut prendre un certain temps)...")
preprocessed_documents = [preprocess_text(doc) for doc in newsgroups.data]
print("Prétraitement terminé.")

# Vérification des documents vides après prétraitement
empty_docs_count = preprocessed_documents.count("")
print(f"Nombre de documents vides après prétraitement : {empty_docs_count}")

# %% Phase 2 : Exploration et Analyse des Données - Mots les plus fréquents par catégorie (après prétraitement)
# Initialiser un dictionnaire pour stocker les compteurs de mots par catégorie
category_word_counts = {category: Counter() for category in newsgroups.target_names}

# Parcourir les documents PRÉTRAITÉS et compter les mots par catégorie
for i, doc_text in enumerate(preprocessed_documents):
    category_index = newsgroups.target[i]
    category_name = newsgroups.target_names[category_index]
    tokens = doc_text.split()
    category_word_counts[category_name].update(tokens)

print("\nMots les plus fréquents par catégorie (Top 10) après prétraitement:")
for category, word_counts in category_word_counts.items():
    print(f"\n--- Catégorie : {category} ---")
    for word, count in word_counts.most_common(10):
        print(f"  {word}: {count}")

# %% Phase 4 : Extraction de Caractéristiques (Bag of Words)
print("\n--- Phase 4 : Extraction de Caractéristiques (Bag of Words) ---")

# Initialiser le CountVectorizer
vectorizer_bow = CountVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)

# Appliquer le vectorizer sur les documents prétraités
X_bow = vectorizer_bow.fit_transform(preprocessed_documents)

print(f"Dimensions de la matrice BoW : {X_bow.shape}")
print(f"Nombre de caractéristiques (mots/bigrammes) : {len(vectorizer_bow.get_feature_names_out())}")

print("\nQuelques caractéristiques apprises par le vectorizer BoW :")
print(vectorizer_bow.get_feature_names_out()[:20])

# %% Phase 4 : Extraction de Caractéristiques (TF-IDF)
print("\n--- Phase 4 : Extraction de Caractéristiques (TF-IDF) ---")

# Initialiser le TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    sublinear_tf=True
)

# Appliquer le vectorizer sur les documents prétraités
X_tfidf = tfidf_vectorizer.fit_transform(preprocessed_documents)

print(f"Dimensions de la matrice TF-IDF : {X_tfidf.shape}")
print(f"Nombre de caractéristiques (mots/bigrammes) : {len(tfidf_vectorizer.get_feature_names_out())}")

print("\nQuelques caractéristiques apprises par le vectorizer TF-IDF :")
print(tfidf_vectorizer.get_feature_names_out()[:20])

# %% Phase 5 : Division des Données
print("\n--- Phase 5 : Division des Données ---")

y = newsgroups.target

X_train_tfidf, X_temp_tfidf, y_train, y_temp = train_test_split(
    X_tfidf, y, test_size=0.4, random_state=42, stratify=y
)

X_train_bow, X_temp_bow, _, _ = train_test_split(
    X_bow, y, test_size=0.4, random_state=42, stratify=y
)

X_val_tfidf, X_test_tfidf, y_val, y_test = train_test_split(
    X_temp_tfidf, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

X_val_bow, X_test_bow, _, _ = train_test_split(
    X_temp_bow, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Taille de l'ensemble d'entraînement (X_train_tfidf) : {X_train_tfidf.shape}")
print(f"Taille de l'ensemble de validation (X_val_tfidf) : {X_val_tfidf.shape}")
print(f"Taille de l'ensemble de test (X_test_tfidf) : {X_test_tfidf.shape}")

print(f"\nNombre de documents dans l'ensemble d'entraînement : {len(y_train)}")
print(f"Nombre de documents dans l'ensemble de validation : {len(y_val)}")
print(f"Nombre de documents dans l'ensemble de test : {len(y_test)}")

classes = newsgroups.target_names

# %% Phase 6.1 : Entraînement des Modèles Classiques (Naive Bayes)
print("\n--- Phase 6.1 : Entraînement des Modèles Classiques (Naive Bayes) ---")

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_val_tfidf)

print("\nNaive Bayes - Rapport de classification sur l'ensemble de validation:")
print(classification_report(y_val, y_pred_nb, target_names=classes))

# %% Phase 6.1 : Entraînement des Modèles Classiques (SVM)
print("\n--- Phase 6.1 : Entraînement des Modèles Classiques (SVM) ---")

svm_model = SVC(kernel='linear', random_state=42)
print("Entraînement du modèle SVM (cela peut prendre quelques minutes)...")
svm_model.fit(X_train_tfidf, y_train)
print("Entraînement SVM terminé.")
y_pred_svm = svm_model.predict(X_val_tfidf)

print("\nSVM - Rapport de classification sur l'ensemble de validation:")
print(classification_report(y_val, y_pred_svm, target_names=classes))

# %% Phase 6.1 : Entraînement des Modèles Classiques (Random Forest)
print("\n--- Phase 6.1 : Entraînement des Modèles Classiques (Random Forest) ---")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Entraînement du modèle Random Forest (cela peut prendre quelques minutes)...")
rf_model.fit(X_train_tfidf, y_train)
print("Entraînement Random Forest terminé.")
y_pred_rf = rf_model.predict(X_val_tfidf)

print("\nRandom Forest - Rapport de classification sur l'ensemble de validation:")
print(classification_report(y_val, y_pred_rf, target_names=classes))

# %% Phase 6.2 : Entraînement des Modèles de Deep Learning (Réseau de neurones simple)
print("\n--- Phase 6.2 : Entraînement des Modèles de Deep Learning (Réseau de neurones simple) ---")

num_classes = len(classes)

model_nn = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model_nn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_nn.summary()

print("\nEntraînement du réseau de neurones (cela peut prendre un certain temps)...")
history_nn = model_nn.fit(
    X_train_tfidf, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val_tfidf, y_val),
    verbose=1
)
print("Entraînement du réseau de neurones terminé.")

loss_nn, accuracy_nn = model_nn.evaluate(X_val_tfidf, y_val, verbose=0)
print(f"\nAccuracy du réseau de neurones sur l'ensemble de validation : {accuracy_nn:.4f}")

y_pred_nn_probs = model_nn.predict(X_val_tfidf)
y_pred_nn = np.argmax(y_pred_nn_probs, axis=1)

print("\nRéseau de neurones simple - Rapport de classification sur l'ensemble de validation:")
print(classification_report(y_val, y_pred_nn, target_names=classes))

# %% Phase 7.1 : Évaluation des Modèles sur l'ensemble de test
print("\n--- Phase 7.1 : Évaluation des Modèles sur l'ensemble de test ---")

def evaluate_model(model, X_test, y_test, model_name, classes):
    y_pred = model.predict(X_test)
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred, target_names=classes))
    return accuracy_score(y_test, y_pred)

def evaluate_nn_model(model, X_test, y_test, model_name, classes):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred, target_names=classes))
    return accuracy_score(y_test, y_pred)

accuracy_nb_test = evaluate_model(nb_model, X_test_tfidf, y_test, "Naive Bayes (Test)", classes)
accuracy_svm_test = evaluate_model(svm_model, X_test_tfidf, y_test, "SVM (Test)", classes)
accuracy_rf_test = evaluate_model(rf_model, X_test_tfidf, y_test, "Random Forest (Test)", classes)
accuracy_nn_test = evaluate_nn_model(model_nn, X_test_tfidf, y_test, "Réseau de Neurones (Test)", classes)

print("\n--- Comparaison des Accuracies sur l'ensemble de test ---")
print(f"Naive Bayes : {accuracy_nb_test:.4f}")
print(f"SVM         : {accuracy_svm_test:.4f}")
print(f"Random Forest: {accuracy_rf_test:.4f}")
print(f"Réseau de Neurones: {accuracy_nn_test:.4f}")

# %% Phase 8.1 : Optimisation des Hyperparamètres (SVM avec GridSearchCV)
print("\n--- Phase 8.1 : Optimisation des Hyperparamètres (SVM avec GridSearchCV) ---")

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
}

svm_base_model = SVC(random_state=42)

grid_search_svm = GridSearchCV(
    svm_base_model,
    param_grid_svm,
    cv=5,
    scoring='f1_macro',
    verbose=3,
    n_jobs=-1
)

print("\nLancement de GridSearchCV pour SVM (cela peut prendre beaucoup de temps)...")
grid_search_svm.fit(X_train_tfidf, y_train)
print("GridSearchCV terminé.")

print(f"\nMeilleurs paramètres pour SVM : {grid_search_svm.best_params_}")

best_svm_model = grid_search_svm.best_estimator_

y_pred_best_svm_val = best_svm_model.predict(X_val_tfidf)
print("\nMeilleur SVM - Rapport de classification sur l'ensemble de validation:")
print(classification_report(y_val, y_pred_best_svm_val, target_names=classes))

y_pred_best_svm_test = best_svm_model.predict(X_test_tfidf)
print("\nMeilleur SVM - Rapport de classification sur l'ensemble de test:")
print(classification_report(y_test, y_pred_best_svm_test, target_names=classes))

accuracy_best_svm_test = accuracy_score(y_test, y_pred_best_svm_test)
print(f"\nAccuracy du meilleur SVM sur l'ensemble de test : {accuracy_best_svm_test:.4f}")

print("\n--- Comparaison des Accuracies sur l'ensemble de test (avec meilleur SVM) ---")
print(f"Naive Bayes : {accuracy_nb_test:.4f}")
print(f"SVM (initial): {accuracy_svm_test:.4f}")
print(f"SVM (optimisé): {accuracy_best_svm_test:.4f}")
print(f"Random Forest: {accuracy_rf_test:.4f}")
print(f"Réseau de Neurones: {accuracy_nn_test:.4f}")

# %% Phase 9.1 : Sauvegarde du modèle et du vectorizer
print("\n--- Phase 9.1 : Sauvegarde du modèle et du vectorizer ---")

model_filename = 'models/classification_model.pkl' # Chemin mis à jour
joblib.dump(best_svm_model, model_filename)
print(f"Modèle sauvegardé sous : {model_filename}")

vectorizer_filename = 'models/tfidf_vectorizer.pkl' # Chemin mis à jour
joblib.dump(tfidf_vectorizer, vectorizer_filename)
print(f"Vectorizer sauvegardé sous : {vectorizer_filename}")

classes_filename = 'models/classes.pkl' # Chemin mis à jour
with open(classes_filename, 'wb') as f:
    pickle.dump(classes, f)
print(f"Noms des classes sauvegardés sous : {classes_filename}")

print("\nSauvegarde terminée. Les fichiers .pkl sont prêts pour le déploiement.")

# %% Phase 10 : Monitoring et Maintenance (Conceptuel)

# Cette fonction simule un ré-entraînement avec de nouvelles données
# En production, 'new_data_path' pointerait vers un nouveau dataset
# et 'existing_data' serait chargé depuis une base de données ou un stockage.
def retrain_model_concept(new_data_sample_size=1000):
    print("\n--- Début du processus de ré-entraînement conceptuel ---")

    # 1. Simuler la collecte de nouvelles données (ici, on prend un échantillon aléatoire)
    # En production, ce seraient de VRAIES nouvelles données labellisées.
    print(f"Simulation de la collecte de {new_data_sample_size} nouvelles données...")
    
    # S'assurer que newsgroups.data et newsgroups.target sont disponibles
    if 'newsgroups' not in globals():
        print("Chargement du dataset newsgroups pour la simulation de nouvelles données...")
        global newsgroups
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    # Sélectionner un échantillon aléatoire pour simuler de nouvelles données
    sample_indices = np.random.choice(len(newsgroups.data), new_data_sample_size, replace=False)
    new_docs = [newsgroups.data[i] for i in sample_indices]
    new_labels = [newsgroups.target[i] for i in sample_indices]

    # 2. Combiner avec les données existantes (pour un ré-entraînement complet)
    # En production, tu chargerais toutes les données labellisées historiques.
    print("Combinaison des données existantes et nouvelles...")
    all_docs_raw = list(newsgroups.data) # Utilise toutes les données originales comme "existantes"
    all_labels = list(newsgroups.target)

    # Pour cet exemple, nous allons simplement ré-entraîner sur l'ensemble complet
    # des données originales pour simuler un ré-entraînement complet.
    # Dans un vrai scénario, tu combinerais les anciennes données avec les nouvelles.
    
    # 3. Prétraitement de toutes les données
    print("Prétraitement de toutes les données pour le ré-entraînement...")
    preprocessed_all_docs = [preprocess_text(doc) for doc in all_docs_raw]

    # 4. Ré-entraînement du vectorizer (sur toutes les données)
    print("Ré-entraînement du vectorizer...")
    # Recréer le vectorizer pour s'assurer qu'il apprend de tout le vocabulaire
    global tfidf_vectorizer # Utilise le vectorizer global
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        sublinear_tf=True
    )
    X_all_tfidf = tfidf_vectorizer.fit_transform(preprocessed_all_docs)

    # 5. Ré-entraînement du modèle (sur toutes les données)
    print("Ré-entraînement du modèle SVM optimisé...")
    # Utilise les meilleurs paramètres trouvés précédemment
    retrained_svm_model = SVC(C=10, kernel='rbf', random_state=42, probability=True) # Ajout de probability=True pour predict_proba
    retrained_svm_model.fit(X_all_tfidf, all_labels)

    # 6. Évaluation du modèle ré-entraîné (sur l'ensemble de test original pour comparaison)
    print("\nÉvaluation du modèle ré-entraîné sur l'ensemble de test original:")
    y_pred_retrained = retrained_svm_model.predict(X_test_tfidf)
    accuracy_retrained = accuracy_score(y_test, y_pred_retrained)
    print(f"Accuracy du modèle ré-entraîné sur l'ensemble de test : {accuracy_retrained:.4f}")
    print(classification_report(y_test, y_pred_retrained, target_names=classes))

    # 7. Sauvegarde du nouveau modèle et vectorizer
    print("\nSauvegarde du modèle et vectorizer ré-entraînés...")
    joblib.dump(retrained_svm_model, 'models/classification_model.pkl')
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    print("Modèle et vectorizer ré-entraînés sauvegardés.")

    print("--- Processus de ré-entraînement conceptuel terminé ---")

# Pour exécuter cette fonction, tu peux décommenter la ligne ci-dessous
# retrain_model_concept(new_data_sample_size=500) # Exécute avec un petit échantillon

# %%
