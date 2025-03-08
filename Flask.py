import os
import pickle
import pandas as pd
import re
import time
import jwt
import bcrypt
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS

# Flask Setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # Allow all origins

# User Data Storage (Mock Database)
users = {}


class RecipeIndexer:
    def __init__(self, file_path='resource/completed_recipes.csv', is_reset=False):
        self.stored_file = 'resource/recipes_pickle.pkl'
        self.file_path = file_path

        if not is_reset and os.path.isfile(self.stored_file):
            try:
                with open(self.stored_file, 'rb') as f:
                    cached_dict = pickle.load(f)
                self.__dict__.update(cached_dict)
            except (pickle.UnpicklingError, EOFError):
                print("Corrupted index file detected. Rebuilding index...")
                self.run_indexer()
        else:
            self.run_indexer()

    @staticmethod
    def preprocess_text(text):
        """Preprocess text by removing special characters and converting to lowercase."""
        text = text.lower()
        text = re.sub(r"^c[\"\s]*", "", text)  # Remove leading 'c', quotes, and spaces
        text = re.sub(r"[^\w\s.,/:-]", "", text)  # Keep valid characters
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def preprocess_img(text):
        if not isinstance(text, str) or not text.strip():
            return ""  # Return empty if the field is missing or not a string

        # Remove c(" at the start and ") at the end
        text = re.sub(r'^c\(["\s]*', '', text)  # Remove 'c("'
        text = re.sub(r'["\s]*\)$', '', text)  # Remove '")'

        # Remove any leading and trailing quotes
        text = re.sub(r'^"+|"+$', '', text)

        return text.strip()

    def run_indexer(self):
        """Reads the CSV, processes text data, and indexes it."""
        df = pd.read_csv(self.file_path)

        # Process relevant fields
        df['RecipeIngredientParts'] = df['RecipeIngredientParts'].astype(str).apply(self.preprocess_text)
        df['RecipeInstructions'] = df['RecipeInstructions'].astype(str).apply(self.preprocess_text)
        df['image_link'] = df['image_link'].astype(str).apply(self.preprocess_img)

        # Combine 'title' and 'text' for indexing
        df['text_data'] = df[['RecipeIngredientParts', 'RecipeInstructions']].fillna('').agg(' '.join, axis=1)
        corpus = df['text_data'].tolist()

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.recipes_df = df

        with open(self.stored_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

        print("TF-IDF index created successfully.")

    def search_query(self, query, top_n=50):
        query = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        self.recipes_df['score'] = similarity_scores
        results_df = self.recipes_df.nlargest(top_n, 'score').copy()

        return results_df[
            ['RecipeId', 'Name', 'Description','image_link' , 'RecipeIngredientParts', 'RecipeInstructions', 'score', 'TotalTime',
             'Calories']].to_dict('records')


# Initialize Indexer
indexer = RecipeIndexer()

@app.after_request
def add_cors_headers(response):
    """Ensure all responses contain correct CORS headers."""
    response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if username in users:
        return jsonify({'status': 'error', 'message': 'User already exists'}), 400

    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[username] = hashed_pw
    return jsonify({'status': 'success', 'message': 'User registered'})


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if username not in users or not bcrypt.checkpw(password.encode(), users[username].encode()):
        return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401

    token = jwt.encode({'username': username, 'exp': datetime.utcnow() + timedelta(days=1)}, app.config['SECRET_KEY'],
                       algorithm='HS256')
    return jsonify({'status': 'success', 'token': token})


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    try:
        results = indexer.search_query(query, top_n=50)
        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/search', methods=['OPTIONS'])
def handle_options():
    """Handle preflight CORS request."""
    response = jsonify({'status': 'success', 'message': 'Preflight OK'})
    response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response, 204


if __name__ == '__main__':
    app.run(debug=False)
