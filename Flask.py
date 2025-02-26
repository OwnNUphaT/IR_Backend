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
CORS(app)
app.config['SECRET_KEY'] = 'your_secret_key'

# User Data Storage (Mock Database)
users = {}
bookmarks = {}


# User Authentication (JWT)
def generate_token(username):
    expiration = datetime.utcnow() + timedelta(days=1)
    return jwt.encode({'username': username, 'exp': expiration}, app.config['SECRET_KEY'], algorithm='HS256')


def verify_token(token):
    try:
        data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return data['username']
    except:
        return None


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

    token = generate_token(username)
    return jsonify({'status': 'success', 'token': token})


# Recipe Indexing Class
class RecipeIndexer:
    def __init__(self, file_path='resource/recipes.csv', is_reset=False):
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

    def preprocess_text(self, text):
        return re.sub(r'[^\w\s]', '', text.lower()).strip()

    def run_indexer(self):
        df = pd.read_csv(self.file_path)
        df['text_data'] = df[['Description', 'RecipeIngredientParts', 'RecipeInstructions']].fillna('').agg(' '.join,
                                                                                                            axis=1)
        corpus = df['text_data'].apply(self.preprocess_text).tolist()

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.recipes_df = df

        with open(self.stored_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def search_query(self, query, top_n=10):
        query_vector = self.vectorizer.transform([self.preprocess_text(query)])
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        self.recipes_df['score'] = similarity_scores
        return self.recipes_df.nlargest(top_n, 'score')[
            ['RecipeId', 'Name', 'Images', 'Description', 'RecipeIngredientParts', 'RecipeInstructions',
             'score', 'TotalTime', 'Calories']].to_dict('records')


# Initialize Indexer
indexer = RecipeIndexer()


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    results = indexer.search_query(query, top_n=10)
    return jsonify({'status': 'success', 'results': results})


if __name__ == '__main__':
    app.run(debug=True)
