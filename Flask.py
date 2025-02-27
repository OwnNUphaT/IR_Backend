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

    def preprocess_text(self, text, column=None):
        """Cleans text data by removing unwanted characters and keeping only one image link."""
        if isinstance(text, str):
            text = re.sub(r'^c["\s]*', '', text)  # Remove leading 'c', quotes, and spaces

            if column == 'Images':
                # Extract all valid URLs
                urls = re.findall(r'https?://\S+', text)
                return urls[0] if urls else ''  # Keep only the first URL if multiple exist

            text = re.sub(r'[^\w\s./:-]', '', text.lower()).strip()  # Keep valid characters
        return text

    def run_indexer(self):
        """Reads the CSV, processes text data, and indexes it."""
        df = pd.read_csv(self.file_path)

        # Apply preprocessing to clean unwanted `c` and formatting issues
        for column in ['RecipeIngredientParts', 'RecipeInstructions']:
            df[column] = df[column].astype(str).apply(lambda x: self.preprocess_text(x))

        # Special handling for Images: Remove multiple links and keep only one
        df['Images'] = df['Images'].astype(str).apply(lambda x: self.preprocess_text(x, column='Images'))

        df['text_data'] = df[['RecipeIngredientParts', 'RecipeInstructions']].fillna('').agg(' '.join, axis=1)
        corpus = df['text_data'].tolist()

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.recipes_df = df

        # Save the cleaned index
        with open(self.stored_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def assign_nearest_image(self, recipe_index):
        """Assigns the nearest recipe image if an image is missing."""
        if self.recipes_df.loc[recipe_index, 'Images']:
            return self.recipes_df.loc[recipe_index, 'Images']

        # Compute similarity with other recipes
        similarity_scores = cosine_similarity(
            self.tfidf_matrix[recipe_index], self.tfidf_matrix
        ).flatten()

        # Find nearest neighbor with an image
        sorted_indices = similarity_scores.argsort()[::-1]
        for idx in sorted_indices:
            if idx != recipe_index and self.recipes_df.loc[idx, 'Images']:
                return self.recipes_df.loc[idx, 'Images']

        return ''  # If no image is found, return empty string

    def search_query(self, query, top_n=50):
        """Searches for relevant recipes based on the input query."""
        query_vector = self.vectorizer.transform([self.preprocess_text(query)])
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        self.recipes_df['score'] = similarity_scores

        top_results = self.recipes_df.nlargest(top_n, 'score').copy()

        # Assign nearest image if missing
        for index in top_results.index:
            if not top_results.at[index, 'Images']:
                top_results.at[index, 'Images'] = self.assign_nearest_image(index)

        return top_results[
            ['RecipeId', 'Name', 'Images', 'Description', 'RecipeIngredientParts', 'RecipeInstructions',
             'score', 'TotalTime', 'Calories']
        ].to_dict('records')


# Initialize Indexer
indexer = RecipeIndexer()


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    results = indexer.search_query(query, top_n=50)
    return jsonify({'status': 'success', 'results': results})


if __name__ == '__main__':
    app.run(debug=True)
