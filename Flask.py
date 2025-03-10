import os
import pickle
import pandas as pd
import re
import time
import jwt
import bcrypt
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, g
from flask_cors import CORS

# Flask Setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # Allow all origins
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Add a secret key for JWT
app.config['DATABASE'] = 'recipe_app.db'  # Database file path


# Database handling functions
def get_db():
    """Connect to the database and return the connection."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
        db.row_factory = sqlite3.Row  # Return rows as dictionaries
    return db


@app.teardown_appcontext
def close_connection(exception):
    """Close the database connection when the application context ends."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    """Initialize the database with necessary tables."""
    with app.app_context():
        db = get_db()
        cursor = db.cursor()

        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create saved_recipes table (for user favorites)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            recipe_id TEXT,
            saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')

        db.commit()


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
    def extract_image_url(text):
        """Extract and clean the image URL from various formats."""
        if not isinstance(text, str) or not text.strip():
            return ""  # Return empty if the field is missing or not a string

        # Extract URL with regex pattern matching common image URL patterns
        url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif)'
        matches = re.findall(url_pattern, text)

        if matches:
            # Return the first valid image URL found
            return matches[0]

        # Remove c(" at the start and ") at the end
        text = re.sub(r'^c\(["\s]*', '', text)  # Remove 'c("'
        text = re.sub(r'["\s]*\)$', '', text)  # Remove '")'
        # Remove any leading and trailing quotes
        text = re.sub(r'^"+|"+$', '', text)

        # Check if the cleaned text is a valid URL
        if text.startswith('http') and ('jpg' in text or 'jpeg' in text or 'png' in text or 'gif' in text):
            return text.strip()

        return ""  # Return empty if no valid URL found

    def run_indexer(self):
        """Reads the CSV, processes text data, and indexes it."""
        df = pd.read_csv(self.file_path)

        # Process relevant fields
        df['RecipeIngredientParts'] = df['RecipeIngredientParts'].astype(str).apply(self.preprocess_text)
        df['RecipeInstructions'] = df['RecipeInstructions'].astype(str).apply(self.preprocess_text)

        # Handle image links - consolidate to a single field
        df['image_url'] = ""

        # Try to extract from 'image_link' first
        if 'image_link' in df.columns:
            df['image_url'] = df['image_link'].astype(str).apply(self.extract_image_url)

        # If no valid URL found, try alternative fields (fallback mechanism)
        mask = df['image_url'] == ""
        if mask.any() and 'Images' in df.columns:
            df.loc[mask, 'image_url'] = df.loc[mask, 'Images'].astype(str).apply(self.extract_image_url)

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
            ['RecipeId', 'Name', 'Description', 'image_url', 'RecipeIngredientParts', 'RecipeInstructions', 'score',
             'TotalTime', 'Calories']].to_dict('records')


# Initialize database
with app.app_context():
    init_db()

# Initialize Indexer
indexer = RecipeIndexer()


def get_user_by_username(username):
    """Get user data from the database by username."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    return cursor.fetchone()


def get_user_by_id(user_id):
    """Get user data from the database by ID."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()


def authenticate_token():
    """Validate the JWT token from request headers."""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None

    token = auth_header.split(' ')[1]
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user = get_user_by_username(payload['username'])
        return user
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


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

    if not username or not password:
        return jsonify({'status': 'error', 'message': 'Username and password are required'}), 400

    # Check if user already exists
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    existing_user = cursor.fetchone()

    if existing_user:
        return jsonify({'status': 'error', 'message': 'User already exists'}), 400

    # Hash password and insert new user
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
    db.commit()

    return jsonify({'status': 'success', 'message': 'User registered successfully'})


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'status': 'error', 'message': 'Username and password are required'}), 400

    user = get_user_by_username(username)

    if not user or not bcrypt.checkpw(password.encode(), user['password'].encode()):
        return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401

    return jsonify({'status': 'success', 'message': 'Login successful', 'user_id': user['id']})

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    try:
        results = indexer.search_query(query, top_n=50)
        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/save_recipe', methods=['POST'])
def save_recipe():
    """Save a recipe to user's favorites."""
    user = authenticate_token()
    if not user:
        return jsonify({'status': 'error', 'message': 'Authentication required'}), 401

    data = request.json
    recipe_id = data.get('recipe_id')

    if not recipe_id:
        return jsonify({'status': 'error', 'message': 'Recipe ID is required'}), 400

    db = get_db()
    cursor = db.cursor()

    # Check if recipe is already saved by this user
    cursor.execute(
        "SELECT * FROM saved_recipes WHERE user_id = ? AND recipe_id = ?",
        (user['id'], recipe_id)
    )

    if cursor.fetchone():
        return jsonify({'status': 'error', 'message': 'Recipe already saved'}), 400

    # Save the recipe
    cursor.execute(
        "INSERT INTO saved_recipes (user_id, recipe_id) VALUES (?, ?)",
        (user['id'], recipe_id)
    )
    db.commit()

    return jsonify({'status': 'success', 'message': 'Recipe saved successfully'})


@app.route('/saved_recipes', methods=['GET'])
def get_saved_recipes():
    """Get all saved recipes for the authenticated user."""
    user = authenticate_token()
    if not user:
        return jsonify({'status': 'error', 'message': 'Authentication required'}), 401

    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "SELECT recipe_id FROM saved_recipes WHERE user_id = ? ORDER BY saved_at DESC",
        (user['id'],)
    )

    saved_recipe_ids = [row['recipe_id'] for row in cursor.fetchall()]

    # Get full recipe details for each saved ID
    saved_recipes = []
    for recipe_id in saved_recipe_ids:
        # Find recipe in indexer data
        recipe = next(
            (r for r in indexer.recipes_df.to_dict('records') if str(r['RecipeId']) == recipe_id),
            None
        )
        if recipe:
            saved_recipes.append(recipe)

    return jsonify({'status': 'success', 'recipes': saved_recipes})


@app.route('/remove_saved_recipe', methods=['DELETE'])
def remove_saved_recipe():
    """Remove a recipe from user's saved recipes."""
    user = authenticate_token()
    if not user:
        return jsonify({'status': 'error', 'message': 'Authentication required'}), 401

    recipe_id = request.args.get('recipe_id')
    if not recipe_id:
        return jsonify({'status': 'error', 'message': 'Recipe ID is required'}), 400

    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "DELETE FROM saved_recipes WHERE user_id = ? AND recipe_id = ?",
        (user['id'], recipe_id)
    )
    db.commit()

    return jsonify({'status': 'success', 'message': 'Recipe removed from saved list'})


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
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    app.run(debug=False)