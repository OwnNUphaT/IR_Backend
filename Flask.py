import os
import pickle
import urllib

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
from textblob import TextBlob
from rank_bm25 import BM25Okapi

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

        # Create folders table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')

        # Modify saved_recipes table to include a rating column (1-5)
        cursor.execute('''
                CREATE TABLE IF NOT EXISTS saved_recipes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    folder_id INTEGER NOT NULL,
                    recipe_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    image_url TEXT,
                    ingredients TEXT,
                    instructions TEXT,
                    total_time TEXT,
                    calories TEXT,
                    rating INTEGER DEFAULT NULL,  -- Store rating from 1-5
                    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (folder_id) REFERENCES folders (id)
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
        """Extract and clean a valid image URL from the 'image_url' field."""
        if not isinstance(text, str) or not text.strip():
            return ""

        text = text.strip().replace("c(", "").replace(")", "").replace('"', '').replace('\\n', '')
        text = urllib.parse.unquote(text)

        url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif)'
        matches = re.findall(url_pattern, text)

        if matches:
            return matches[0]

        return text if text.startswith('http') else ""

    def run_indexer(self):
        """Reads the CSV, processes text data, and indexes it using BM25."""
        df = pd.read_csv(self.file_path)

        df['RecipeIngredientParts'] = df['RecipeIngredientParts'].astype(str).apply(self.preprocess_text)
        df['RecipeInstructions'] = df['RecipeInstructions'].astype(str).apply(self.preprocess_text)

        df['image_url'] = ""
        if 'image_link' in df.columns:
            df['image_url'] = df['image_link'].astype(str).apply(self.extract_image_url)

        mask = df['image_url'] == ""
        if mask.any() and 'Images' in df.columns:
            df.loc[mask, 'image_url'] = df.loc[mask, 'Images'].astype(str).apply(self.extract_image_url)

        # Tokenize text for BM25
        df['tokenized_text'] = df[['RecipeIngredientParts', 'RecipeInstructions']].fillna('').agg(' '.join, axis=1)
        self.tokenized_corpus = [doc.split() for doc in df['tokenized_text'].tolist()]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.recipes_df = df

        with open(self.stored_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

        print("BM25 index created successfully.")

    def search_query(self, query, top_n=50):
        query = self.preprocess_text(query).split()
        scores = self.bm25.get_scores(query)
        self.recipes_df['score'] = scores

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


def authenticate_user(username):
    """Check if a username exists in the database."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    return cursor.fetchone()  # Returns user record if found, else None



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
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify({'status': 'error', 'message': 'Empty search query'}), 400

    corrected_query = str(TextBlob(query).correct())
    if corrected_query.lower() == query.lower():
        corrected_query = None

    try:
        results = indexer.search_query(query, top_n=50)
        return jsonify({'status': 'success', 'results': results, 'corrected_query': corrected_query})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})



@app.route('/create_folder', methods=['POST'])
def create_folder():
    """Create a personal folder for the user without using JWT."""
    data = request.json
    username = data.get('username')
    folder_name = data.get('folder_name')

    if not username or not folder_name:
        return jsonify({'status': 'error', 'message': 'Username and folder name are required'}), 400

    user = authenticate_user(username)
    if not user:
        return jsonify({'status': 'error', 'message': 'Invalid username'}), 401

    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO folders (user_id, name) VALUES (?, ?)",
        (user['id'], folder_name)
    )
    db.commit()

    return jsonify({'status': 'success', 'message': 'Folder created successfully'})


@app.route('/delete_folder', methods=['DELETE'])
def delete_folder():
    """Delete a folder and all its saved recipes."""
    data = request.json
    username = data.get('username')
    folder_id = data.get('folder_id')

    if not username or not folder_id:
        return jsonify({'status': 'error', 'message': 'Username and Folder ID are required'}), 400

    user = get_user_by_username(username)  # Get user ID from username
    if not user:
        return jsonify({'status': 'error', 'message': 'Invalid username'}), 401

    db = get_db()
    cursor = db.cursor()

    # Check if the folder exists
    cursor.execute("SELECT * FROM folders WHERE id = ? AND user_id = ?", (folder_id, user['id']))
    folder = cursor.fetchone()
    if not folder:
        return jsonify(
            {'status': 'error', 'message': 'Folder not found or you don’t have permission to delete it'}), 403

    # Delete all saved recipes inside the folder
    cursor.execute("DELETE FROM saved_recipes WHERE folder_id = ?", (folder_id,))

    # Delete the folder
    cursor.execute("DELETE FROM folders WHERE id = ?", (folder_id,))

    db.commit()

    return jsonify({'status': 'success', 'message': 'Folder deleted successfully'})


@app.route('/folders', methods=['POST'])
def get_folders():
    """Retrieve folders for a user without using JWT."""
    data = request.json
    username = data.get('username')

    if not username:
        return jsonify({'status': 'error', 'message': 'Username is required'}), 400

    user = authenticate_user(username)
    if not user:
        return jsonify({'status': 'error', 'message': 'Invalid username'}), 401

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM folders WHERE user_id = ?", (user['id'],))
    folders = [dict(row) for row in cursor.fetchall()]

    return jsonify({'status': 'success', 'folders': folders})


@app.route('/save_recipe', methods=['POST'])
def save_recipe():
    """Save a recipe for a specific user with a rating."""
    data = request.json
    username = data.get('username')
    folder_id = data.get('folder_id')
    recipe = data.get('recipe')
    rating = data.get('rating')  # Capture rating

    if not username or not folder_id or not recipe or rating is None:
        return jsonify({'status': 'error', 'message': 'Username, folder ID, recipe data, and rating are required'}), 400

    user = authenticate_user(username)
    if not user:
        return jsonify({'status': 'error', 'message': 'Invalid username'}), 401

    db = get_db()
    cursor = db.cursor()

    # Insert recipe along with rating
    cursor.execute(
        "INSERT INTO saved_recipes (user_id, folder_id, recipe_id, name, description, image_url, ingredients, instructions, total_time, calories, rating) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user['id'], folder_id, recipe.get('RecipeId'), recipe.get('Name'), recipe.get('Description'),
            recipe.get('image_url'), recipe.get('RecipeIngredientParts'), recipe.get('RecipeInstructions'),
            recipe.get('TotalTime'), recipe.get('Calories'), rating
        )
    )
    db.commit()

    return jsonify({'status': 'success', 'message': 'Recipe saved with rating successfully'})


@app.route('/update_rating', methods=['POST'])
def update_rating():
    """Update the rating of a saved recipe in a folder."""
    data = request.json
    username = data.get('username')
    folder_id = data.get('folder_id')
    recipe_id = data.get('recipe_id')
    rating = data.get('rating')

    if not username or not folder_id or not recipe_id or rating is None:
        return jsonify({'status': 'error', 'message': 'Username, Folder ID, Recipe ID, and rating are required'}), 400

    if rating not in range(1, 6):  # Rating must be between 1 and 5
        return jsonify({'status': 'error', 'message': 'Rating must be between 1 and 5'}), 400

    user = authenticate_user(username)
    if not user:
        return jsonify({'status': 'error', 'message': 'Invalid username'}), 401

    db = get_db()
    cursor = db.cursor()

    cursor.execute(
        "UPDATE saved_recipes SET rating = ? WHERE user_id = ? AND folder_id = ? AND recipe_id = ?",
        (rating, user['id'], folder_id, recipe_id)
    )
    db.commit()

    return jsonify({'status': 'success', 'message': 'Rating updated successfully'})


@app.route('/remove_saved_recipe', methods=['DELETE'])
def remove_saved_recipe():
    """Remove a saved recipe from a folder for a user without using JWT."""
    data = request.json
    username = data.get('username')
    recipe_id = data.get('recipe_id')
    folder_id = data.get('folder_id')

    if not username or not recipe_id or not folder_id:
        return jsonify({'status': 'error', 'message': 'Username, Recipe ID, and Folder ID are required'}), 400

    user = authenticate_user(username)  # Verify username
    if not user:
        return jsonify({'status': 'error', 'message': 'Invalid username'}), 401

    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "DELETE FROM saved_recipes WHERE user_id = ? AND folder_id = ? AND recipe_id = ?",
        (user['id'], folder_id, recipe_id)
    )
    db.commit()

    return jsonify({'status': 'success', 'message': 'Recipe removed from folder'})


@app.route('/folder_recipes/<int:folder_id>', methods=['POST'])
def get_folder_recipes(folder_id):
    """Retrieve saved recipes for a user inside a specific folder, ranked by rating."""
    data = request.json
    username = data.get('username')

    if not username:
        return jsonify({'status': 'error', 'message': 'Username is required'}), 400

    user = authenticate_user(username)
    if not user:
        return jsonify({'status': 'error', 'message': 'Invalid username'}), 401

    db = get_db()
    cursor = db.cursor()

    # Fetch recipes sorted by rating (higher rating first)
    cursor.execute("SELECT * FROM saved_recipes WHERE user_id = ? AND folder_id = ? ORDER BY rating DESC, saved_at DESC", (user['id'], folder_id))
    recipes = [dict(row) for row in cursor.fetchall()]

    return jsonify({'status': 'success', 'recipes': recipes})


@app.route('/personalized_recommendation', methods=['POST'])
def personalized_recommendation():
    data = request.json
    username = data.get('username')
    if not username:
        return jsonify({'status': 'error', 'message': 'Username required'}), 400

    user = authenticate_user(username)
    if not user:
        return jsonify({'status': 'error', 'message': 'Invalid user'}), 401

    db = get_db()
    cursor = db.cursor()

    # Use a single query to get both recipes and saved IDs
    cursor.execute("""
        SELECT r.ingredients, r.instructions, 
               GROUP_CONCAT(DISTINCT sr.recipe_id) as saved_ids
        FROM saved_recipes r
        JOIN (SELECT user_id, recipe_id FROM saved_recipes WHERE user_id = ?) sr
        ON sr.user_id = ?
        WHERE r.user_id = ?
        GROUP BY r.user_id
    """, (user['id'], user['id'], user['id']))

    result = cursor.fetchone()
    if not result:
        return jsonify({'status': 'error', 'message': 'No saved recipes.'}), 404

    # Parse saved IDs into a set for faster lookups
    saved_ids = set(result['saved_ids'].split(',') if result['saved_ids'] else [])

    # Process all recipes data at once
    cursor.execute("SELECT ingredients, instructions FROM saved_recipes WHERE user_id = ?", (user['id'],))
    saved_data = cursor.fetchall()

    # Cache the combined query for potential reuse
    combined_query = ' '.join(f"{row['ingredients']} {row['instructions']}" for row in saved_data)

    # Use vectorized operations if possible with your indexer
    candidates = indexer.search_query(combined_query, top_n=20)  # Fetch more to account for filtering

    # Optimize filtering with set operations
    recommendations = []
    count = 0

    for r in candidates:
        if str(r['RecipeId']) not in saved_ids:
            recommendations.append(r)
            count += 1
            if count >= 6:
                break

    return jsonify({'status': 'success', 'recommendations': recommendations})


@app.route('/generate_suggestions', methods=['POST'])
def generate_suggestions():
    data = request.json
    username = data.get('username')
    folder_id = data.get('folder_id')

    if not username or not folder_id:
        return jsonify({'status': 'error', 'message': 'Username and folder_id required'}), 400

    user = authenticate_user(username)
    if not user:
        return jsonify({'status': 'error', 'message': 'Invalid user'}), 401

    db = get_db()
    cursor = db.cursor()

    # Combine both queries into one to reduce database round trips
    cursor.execute("""
        SELECT 
            r.ingredients, r.instructions, 
            GROUP_CONCAT(DISTINCT sr.recipe_id) as folder_recipe_ids
        FROM saved_recipes r
        LEFT JOIN (
            SELECT recipe_id 
            FROM saved_recipes 
            WHERE user_id = ? AND folder_id = ?
        ) sr ON 1=1
        WHERE r.user_id = ? AND r.folder_id = ?
        GROUP BY r.folder_id
    """, (user['id'], folder_id, user['id'], folder_id))

    result = cursor.fetchone()
    if not result:
        return jsonify({'status': 'error', 'message': 'Folder is empty.'}), 404

    # Parse folder recipe IDs
    folder_recipe_ids = set(result['folder_recipe_ids'].split(',') if result['folder_recipe_ids'] else [])

    # Get all folder data in one go
    cursor.execute("SELECT ingredients, instructions FROM saved_recipes WHERE user_id = ? AND folder_id = ?",
                   (user['id'], folder_id))
    folder_data = cursor.fetchall()

    # Cache and reuse the combined query
    combined_query = ' '.join(f"{row['ingredients']} {row['instructions']}" for row in folder_data)

    # Search with a batch operation and get more candidates to account for filtering
    suggestions = indexer.search_query(combined_query, top_n=20)

    # Use early termination in filtering
    final_suggestions = []
    count = 0

    for r in suggestions:
        if str(r['RecipeId']) not in folder_recipe_ids:
            final_suggestions.append(r)
            count += 1
            if count >= 5:
                break

    return jsonify({'status': 'success', 'suggestions': final_suggestions})


# Optional caching layer if these endpoints are called frequently
from functools import lru_cache


# Cache user authentication results
@lru_cache(maxsize=100)
def cached_authenticate_user(username):
    return authenticate_user(username)


# If the indexer search is expensive, consider adding a cache
@lru_cache(maxsize=50)
def cached_search_query(query, top_n):
    return indexer.search_query(query, top_n)





if __name__ == '__main__':
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    app.run(debug=False)