from flask import Flask, render_template, request, redirect, url_for, flash,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_cors import CORS
import os
import cv2
import numpy as np
from keras.models import model_from_json
import base64
from sqlalchemy.sql import text
from sqlalchemy.orm import scoped_session
import requests
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://youdp/dbname'
app.config['SECRET_KEY'] = 'your_secret_key'

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
CORS(app)
API_KEY = 'yourapi'

def convert_to_python_types(data):
    """Recursively convert NumPy int64 to Python int."""
    if isinstance(data, list):
        return [convert_to_python_types(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_python_types(value) for key, value in data.items()}
    elif isinstance(data, np.int64):  # Convert int64 to int
        return int(data)
    else:
        return data
def fetch_movie_details(movie_id):
    """Fetch movie details (including poster path) from TMDB."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&append_to_response=keywords"
    response = requests.get(url, verify=False)

    if response.status_code != 200:
        return None

    data = response.json()

    genres = [genre['name'] for genre in data.get('genres', [])]
    keywords = [keyword['name'] for keyword in data.get('keywords', {}).get('keywords', [])]
    overview = data.get('overview', '')

   
    poster_path = data.get('poster_path', None)
    image = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/200x300?text=No+Image"

    return {
        'id': movie_id,
        'title': data.get('title', ''),
        'genres': " ".join(genres),
        'keywords': " ".join(keywords),
        'overview': overview,
        'image': image  
    }


def get_content_based_recommendations(user_id):
    """Generate recommendations using Content-Based Filtering."""
    
   
    liked_movies = db.session.execute(
        text("SELECT movie_id FROM liked_movies WHERE user_id = :user_id"),
        {"user_id": user_id}
    ).fetchall()

    if not liked_movies:
        print("No liked movies found.")
        return []

    movie_ids = [movie.movie_id for movie in liked_movies]

    # Fetch details of liked movies
    movie_data = []
    for movie_id in movie_ids:
        details = fetch_movie_details(movie_id)
        if details:
            movie_data.append(details)

    if not movie_data:
        print("No movie details found.")
        return []

   
    additional_movies = []
    for page in range(1, 3):  
        response = requests.get(f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&page={page}",verify=False)
        if response.status_code == 200:
            for movie in response.json().get('results', []):
                details = fetch_movie_details(movie['id'])
                if details:
                    additional_movies.append(details)

    
    all_movies = movie_data + additional_movies

    
    df = pd.DataFrame(all_movies)

    
    df['content'] = df['genres'] + " " + df['keywords'] + " " + df['overview']

    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['content'])

  
    cosine_sim = cosine_similarity(tfidf_matrix[:len(movie_data)], tfidf_matrix)

   
    similar_movies = {}
    for idx, movie_id in enumerate(df['id'][:len(movie_data)]): 
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  

        for sim_idx, score in sim_scores:
            similar_movie_id = df['id'][sim_idx]
            if similar_movie_id not in movie_ids: 
                similar_movies[similar_movie_id] = score

    
    recommended_movie_ids = sorted(similar_movies, key=similar_movies.get, reverse=True)[:10]

   
    recommended_movies = []
    for movie_id in recommended_movie_ids:
        details = fetch_movie_details(movie_id)
        if details:
            recommended_movies.append(details)

    print("Final Recommended Movies:", recommended_movies)  
    return recommended_movies



def detect_emotion(img):
    try:
        with open(r"emotion_model_equal.json", 'r') as json_file:
            loaded_model_json = json_file.read()
        emotion_model = model_from_json(loaded_model_json)
        emotion_model.load_weights(r"emotion_model_equal.weights.h5")

        frame = cv2.resize(img, (1280, 720))
        face_detector = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        maxindex = None

        for (x, y, w, h) in num_faces:
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))

        emotion_map = {
            0: "Angry",
            1: "Happy",
            2: "Neutral",
            3: "Sad",
            4: "Surprised"
        }
        
        return emotion_map.get(maxindex, "Neutral")
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Neutral"

# API route for emotion detection
@app.route('/detect-emotion', methods=['POST'])
def process_image():
    try:
       
        image_data = request.json['image'].split(',')[1]
        
       
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        
        emotion = detect_emotion(img)
        
        return jsonify({'emotion': emotion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def is_valid_email(email):
    """Check if the email format is valid."""
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        
        if not is_valid_email(email):
            return render_template('register.html', error="Invalid email format. Please enter a valid email.")

       
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            if existing_user.username == username:
                return render_template('register.html', error="Username is already taken. Please choose another.")
            elif existing_user.email == email:
                return render_template('register.html', error="Email is already registered. Please log in.")

        
        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

       
        if not is_valid_email(email):
            return render_template('login.html', error="Invalid email format. Please enter a valid email.")

        user = User.query.filter_by(email=email).first()
        if not user:
            return render_template('login.html', error="User not found.")
        if not bcrypt.check_password_hash(user.password, password):
            return render_template('login.html', error="Invalid credentials.")

        login_user(user)
        flash(f'Welcome back, {user.username}!', 'success')
        return redirect(url_for('index'))

    return render_template('login.html')



@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))
    
@app.route('/liked-movies', methods=['POST', 'DELETE'])
@login_required
def toggle_like_movie():
    data = request.json
    movie_id = data.get('movie_id')
    user_id = current_user.id

    if request.method == 'POST':
       
        existing_like = db.session.execute(
            text("SELECT * FROM liked_movies WHERE user_id = :user_id AND movie_id = :movie_id"),
            {"user_id": user_id, "movie_id": movie_id}
        ).fetchone()

        if existing_like:
            return jsonify({"message": "Movie already liked"}), 400

     
        db.session.execute(
            text("INSERT INTO liked_movies (user_id, movie_id) VALUES (:user_id, :movie_id)"),
            {"user_id": user_id, "movie_id": movie_id}
        )
        db.session.commit()
        return jsonify({"message": "Movie liked successfully!"})

    elif request.method == 'DELETE':
       
        db.session.execute(
            text("DELETE FROM liked_movies WHERE user_id = :user_id AND movie_id = :movie_id"),
            {"user_id": user_id, "movie_id": movie_id}
        )
        db.session.commit()
        return jsonify({"message": "Movie unliked successfully!"})

@app.route('/liked-movies', methods=['GET'])
@login_required
def get_liked_movies():
    user_id = current_user.id
    liked_movies = db.session.execute(
        text("SELECT movie_id FROM liked_movies WHERE user_id = :user_id"),
        {"user_id": user_id}
    ).fetchall()

    movie_ids = [movie.movie_id for movie in liked_movies]

   
    movies = []
    for movie_id in movie_ids:
        response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=b4c3fd4bd79a7a30f43668f17e0d25bb',verify=False)
        if response.status_code == 200:
            movies.append(response.json())

    return jsonify(movies)

@app.route('/liked-movies-page')
@login_required
def liked_movies_page():
    return render_template('liked_movies.html')




@app.route('/for-you', methods=['GET'])
@login_required
def for_you_recommendations():
    movies = get_content_based_recommendations(current_user.id)
    
    
    movies_cleaned = convert_to_python_types(movies)
    
    return jsonify(movies_cleaned)  



@app.route('/')
@login_required
def index():
    user_id = current_user.id
    liked_movies = db.session.execute(
        text("SELECT movie_id FROM liked_movies WHERE user_id = :user_id"),
        {"user_id": user_id}
    ).fetchall()

    liked_movie_ids = [movie.movie_id for movie in liked_movies]

    return render_template('final2.html', liked_movie_ids=liked_movie_ids, current_user=current_user)


if __name__ == '__main__':
    # Create the database tables (only needed the first time)   
    with app.app_context():
        db.create_all()
    app.run(port=5000, debug=True)
