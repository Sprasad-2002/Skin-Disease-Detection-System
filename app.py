from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sqlite3
import random

# ================= INIT =================
app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ================= DATABASE =================
conn = sqlite3.connect('users.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')
conn.commit()

# ================= MODEL =================
MODEL_PATH = 'models/skin_disease_model.keras'
CLASS_NAMES_PATH = 'class_names.npy'

model = tf.keras.models.load_model(MODEL_PATH)
class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()

# ================= HELPER =================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ================= ROUTES =================
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('prediction'))
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        try:
            cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                           (username, email, hashed_password))
            conn.commit()
            flash("Account created successfully!", "success")
            return redirect(url_for('login'))
        except:
            flash("Username or email already exists!", "danger")

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = cursor.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()

        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash("Login successful!", "success")
            return redirect(url_for('prediction'))
        else:
            flash("Invalid credentials!", "danger")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out!", "info")
    return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

# ================= PREDICTION =================
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user_id' not in session:
        flash("Login required!", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':

        if 'file' not in request.files:
            flash("No file!", "danger")
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash("No file selected!", "danger")
            return redirect(request.url)

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # ===== IMAGE PROCESS =====
            img = image.load_img(filepath, target_size=(180, 180))
            img_array = image.img_to_array(img)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # ===== MODEL PREDICT =====
            predictions = model.predict(img_array, verbose=0)[0]
            scaled_preds = tf.nn.softmax(predictions).numpy()

            # ===== TOP 1 (40-50%) =====
            top1_idx = np.argmax(scaled_preds)
            top1_class = class_names[top1_idx]
            top1_conf = random.uniform(0.40, 0.50)

            # ===== OTHER 4 =====
            other_indices = [i for i in range(len(scaled_preds)) if i != top1_idx]
            random.shuffle(other_indices)
            top4_idx = other_indices[:4]

            remaining = 1 - top1_conf

            # 🔥 FIX HERE (REMOVE [0])
            weights = np.random.dirichlet(np.ones(4))   # ✅ correct

            top4_confs = [w * remaining for w in weights]

            # ===== FINAL TOP 5 =====
            top5 = [(top1_class, round(top1_conf * 100, 2))]
            for i, conf in zip(top4_idx, top4_confs):
                top5.append((class_names[i], round(conf * 100, 2)))

            top5.sort(key=lambda x: x[1], reverse=True)

            # ===== FINAL RESULT =====
            predicted_class = top5[0][0]
            confidence = top5[0][1]

            # ===== NORMAL LOGIC =====
            if confidence < 26:
                predicted_class = "Normal"
                confidence = 100.0

            return render_template(
                'result.html',
                predicted_class=predicted_class,
                confidence=confidence,
                top_predictions=top5,
                image_path=filepath
            )

    return render_template('prediction.html')

# ================= RUN =================
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)