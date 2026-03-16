
import os
import json
import sqlite3
import base64
import threading
import cv2
import numpy as np
import pyttsx3
from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, Response
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# YOLO for Motion Detection
try:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")  # Use the nano model for speed
except ImportError:
    yolo_model = None
    print("Warning: ultralytics is not installed. Motion detection will fallback to basic heuristic.")

import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    task_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    hand_detector = vision.HandLandmarker.create_from_options(options)
except Exception as e:
    print(f"Warning: Failed to load hand_landmarker.task: {e}")
    hand_detector = None

app = Flask(__name__)
app.secret_key = "kinder_edge_ai_secret_2024"

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "school.db")

# ─────────────────────────────────────────────
# TTS Engine (thread-safe singleton)
# ─────────────────────────────────────────────
tts_lock = threading.Lock()

def speak(text: str):
    def _run():
        with tts_lock:
            engine = pyttsx3.init()
            engine.setProperty("rate", 130)
            engine.setProperty("volume", 1.0)
            # Pick an appropriate English voice
            voices = engine.getProperty("voices")
            en_voice_id = None
            for voice in voices:
                langs = str(voice.languages).lower()
                vid = voice.id.lower()
                if 'en' in langs or 'en' in vid:
                    if 'samantha' in voice.name.lower():
                        en_voice_id = voice.id
                        break
                    if not en_voice_id:
                        en_voice_id = voice.id
            if en_voice_id:
                engine.setProperty("voice", en_voice_id)
            
            engine.say(text)
            engine.runAndWait()
    t = threading.Thread(target=_run, daemon=True)
    t.start()

# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            school_name TEXT NOT NULL,
            email TEXT UNIQUE,
            mobile TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_login TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS remembered_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identifier TEXT UNIQUE NOT NULL,
            school_name TEXT NOT NULL,
            last_used TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# MediaPipe Hand Gesture
# ─────────────────────────────────────────────

def count_fingers(frame_bytes: bytes) -> int:
    """Return number of fingers shown (0-10) from a base64 JPEG frame."""
    if hand_detector is None:
        return -1
        
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return -1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    results = hand_detector.detect(mp_image)

    if not results.hand_landmarks:
        return 0  # no hands = 0 fingers

    total_count = 0
    for hand_idx in range(len(results.hand_landmarks)):
        lm = results.hand_landmarks[hand_idx]
        tips = [4, 8, 12, 16, 20]
        pip  = [3, 6, 10, 14, 18]

        count = 0
        def dist(p1, p2):
            return math.hypot(p1.x - p2.x, p1.y - p2.y)
            
        # Thumb: compare distance from tip to pinky base vs thumb base to pinky base
        if dist(lm[4], lm[17]) > dist(lm[2], lm[17]):
            count += 1
            
        # Other 4 fingers: compare y-axis (tip above pip = extended)
        for i in range(1, 5):
            if lm[tips[i]].y < lm[pip[i]].y:
                count += 1
                
        total_count += count
        
    return total_count

# ─────────────────────────────────────────────
# MediaPipe Pose for Group Motion Game
# ─────────────────────────────────────────────
mp_pose = None  # Will be None since mediapipe.solutions.pose is not available

def count_people_and_pose(frame_bytes: bytes) -> dict:
    """Uses YOLOv8 to count the number of people in the frame."""
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"count": 0}
        
    if yolo_model is not None:
        # Run YOLO inference
        results = yolo_model(frame, classes=[0], verbose=False) # class 0 is person
        
        person_count = 0
        if len(results) > 0:
            # Count how many boxes were detected
            person_count = len(results[0].boxes)
            
        return {"count": person_count}
    else:
        # Fallback to old heuristic if YOLO isn't loaded
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        people = [c for c in contours if cv2.contourArea(c) > 3000]
        return {"count": len(people)}

# ─────────────────────────────────────────────
# Static Data
# ─────────────────────────────────────────────
RHYMES = [
    {"title": "Baby Shark", "thumbnail": "🦈", "url": "https://www.youtube.com/watch?v=XqZsoesa55w"},
    {"title": "Wheels on the Bus", "thumbnail": "🚌", "url": "https://www.youtube.com/watch?v=e_04ZrNroTo"},
    {"title": "Twinkle Twinkle", "thumbnail": "⭐", "url": "https://www.youtube.com/watch?v=yCjJyiqpAuU"},
    {"title": "Old MacDonald", "thumbnail": "🐄", "url": "https://www.youtube.com/watch?v=5HbdM4aP9Bg"},
    {"title": "Johny Johny Yes Papa", "thumbnail": "👶", "url": "https://www.youtube.com/watch?v=oaFEMgWkOoI"},
    {"title": "If You're Happy", "thumbnail": "😊", "url": "https://www.youtube.com/watch?v=71hqRT9U0wg"},
    {"title": "Baa Baa Black Sheep", "thumbnail": "🐑", "url": "https://www.youtube.com/watch?v=aBR_TKbhOcI"},
    {"title": "Mary Had a Little Lamb", "thumbnail": "🐏", "url": "https://www.youtube.com/watch?v=egGPtMCVjEo"},
    {"title": "Head Shoulders Knees", "thumbnail": "🕺", "url": "https://www.youtube.com/watch?v=ZanHgPprl-0"},
    {"title": "Cocomelon ABC Song", "thumbnail": "🍉", "url": "https://www.youtube.com/watch?v=75p-N9YKqNo"},
    {"title": "Five Little Monkeys", "thumbnail": "🐒", "url": "https://www.youtube.com/watch?v=JCNQ4OFFMf0"},
    {"title": "Row Row Your Boat", "thumbnail": "🚣", "url": "https://www.youtube.com/watch?v=7oyiPBjLAWY"},
]

ALPHABETS = [
    {"letter": L, "word": W, "emoji": E}
    for L, W, E in [
        ("A","Apple","🍎"), ("B","Ball","⚽"), ("C","Cat","🐱"), ("D","Dog","🐶"),
        ("E","Elephant","🐘"), ("F","Fish","🐟"), ("G","Grapes","🍇"), ("H","Hat","🎩"),
        ("I","Ice cream","🍦"), ("J","Juice","🧃"), ("K","Kite","🪁"), ("L","Lion","🦁"),
        ("M","Monkey","🐒"), ("N","Nest","🪺"), ("O","Orange","🍊"), ("P","Parrot","🦜"),
        ("Q","Queen","👑"), ("R","Rainbow","🌈"), ("S","Sun","☀️"), ("T","Train","🚂"),
        ("U","Umbrella","☂️"), ("V","Violin","🎻"), ("W","Watermelon","🍉"),
        ("X","Xylophone","🎵"), ("Y","Yogurt","🥛"), ("Z","Zebra","🦓"),
    ]
]

WORDS = [
    {"word": w, "emoji": e}
    for w, e in [
        ("Acorn","🌰"), ("Brave","🦸"), ("Cloud","☁️"), ("Dream","💭"), ("Eagle","🦅"),
        ("Flame","🔥"), ("Ghost","👻"), ("Honey","🍯"), ("Island","🏝️"), ("Jelly","🍮"),
        ("Koala","🐨"), ("Lemon","🍋"), ("Mouse","🐭"), ("Night","🌙"), ("Ocean","🌊"),
        ("Puppy","🐶"), ("Quilt","🛏️"), ("River","🏞️"), ("Smile","😊"), ("Train","🚂"),
        ("Uncle","👨"), ("Voice","🗣️"), ("Water","💧"), ("X-ray","🩻"),
        ("Yacht","🛥️"), ("Zebra","🦓"),
    ]
]

KARAOKE_RHYMES = [
    {
        "title": "Twinkle Twinkle Little Star",
        "lines": [
            "Twinkle, twinkle, little star,",
            "How I wonder what you are!",
            "Up above the world so high,",
            "Like a diamond in the sky.",
            "Twinkle, twinkle, little star,",
            "How I wonder what you are!",
        ]
    },
    {
        "title": "Baa Baa Black Sheep",
        "lines": [
            "Baa, baa, black sheep,",
            "Have you any wool?",
            "Yes sir, yes sir,",
            "Three bags full!",
            "One for the master,",
            "And one for the dame,",
            "And one for the little boy",
            "Who lives down the lane.",
        ]
    },
    {
        "title": "Mary Had a Little Lamb",
        "lines": [
            "Mary had a little lamb,",
            "Little lamb, little lamb,",
            "Mary had a little lamb,",
            "Its fleece was white as snow.",
            "And everywhere that Mary went,",
            "Mary went, Mary went,",
            "Everywhere that Mary went,",
            "The lamb was sure to go.",
        ]
    },
    {
        "title": "Johny Johny Yes Papa",
        "lines": [
            "Johny Johny, Yes Papa?",
            "Eating sugar? No Papa!",
            "Telling lies? No Papa!",
            "Open your mouth, Ha ha ha!",
            "Johny Johny, Yes Papa?",
            "Eating candy? No Papa!",
            "Telling lies? No Papa!",
            "Open your mouth, Ha ha ha!",
        ]
    },
    {
        "title": "Wheels on the Bus",
        "lines": [
            "The wheels on the bus go round and round,",
            "Round and round, round and round.",
            "The wheels on the bus go round and round,",
            "All through the town!",
            "The wipers on the bus go swish swish swish,",
            "Swish swish swish, swish swish swish.",
            "The wipers on the bus go swish swish swish,",
            "All through the town!",
        ]
    },
]

# ─────────────────────────────────────────────
# Routes – Auth
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    remembered = []
    conn = get_db()
    rows = conn.execute(
        "SELECT identifier, school_name FROM remembered_users ORDER BY last_used DESC LIMIT 5"
    ).fetchall()
    conn.close()
    remembered = [dict(r) for r in rows]

    if request.method == "POST":
        data = request.get_json()
        identifier = data.get("identifier", "").strip().lower()
        password   = data.get("password", "")

        conn = get_db()
        school = conn.execute(
            "SELECT * FROM schools WHERE LOWER(email)=? OR mobile=?",
            (identifier, identifier)
        ).fetchone()

        if school and check_password_hash(school["password_hash"], password):
            session["school_id"]   = school["id"]
            session["school_name"] = school["school_name"]
            conn.execute(
                "UPDATE schools SET last_login=? WHERE id=?",
                (datetime.now().isoformat(), school["id"])
            )
            conn.execute(
                """INSERT INTO remembered_users (identifier, school_name, last_used)
                   VALUES (?,?,?) ON CONFLICT(identifier)
                   DO UPDATE SET last_used=excluded.last_used""",
                (identifier, school["school_name"], datetime.now().isoformat())
            )
            conn.commit()
            conn.close()
            return jsonify({"success": True})
        conn.close()
        return jsonify({"success": False, "message": "Invalid credentials"})

    return render_template("login.html", remembered=remembered)

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    school_name = data.get("school_name", "").strip()
    email       = data.get("email", "").strip().lower() or None
    mobile      = data.get("mobile", "").strip() or None
    password    = data.get("password", "")

    if not school_name or not password or (not email and not mobile):
        return jsonify({"success": False, "message": "All fields required"})

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO schools (school_name, email, mobile, password_hash) VALUES (?,?,?,?)",
            (school_name, email, mobile, generate_password_hash(password))
        )
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"success": False, "message": "Email or mobile already registered"})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ─────────────────────────────────────────────
# Routes – Main App
# ─────────────────────────────────────────────
def require_login():
    return "school_id" not in session

@app.route("/home")
def home():
    if require_login():
        return redirect(url_for("login"))
    return render_template("home.html", school_name=session.get("school_name", ""))

# ── Rhymes ────────────────────────────────────
@app.route("/rhymes")
def rhymes():
    if require_login(): return redirect(url_for("login"))
    return render_template("rhymes.html", rhymes=RHYMES)

# ── Sing Along ────────────────────────────────
@app.route("/singalong")
def singalong():
    if require_login(): return redirect(url_for("login"))
    return render_template("singalong.html")

@app.route("/singalong/alphabets")
def alphabets():
    if require_login(): return redirect(url_for("login"))
    return render_template("alphabets.html", alphabets=ALPHABETS)

@app.route("/singalong/words")
def words():
    if require_login(): return redirect(url_for("login"))
    return render_template("words.html", words=WORDS)

@app.route("/singalong/karaoke")
def karaoke():
    if require_login(): return redirect(url_for("login"))
    return render_template("karaoke.html", rhymes=KARAOKE_RHYMES)

# ── Numbers Game ──────────────────────────────
@app.route("/numbers")
def numbers():
    if require_login(): return redirect(url_for("login"))
    return render_template("numbers.html")

@app.route("/api/detect_fingers", methods=["POST"])
def detect_fingers():
    if require_login(): return jsonify({"error": "not logged in"}), 401
    data = request.get_json()
    img_b64 = data.get("image", "")
    # Remove data URL prefix
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    frame_bytes = base64.b64decode(img_b64)
    count = count_fingers(frame_bytes)
    if count >= 0:
        speak(str(count))
    return jsonify({"fingers": count})

# ── Motion Games ──────────────────────────────
@app.route("/motion")
def motion():
    if require_login(): return redirect(url_for("login"))
    return render_template("motion.html")

@app.route("/api/detect_group", methods=["POST"])
def detect_group():
    if require_login(): return jsonify({"error": "not logged in"}), 401
    data = request.get_json()
    img_b64 = data.get("image", "")
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    frame_bytes = base64.b64decode(img_b64)
    result = count_people_and_pose(frame_bytes)
    return jsonify(result)

# ── TTS API ───────────────────────────────────
@app.route("/api/speak", methods=["POST"])
def api_speak():
    data = request.get_json()
    text = data.get("text", "")
    if text:
        speak(text)
    return jsonify({"ok": True})

# ─────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("=" * 50)
    print("  KinderLearn EdgeAI App Starting...")
    print("  Open http://localhost:5001 in browser")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5001, threaded=True)