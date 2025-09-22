from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
from pymongo import MongoClient
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import os
import logging

# ---------------------------
# Config
# ---------------------------
MONGO_URI = "mongodb+srv://beamlakbekele197_db_user:m5pG8eSXLFxM0Y9e@cluster0.mrqsfdq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "attendance_system"
THRESHOLD = 0.5
DEVICE_NAME = "Device-01"
ADMIN_USER = "admin"
ADMIN_PASS = "1234"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# MongoDB
# ---------------------------
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    students_col = db["students"]
    attendance_col = db["attendance"]
    admins_col = db["admins"]  # New collection for admin credentials
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    raise

# ---------------------------
# FaceAnalysis
# ---------------------------
try:
    app_face = FaceAnalysis(providers=['CPUExecutionProvider'])
    app_face.prepare(ctx_id=0, det_size=(640, 640))
except Exception as e:
    logger.error(f"FaceAnalysis initialization failed: {e}")
    raise

# ---------------------------
# Flask
# ---------------------------
app = Flask(__name__)
app.secret_key = "supersecret"

# Define datetimeformat filter
@app.template_filter('datetimeformat')
def datetimeformat(value):
    if isinstance(value, datetime.datetime):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    return ''

# ---------------------------
# Helper Functions
# ---------------------------
def load_known_faces():
    known_embeddings = {}
    try:
        for doc in students_col.find():
            known_embeddings[doc["student_id"]] = {
                "name": doc["name"],
                "embedding": np.array(doc["embedding"]),
                "email": doc.get("email", ""),
                "phone": doc.get("phone", ""),
                "department": doc.get("department", "")
            }
        logger.info(f"Loaded {len(known_embeddings)} faces")
    except Exception as e:
        logger.error(f"Failed to load faces: {e}")
    return known_embeddings

def log_attendance(student_id, name, device=DEVICE_NAME):
    try:
        now = datetime.datetime.now()
        attendance_col.insert_one({
            "student_id": student_id,
            "name": name,
            "timestamp": now,
            "device": device
        })
        logger.info(f"Attendance logged for {name}")
    except Exception as e:
        logger.error(f"Attendance logging failed: {e}")

def get_cap():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Camera failed to open")
            raise RuntimeError("Could not open camera")
    return cap

# ---------------------------
# Globals
# ---------------------------
known_embeddings = load_known_faces()
current_name = "No Face"
cap = None
is_registration_mode = False
last_unknown_face = None
last_unknown_frame = None

# ---------------------------
# Video Stream
# ---------------------------
def gen_frames():
    global current_name, known_embeddings, is_registration_mode, last_unknown_face, last_unknown_frame
    cap = get_cap()
    while True:
        success, frame = cap.read()
        if not success:
            logger.error("Camera read failed")
            break

        faces = app_face.get(frame)

        if len(faces) == 0:
            current_name = "No Face"
            last_unknown_face = None
            last_unknown_frame = None
        elif len(faces) > 1:
            current_name = "Multiple Faces"
            last_unknown_face = None
            last_unknown_frame = None
        else:
            face = faces[0]
            emb = face.embedding.reshape(1, -1)
            matched = False
            for sid, data in known_embeddings.items():
                sim = cosine_similarity(emb, data["embedding"].reshape(1, -1))[0][0]
                if sim > THRESHOLD:
                    current_name = data["name"]
                    matched = True
                    if not is_registration_mode:
                        log_attendance(sid, data["name"], DEVICE_NAME)
                    break
            if not matched:
                current_name = "Unknown"
                last_unknown_face = face
                last_unknown_frame = frame.copy()
            else:
                last_unknown_face = None
                last_unknown_frame = None

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, current_name if len(faces) == 1 else "Multiple", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------------------
# Routes
# ---------------------------
@app.route("/video_feed")
def video_feed():
    global is_registration_mode
    is_registration_mode = False
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/register_video_feed")
def register_video_feed():
    global is_registration_mode
    is_registration_mode = True
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global cap
    try:
        if cap is not None:
            cap.release()
            cap = None
            logger.info("Camera stopped")
        return jsonify({"status": "success", "message": "Camera stopped"})
    except Exception as e:
        logger.error(f"Camera stop failed: {e}")
        return jsonify({"status": "error", "message": "Failed to stop camera"}), 500

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def do_login():
    username = request.form.get("username")
    password = request.form.get("password")
    # Check hardcoded admin credentials
    if username == ADMIN_USER and password == ADMIN_PASS:
        session["admin"] = username
        return redirect(url_for("home"))
    # Check MongoDB admins collection
    admin = admins_col.find_one({"name": username, "password": password})
    if admin:
        session["admin"] = username
        return redirect(url_for("home"))
    return render_template("login.html", error="Invalid credentials")

@app.route("/home")
def home():
    if not session.get("admin"):
        return redirect(url_for("login"))
    return render_template("home.html")

@app.route("/register_page")
def register_page():
    if not session.get("admin"):
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/admin_register_page")
def admin_register_page():
    if not session.get("admin"):
        return redirect(url_for("login"))
    return render_template("admin_register.html")

@app.route("/recognize_status")
def recognize_status():
    return jsonify({"name": current_name})

@app.route("/register_unknown", methods=["POST"])
def register_unknown():
    global known_embeddings, last_unknown_face, last_unknown_frame
    data = request.json
    student_id = data["student_id"]
    name = data["name"]

    retry_count = 3
    while last_unknown_face is None and retry_count > 0:
        logger.info(f"Retrying face detection ({retry_count} left)")
        cap = get_cap()
        success, frame = cap.read()
        if success:
            faces = app_face.get(frame)
            if len(faces) == 1 and not any(cosine_similarity(faces[0].embedding.reshape(1, -1), 
                                                             data["embedding"].reshape(1, -1))[0][0] > THRESHOLD 
                                           for data in known_embeddings.values()):
                last_unknown_face = faces[0]
                last_unknown_frame = frame.copy()
        retry_count -= 1
        import time
        time.sleep(0.5)

    if last_unknown_face is None or last_unknown_frame is None:
        logger.error("No unknown face detected")
        return jsonify({"status": "error", "message": "No unknown face detected. Please try again."}), 400

    if last_unknown_face.det_score < 0.7:
        logger.error("Poor image quality")
        return jsonify({"status": "error", "message": "Poor image quality. Please ensure good lighting and clear face."}), 400

    if students_col.find_one({"student_id": student_id}):
        logger.error("Student ID already exists")
        return jsonify({"status": "error", "message": "Student ID already registered"}), 400

    embedding = last_unknown_face.embedding.tolist()

    try:
        os.makedirs('static/images', exist_ok=True)
        image_path = f'static/images/{student_id}.jpg'
        cv2.imwrite(image_path, last_unknown_frame)
        logger.info(f"Image saved for {student_id}")
    except Exception as e:
        logger.error(f"Image save failed: {e}")
        return jsonify({"status": "error", "message": "Failed to save image"}), 500

    try:
        students_col.insert_one({
            "student_id": student_id,
            "name": name,
            "embedding": embedding,
            "email": data["email"],
            "phone": data["phone"],
            "department": data["department"],
            "timestamp": datetime.datetime.now(),
            "admin_name": session.get("admin", ADMIN_USER)  # Use logged-in admin's name
        })
        logger.info(f"Registered {name}")
    except Exception as e:
        logger.error(f"Database save failed: {e}")
        return jsonify({"status": "error", "message": "Failed to save to database"}), 500

    known_embeddings = load_known_faces()
    last_unknown_face = None
    last_unknown_frame = None
    return jsonify({"status": "success", "message": f"{name} registered successfully!"})

# ... (other imports and code remain unchanged)

# ... (other imports and code remain unchanged)

@app.route("/admin_register", methods=["POST"])
def admin_register():
    global known_embeddings, last_unknown_face, last_unknown_frame
    data = request.json
    logger.info(f"Received payload: {data}")  # Log the payload
    admin_id = data.get("admin_id")
    name = data.get("name")
    email = data.get("email")
    phone = data.get("phone")
    password = data.get("password")

    # Validate input fields
    if not all([admin_id, name, email, phone, password]):
        logger.error("Missing required fields in admin registration")
        return jsonify({"status": "error", "message": "All fields (Admin ID, Name, Email, Phone, Password) are required"}), 400

    # Retry face detection
    retry_count = 5
    while last_unknown_face is None and retry_count > 0:
        logger.info(f"Retrying face detection ({retry_count} left)")
        cap = get_cap()
        success, frame = cap.read()
        if success:
            faces = app_face.get(frame)
            if len(faces) == 1:
                emb = faces[0].embedding.reshape(1, -1)
                is_known = any(cosine_similarity(emb, data["embedding"].reshape(1, -1))[0][0] > THRESHOLD 
                               for data in known_embeddings.values())
                if not is_known:
                    last_unknown_face = faces[0]
                    last_unknown_frame = frame.copy()
                    logger.info(f"Face detected with det_score: {last_unknown_face.det_score}")
        retry_count -= 1
        import time
        time.sleep(1)

    if last_unknown_face is None or last_unknown_frame is None:
        logger.error("No unknown face detected after retries")
        return jsonify({"status": "error", "message": "No unknown face detected. Ensure a single face is visible and try again."}), 400

    if last_unknown_face.det_score < 0.7:
        logger.error(f"Poor image quality: det_score={last_unknown_face.det_score}")
        return jsonify({"status": "error", "message": f"Poor image quality (det_score: {last_unknown_face.det_score:.2f}). Ensure good lighting and clear face."}), 400

    if students_col.find_one({"student_id": admin_id}):
        logger.error(f"Admin ID {admin_id} already exists")
        return jsonify({"status": "error", "message": f"Admin ID {admin_id} already registered"}), 400

    if admins_col.find_one({"name": name}):
        logger.error(f"Admin name {name} already exists")
        return jsonify({"status": "error", "message": f"Admin name {name} already registered"}), 400

    embedding = last_unknown_face.embedding.tolist()

    try:
        os.makedirs('static/images', exist_ok=True)
        image_path = f'static/images/{admin_id}.jpg'
        cv2.imwrite(image_path, last_unknown_frame)
        logger.info(f"Image saved for {admin_id}")
    except Exception as e:
        logger.error(f"Image save failed: {e}")
        return jsonify({"status": "error", "message": "Failed to save image due to server error"}), 500

    try:
        students_col.insert_one({
            "student_id": admin_id,
            "name": name,
            "embedding": embedding,
            "email": email,
            "phone": phone,
            "department": "Admin",
            "timestamp": datetime.datetime.now(),
            "admin_name": session.get("admin", ADMIN_USER)
        })
        admins_col.insert_one({
            "name": name,
            "password": password
        })
        logger.info(f"Registered admin {name}")
    except Exception as e:
        logger.error(f"Database save failed: {e}")
        return jsonify({"status": "error", "message": "Failed to save to database due to server error"}), 500

    known_embeddings = load_known_faces()
    last_unknown_face = None
    last_unknown_frame = None
    return jsonify({"status": "success", "message": f"Admin {name} registered successfully!"})

# ... (rest of the app.py code remains unchanged)

@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if not session.get("admin"):
        return redirect(url_for("login"))

    today = datetime.datetime.now().date()
    selected_year = request.form.get("year", today.year)
    selected_month = request.form.get("month", today.month)
    selected_day = request.form.get("day", today.day)

    selected_year = int(selected_year)
    selected_month = int(selected_month)
    selected_day = int(selected_day)

    start = datetime.datetime(selected_year, selected_month, selected_day, 0, 0, 0)
    end = datetime.datetime(selected_year, selected_month, selected_day, 23, 59, 59)

    attendance_records = list(attendance_col.find({"timestamp": {"$gte": start, "$lt": end}}))

    present = []
    present_ids = set()

    for rec in attendance_records:
        student = students_col.find_one({"student_id": rec["student_id"]})
        if student:
            student["timestamp"] = rec["timestamp"]
            present.append(student)
            present_ids.add(rec["student_id"])

    absent = list(students_col.find({"student_id": {"$nin": list(present_ids)}}))

    all_attendance = attendance_col.find().sort("timestamp", 1)
    years = set()
    months = {}
    for rec in all_attendance:
        ts = rec["timestamp"]
        if isinstance(ts, str):
            ts = datetime.datetime.fromisoformat(ts)
        years.add(ts.year)
        months.setdefault(ts.year, set()).add(ts.month)

    years = sorted(years)
    for y in months:
        months[y] = sorted(months[y])

    return render_template("attendance.html",
                           present=present,
                           absent=absent,
                           years=years,
                           months=months,
                           selected_year=selected_year,
                           selected_month=selected_month,
                           selected_day=selected_day)

@app.route("/notifications")
def notifications():
    if not session.get("admin"):
        return redirect(url_for("login"))

    today = datetime.datetime.now().date()
    absents = []
    for student in students_col.find():
        total_absences = attendance_col.count_documents({
            "student_id": student["student_id"],
            "timestamp": {"$gte": today}
        })
        if total_absences < 1:
            streak = 0
            for i in range(3):
                day = today - datetime.timedelta(days=i)
                count = attendance_col.count_documents({
                    "student_id": student["student_id"],
                    "timestamp": {"$gte": datetime.datetime.combine(day, datetime.time.min),
                                  "$lt": datetime.datetime.combine(day, datetime.time.max)}
                })
                if count == 0:
                    streak += 1
            if streak >= 3:
                absents.append(student)

    return render_template("notifications.html", absents=absents)

@app.route("/registered")
def registered():
    if not session.get("admin"):
        return redirect(url_for("login"))
    students = list(students_col.find().sort("timestamp", -1))
    logger.info("Fetched registered staffs")
    return render_template("registered.html", students=students)

# ---------------------------
if __name__ == "__main__":
    try:
        logger.info("Starting Flask app")
        logger.info("Webpage: http://127.0.0.1:5000")
        app.run(debug=True)
    finally:
        if cap is not None:
            cap.release()
            logger.info("Camera stopped on shutdown")