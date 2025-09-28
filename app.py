#----------------------------
#libraries
#----------------------------                                                                                                                                                                           ``
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import os
import logging
import json
from datetime import timedelta
import bcrypt
import re

# ---------------------------
# Config
# ---------------------------
THRESHOLD = 0.6  # Adjusted for stricter matching
DEVICE_NAME = "Device-01"
ADMIN_USER = "admin"
ADMIN_PASS = "1234"
REGISTERED_FILE = "registered.json"
ATTENDANCE_FILE = "attendance.json"
FIRED_FILE = "fired.json"
ADMINS_DEPARTMENT = "Admin"

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ---------------------------
# JSON Helpers
# ---------------------------
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime.datetime):
            return o.isoformat()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

def load_data(file_path, parse_ts=False):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Handle legacy fired.json format
            if file_path == FIRED_FILE and data and isinstance(data[0], str):
                logger.warning(f"Converting legacy {file_path} format (strings) to dictionaries")
                new_data = []
                registered = load_data(REGISTERED_FILE)
                for student_id in data:
                    staff = next((doc for doc in registered if doc["student_id"] == student_id), None)
                    new_data.append({
                        "student_id": student_id,
                        "name": staff.get("name", f"Unknown ({student_id})") if staff else f"Unknown ({student_id})",
                        "email": staff.get("email", "") if staff else "",
                        "phone": staff.get("phone", "") if staff else "",
                        "department": staff.get("department", "") if staff else "",
                        "fired_by": "Unknown",
                        "fired_timestamp": datetime.datetime.now().isoformat()
                    })
                    if not staff:
                        logger.warning(f"No registered data found for fired staff ID {student_id}")
                save_data(file_path, new_data)
                data = new_data
            elif parse_ts:
                for doc in data:
                    if "timestamp" in doc and isinstance(doc["timestamp"], str):
                        try:
                            doc["timestamp"] = datetime.datetime.fromisoformat(doc["timestamp"])
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Invalid timestamp format in {file_path}: {doc.get('timestamp', 'None')}, error: {e}")
                            doc["timestamp"] = datetime.datetime.min
                    if "fired_timestamp" in doc and isinstance(doc["fired_timestamp"], str):
                        try:
                            doc["fired_timestamp"] = datetime.datetime.fromisoformat(doc["fired_timestamp"])
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Invalid fired_timestamp format in {file_path}: {doc.get('fired_timestamp', 'None')}, error: {e}")
                            doc["fired_timestamp"] = datetime.datetime.min
            return data
        else:
            logger.info(f"File {file_path} not found, returning empty list")
            return []
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []

def save_data(file_path, data):
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, cls=DateTimeEncoder, indent=2)
    except Exception as e:
        logger.error(f"Failed to save {file_path}: {e}")

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
        fired = load_data(FIRED_FILE)
        fired_ids = set(doc["student_id"] for doc in fired)
        registered = load_data(REGISTERED_FILE)
        for doc in registered:
            if doc["student_id"] not in fired_ids:
                embedding = doc.get("embedding", [])
                if embedding and isinstance(embedding, list) and len(embedding) == 512:
                    try:
                        embedding_array = np.array(embedding)
                        if embedding_array.size == 512 and embedding_array.shape == (512,):
                            known_embeddings[doc["student_id"]] = {
                                "name": doc["name"],
                                "embedding": embedding_array,
                                "email": doc.get("email", ""),
                                "phone": doc.get("phone", ""),
                                "department": doc.get("department", "")
                            }
                        else:
                            logger.warning(f"Invalid embedding shape for {doc['student_id']}: {embedding_array.shape}")
                    except Exception as e:
                        logger.warning(f"Failed to process embedding for {doc['student_id']}: {e}")
                else:
                    logger.warning(f"Missing or invalid embedding for {doc['student_id']}: length={len(embedding)}")
        logger.info(f"Loaded {len(known_embeddings)} valid embeddings, excluded {len(fired_ids)} fired staff")
    except Exception as e:
        logger.error(f"Failed to load faces: {e}")
    return known_embeddings

def log_attendance(student_id, name, device=DEVICE_NAME):
    try:
        now = datetime.datetime.now()
        attendance = load_data(ATTENDANCE_FILE, parse_ts=True)
        attendance.append({
            "student_id": student_id,
            "name": name,
            "timestamp": now,
            "device": device,
            "admin_name": session.get("admin", ADMIN_USER)
        })
        save_data(ATTENDANCE_FILE, attendance)
        logger.info(f"Attendance logged for {name} (ID: {student_id})")
    except Exception as e:
        logger.error(f"Attendance logging failed for {name} (ID: {student_id}): {e}")

def get_cap():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Camera failed to open")
            raise RuntimeError("Could not open camera")
    return cap

def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    elif not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    elif not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    elif not re.search(r"[0-9]", password):
        return False, "Password must contain at least one number"
    elif not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    return True, ""

def validate_email(email):
    email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
    return bool(re.match(email_regex, email))

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
        logger.debug(f"Detected {len(faces)} faces")

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
                if data["embedding"].size == 0:
                    logger.warning(f"Empty embedding for {sid}")
                    continue
                sim = cosine_similarity(emb, data["embedding"].reshape(1, -1))[0][0]
                logger.debug(f"Comparing with {sid} ({data['name']}): similarity={sim:.4f}, det_score={face.det_score:.4f}")
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
                logger.debug(f"Unknown face detected, det_score={face.det_score:.4f}")
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
    if not username or not password:
        return render_template("login.html", error="Username and password are required")

    if username == ADMIN_USER and password == ADMIN_PASS:
        session["admin"] = username
        return redirect(url_for("home"))

    registered = load_data(REGISTERED_FILE)
    admin = next((doc for doc in registered if doc.get("department") == ADMINS_DEPARTMENT and 
                  (doc.get("student_id") == username or doc.get("email") == username)), None)
    
    if admin and "password" in admin and admin["password"]:
        try:
            if bcrypt.checkpw(password.encode('utf-8'), admin["password"].encode('utf-8')):
                session["admin"] = admin["name"]
                return redirect(url_for("home"))
        except Exception as e:
            logger.error(f"Password verification failed for {username}: {e}")
    
    return render_template("login.html", error="Invalid credentials")

@app.route("/home")
def home():
    if not session.get("admin"):
        return redirect(url_for("login"))

    registered = load_data(REGISTERED_FILE, parse_ts=True)
    fired = load_data(FIRED_FILE, parse_ts=True)
    attendance_data = load_data(ATTENDANCE_FILE, parse_ts=True)
    today = datetime.datetime.now().date()
    start = datetime.datetime.combine(today, datetime.time(0, 0, 0))
    end = datetime.datetime.combine(today, datetime.time(23, 59, 59))

    fired_ids = set(doc["student_id"] for doc in fired)
    total_registered = len([doc for doc in registered if doc["student_id"] not in fired_ids])
    
    present_ids = set(rec["student_id"] for rec in attendance_data 
                     if start <= rec["timestamp"] <= end and rec["student_id"] not in fired_ids)
    present_today = len(present_ids)
    
    absent_today = total_registered - present_today
    
    today_logins = [rec["timestamp"] for rec in attendance_data 
                    if start <= rec["timestamp"] <= end and rec["student_id"] not in fired_ids]
    avg_login_time = "N/A"
    if today_logins:
        avg_seconds = sum((t.hour * 3600 + t.minute * 60 + t.second) for t in today_logins) // len(today_logins)
        hours, remainder = divmod(avg_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        avg_login_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    total_fired = len(fired)
    
    notifications = 0
    for student in [doc for doc in registered if doc["student_id"] not in fired_ids]:
        streak = 0
        for i in range(3):
            day = today - timedelta(days=i)
            count = sum(1 for rec in attendance_data 
                       if rec["student_id"] == student["student_id"] and rec["timestamp"].date() == day)
            if count == 0:
                streak += 1
        if streak >= 3:
            notifications += 1

    stats = {
        "total_registered": total_registered,
        "present_today": present_today,
        "absent_today": absent_today,
        "avg_login_time": avg_login_time,
        "total_fired": total_fired,
        "notifications": notifications
    }

    return render_template("home.html", stats=stats)

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
    student_id = data.get("student_id")
    name = data.get("name")
    email = data.get("email")
    phone = data.get("phone")
    department = data.get("department")

    if not all([student_id, name, email, phone, department]):
        logger.error("Missing required fields in staff registration")
        return jsonify({"status": "error", "message": "All fields (Student ID, Name, Email, Phone, Department) are required"}), 400

    if not validate_email(email):
        logger.error(f"Invalid email format: {email}")
        return jsonify({"status": "error", "message": "Invalid email address"}), 400

    retry_count = 10
    while last_unknown_face is None and retry_count > 0:
        logger.info(f"Retrying face detection ({retry_count} left)")
        cap = get_cap()
        success, frame = cap.read()
        if success:
            faces = app_face.get(frame)
            if len(faces) == 1:
                emb = faces[0].embedding.reshape(1, -1)
                is_known = False
                for sid, known_data in known_embeddings.items():
                    if known_data["embedding"].size == 0:
                        logger.warning(f"Empty embedding for {sid}")
                        continue
                    sim = cosine_similarity(emb, known_data["embedding"].reshape(1, -1))[0][0]
                    logger.debug(f"Checking against {sid} ({known_data['name']}): similarity={sim:.4f}")
                    if sim > THRESHOLD:
                        is_known = True
                        logger.error(f"Face already registered as {known_data['name']} (ID: {sid})")
                        return jsonify({"status": "error", "message": f"Face already registered as {known_data['name']} (ID: {sid})"}), 400
                if not is_known:
                    last_unknown_face = faces[0]
                    last_unknown_frame = frame.copy()
                    logger.info(f"Face detected with det_score: {last_unknown_face.det_score:.4f}")
        retry_count -= 1
        import time
        time.sleep(1)

    if last_unknown_face is None or last_unknown_frame is None:
        logger.error("No unknown face detected after retries")
        return jsonify({"status": "error", "message": "No unknown face detected. Ensure a single face is visible and try again."}), 400

    if last_unknown_face.det_score < 0.7:
        logger.error(f"Poor image quality: det_score={last_unknown_face.det_score:.2f}")
        return jsonify({"status": "error", "message": f"Poor image quality (det_score: {last_unknown_face.det_score:.2f}). Ensure good lighting and clear face."}), 400

    registered = load_data(REGISTERED_FILE)
    if next((doc for doc in registered if doc["student_id"] == student_id), None):
        logger.error(f"Student ID {student_id} already exists")
        return jsonify({"status": "error", "message": f"Student ID {student_id} already registered"}), 400

    if next((doc for doc in registered if doc.get("email") == email), None):
        logger.error(f"Email {email} already exists")
        return jsonify({"status": "error", "message": f"Email {email} already registered"}), 400

    embedding = last_unknown_face.embedding.tolist()

    try:
        os.makedirs('static/images', exist_ok=True)
        image_path = f'static/images/{student_id}.jpg'
        cv2.imwrite(image_path, last_unknown_frame)
        logger.info(f"Image saved for {student_id}")
    except Exception as e:
        logger.error(f"Image save failed: {e}")
        return jsonify({"status": "error", "message": "Failed to save image due to server error"}), 500

    try:
        doc = {
            "student_id": student_id,
            "name": name,
            "embedding": embedding,
            "email": email,
            "phone": phone,
            "department": department,
            "timestamp": datetime.datetime.now(),
            "admin_name": session.get("admin", ADMIN_USER)
        }
        registered.append(doc)
        save_data(REGISTERED_FILE, registered)
        logger.info(f"Registered staff {name} (ID: {student_id})")
    except Exception as e:
        logger.error(f"Save failed for {student_id}: {e}")
        return jsonify({"status": "error", "message": "Failed to save due to server error"}), 500

    known_embeddings = load_known_faces()
    last_unknown_face = None
    last_unknown_frame = None
    return jsonify({"status": "success", "message": f"Staff {name} registered successfully!"})

@app.route("/admin_register", methods=["POST"])
def admin_register():
    global known_embeddings, last_unknown_face, last_unknown_frame
    data = request.json
    logger.info(f"Received admin register payload: {data}")
    admin_id = data.get("admin_id")
    name = data.get("name")
    email = data.get("email")
    phone = data.get("phone")
    password = data.get("password")

    if not all([admin_id, name, email, phone, password]):
        logger.error("Missing required fields in admin registration")
        return jsonify({"status": "error", "message": "All fields (Admin ID, Name, Email, Phone, Password) are required"}), 400

    if not validate_email(email):
        logger.error(f"Invalid email format: {email}")
        return jsonify({"status": "error", "message": "Invalid email address"}), 400

    # Validate phone (Ethiopian format: +251 followed by 9 digits)
    phone_regex = r'^\+251\d{9}$'
    if not re.match(phone_regex, phone):
        logger.error(f"Invalid phone format: {phone}")
        return jsonify({"status": "error", "message": "Phone number must be in format +2519xxxxxxxx"}), 400

    is_valid, password_error = validate_password(password)
    if not is_valid:
        logger.error(f"Password validation failed: {password_error}")
        return jsonify({"status": "error", "message": password_error}), 400

    retry_count = 10
    while last_unknown_face is None and retry_count > 0:
        logger.info(f"Retrying face detection ({retry_count} left)")
        cap = get_cap()
        success, frame = cap.read()
        if success:
            faces = app_face.get(frame)
            if len(faces) == 1:
                emb = faces[0].embedding.reshape(1, -1)
                is_known = False
                for sid, known_data in known_embeddings.items():
                    if known_data["embedding"].size == 0:
                        logger.warning(f"Empty embedding for {sid}")
                        continue
                    sim = cosine_similarity(emb, known_data["embedding"].reshape(1, -1))[0][0]
                    logger.debug(f"Checking against {sid} ({known_data['name']}): similarity={sim:.4f}")
                    if sim > THRESHOLD:
                        is_known = True
                        logger.error(f"Face already registered as {known_data['name']} (ID: {sid})")
                        return jsonify({"status": "error", "message": f"Face already registered as {known_data['name']} (ID: {sid})"}), 400
                if not is_known:
                    last_unknown_face = faces[0]
                    last_unknown_frame = frame.copy()
                    logger.info(f"Face detected with det_score: {last_unknown_face.det_score:.4f}")
        retry_count -= 1
        import time
        time.sleep(1)

    if last_unknown_face is None or last_unknown_frame is None:
        logger.error("No unknown face detected after retries")
        return jsonify({"status": "error", "message": "No unknown face detected. Ensure a single face is visible and try again."}), 400

    if last_unknown_face.det_score < 0.7:
        logger.error(f"Poor image quality: det_score={last_unknown_face.det_score:.2f}")
        return jsonify({"status": "error", "message": f"Poor image quality (det_score: {last_unknown_face.det_score:.2f}). Ensure good lighting and clear face."}), 400

    registered = load_data(REGISTERED_FILE)
    if next((doc for doc in registered if doc["student_id"] == admin_id), None):
        logger.error(f"Admin ID {admin_id} already exists")
        return jsonify({"status": "error", "message": f"Admin ID {admin_id} already registered"}), 400

    if next((doc for doc in registered if doc.get("email") == email), None):
        logger.error(f"Email {email} already exists")
        return jsonify({"status": "error", "message": f"Email {email} already registered"}), 400

    if next((doc for doc in registered if doc.get("name") == name and doc.get("department") == ADMINS_DEPARTMENT), None):
        logger.error(f"Admin name {name} already exists")
        return jsonify({"status": "error", "message": f"Admin name {name} already registered"}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

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
        doc = {
            "student_id": admin_id,
            "name": name,
            "embedding": embedding,
            "email": email,
            "phone": phone,
            "department": ADMINS_DEPARTMENT,
            "timestamp": datetime.datetime.now(),
            "admin_name": session.get("admin", ADMIN_USER),
            "password": hashed_password
        }
        registered.append(doc)
        save_data(REGISTERED_FILE, registered)
        logger.info(f"Registered admin {name} (ID: {admin_id})")
    except Exception as e:
        logger.error(f"Save failed for {admin_id}: {e}")
        return jsonify({"status": "error", "message": "Failed to save due to server error"}), 500

    known_embeddings = load_known_faces()
    last_unknown_face = None
    last_unknown_frame = None
    return jsonify({"status": "success", "message": f"Admin {name} registered successfully!"})

@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if not session.get("admin"):
        return redirect(url_for("login"))

    today = datetime.datetime.now().date()
    selected_year = request.form.get("year", today.year, type=int)
    selected_month = request.form.get("month", today.month, type=int)
    selected_day = request.form.get("day", today.day, type=int)

    start = datetime.datetime(selected_year, selected_month, selected_day, 0, 0, 0)
    end = datetime.datetime(selected_year, selected_month, selected_day, 23, 59, 59)

    fired = load_data(FIRED_FILE, parse_ts=True)
    fired_ids = set(doc["student_id"] for doc in fired)
    attendance_data = load_data(ATTENDANCE_FILE, parse_ts=True)
    attendance_records = [rec for rec in attendance_data if start <= rec["timestamp"] <= end and rec["student_id"] not in fired_ids]

    present = []
    present_ids = set()

    registered = load_data(REGISTERED_FILE, parse_ts=True)

    for rec in attendance_records:
        student = next((doc for doc in registered if doc["student_id"] == rec["student_id"]), None)
        if student:
            student = student.copy()
            student["timestamp"] = rec["timestamp"]
            student["admin_name"] = rec.get("admin_name", ADMIN_USER)
            present.append(student)
            present_ids.add(rec["student_id"])

    absent = [doc for doc in registered if doc["student_id"] not in present_ids and doc["student_id"] not in fired_ids]

    all_attendance = sorted(attendance_data, key=lambda x: x["timestamp"])
    years = set()
    months = {}
    for rec in all_attendance:
        ts = rec["timestamp"]
        years.add(ts.year)
        months.setdefault(ts.year, set()).add(ts.month)

    years = sorted(years)
    for y in months:
        months[y] = sorted(months[y])

    return render_template("attendance.html",
                           present=present,
                           absent=absent,
                           fired=fired,
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
    fired = load_data(FIRED_FILE)
    fired_ids = set(doc["student_id"] for doc in fired)
    registered = load_data(REGISTERED_FILE)
    attendance_data = load_data(ATTENDANCE_FILE, parse_ts=True)
    for student in [doc for doc in registered if doc["student_id"] not in fired_ids]:
        total_absences_today = sum(1 for rec in attendance_data if rec["student_id"] == student["student_id"] and rec["timestamp"].date() >= today)
        if total_absences_today < 1:
            streak = 0
            for i in range(3):
                day = today - timedelta(days=i)
                count = sum(1 for rec in attendance_data if rec["student_id"] == student["student_id"] and rec["timestamp"].date() == day)
                if count == 0:
                    streak += 1
            if streak >= 3:
                absents.append(student)

    return render_template("notifications.html", absents=absents)

@app.route("/registered")
def registered():
    if not session.get("admin"):
        return redirect(url_for("login"))
    registered = load_data(REGISTERED_FILE, parse_ts=True)
    fired = load_data(FIRED_FILE, parse_ts=True)
    fired_ids = set(doc["student_id"] for doc in fired)
    students = sorted([doc for doc in registered if doc["student_id"] not in fired_ids], 
                      key=lambda x: x.get("timestamp", datetime.datetime.min), reverse=True)
    logger.info(f"Fetched {len(students)} registered staff, excluded {len(fired_ids)} fired staff")
    return render_template("registered.html", students=students, fired=fired)

@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    global known_embeddings
    if 'photo' not in request.files:
        logger.error("No photo part in request")
        return jsonify({"status": "error", "message": "No photo part"}), 400
    photo = request.files['photo']
    try:
        img_array = np.frombuffer(photo.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image")
            return jsonify({"status": "error", "message": "Failed to decode image"}), 400
        faces = app_face.get(img)
        if len(faces) != 1:
            logger.info(f"Detected {len(faces)} faces")
            return jsonify({"status": "error", "message": f"Detected {len(faces)} faces, expected 1"}), 400
        embedding = faces[0].embedding.reshape(1, -1)
        for student_id, data in known_embeddings.items():
            similarity = cosine_similarity(embedding, data["embedding"].reshape(1, -1))[0][0]
            logger.debug(f"Upload frame comparison with {student_id} ({data['name']}): similarity={similarity:.4f}")
            if similarity > THRESHOLD:
                today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                attendance_data = load_data(ATTENDANCE_FILE, parse_ts=True)
                if not any(rec["student_id"] == student_id and rec["timestamp"] >= today for rec in attendance_data):
                    log_attendance(student_id, data["name"])
                    return jsonify({"status": "success", "message": f"Attendance logged for {data['name']}"})
                else:
                    logger.info(f"Attendance already logged for {student_id} today")
                    return jsonify({"status": "success", "message": f"Attendance already logged for {data['name']} today"})
        logger.info("No matching face found")
        return jsonify({"status": "error", "message": "No matching face found"})
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return jsonify({"status": "error", "message": f"Error processing frame: {str(e)}"}), 500

@app.route("/fire_staff", methods=["POST"])
def fire_staff():
    global known_embeddings
    data = request.json
    student_id = data.get("student_id")
    password = data.get("password")

    if not student_id or not password:
        logger.error("Missing student_id or password")
        return jsonify({"status": "error", "message": "Student ID and password are required"}), 400

    # Verify admin password
    registered = load_data(REGISTERED_FILE)
    admin = None
    if password == ADMIN_PASS:
        admin = {"name": ADMIN_USER}
    else:
        admin = next((doc for doc in registered if doc.get("department") == ADMINS_DEPARTMENT and 
                      bcrypt.checkpw(password.encode('utf-8'), doc.get("password", "").encode('utf-8'))), None)
    
    if not admin:
        logger.error("Invalid admin password")
        return jsonify({"status": "error", "message": "Invalid admin password"}), 401

    # Check if student_id exists in registered
    staff = next((doc for doc in registered if doc["student_id"] == student_id), None)
    if not staff:
        logger.error(f"Staff with ID {student_id} not found")
        return jsonify({"status": "error", "message": f"Staff with ID {student_id} not found"}), 404

    # Add to fired.json with details
    try:
        fired = load_data(FIRED_FILE)
        if not any(doc["student_id"] == student_id for doc in fired):
            fired.append({
                "student_id": student_id,
                "name": staff["name"],
                "email": staff.get("email", ""),
                "phone": staff.get("phone", ""),
                "department": staff.get("department", ""),
                "fired_by": admin["name"],
                "fired_timestamp": datetime.datetime.now()
            })
            save_data(FIRED_FILE, fired)
            logger.info(f"Added {student_id} to fired.json")
        else:
            logger.info(f"Staff {student_id} already in fired.json")
    except Exception as e:
        logger.error(f"Failed to update fired.json for {student_id}: {e}")
        return jsonify({"status": "error", "message": "Failed to update fired list due to server error"}), 500

    # Remove from registered.json
    try:
        registered = [doc for doc in registered if doc["student_id"] != student_id]
        save_data(REGISTERED_FILE, registered)
        logger.info(f"Removed {student_id} from registered.json")
    except Exception as e:
        logger.error(f"Failed to update registered.json for {student_id}: {e}")
        return jsonify({"status": "error", "message": "Failed to update registered list due to server error"}), 500

    # Update known_embeddings
    known_embeddings = load_known_faces()

    # Remove associated image if it exists
    image_path = f'static/images/{student_id}.jpg'
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
            logger.info(f"Removed image for {student_id}")
    except Exception as e:
        logger.warning(f"Failed to remove image for {student_id}: {e}")

    return jsonify({"status": "success", "message": f"Staff {staff['name']} fired successfully"})

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