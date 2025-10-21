from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
# import datetime
import os
import logging
import json
import urllib.request
# from datetime import timedelta
import bcrypt
import re
import calendar
import smtplib
from email.message import EmailMessage
import time
import requests
# from datetime import datetime
import datetime
from datetime import timedelta
import time
import threading
# import requests
# from datetime import datetime
# import time

# esp32_ip = "http://192.168.4.50/time"

# ---------------------------
# Config
# ---------------------------
THRESHOLD = 0.7
DEVICE_NAME = "Device-01"
ADMIN_USER = "admin"
ADMIN_PASS = "1234"
REGISTERED_FILE = "registered.json"
ATTENDANCE_FILE = "attendance.json"
FIRED_FILE = "fired.json"
# Use data/ directory for JSON storage
REGISTERED_FILE = os.path.join('data', 'registered.json')
ATTENDANCE_FILE = os.path.join('data', 'attendance.json')
FIRED_FILE = os.path.join('data', 'fired.json')
ADMINS_DEPARTMENT = "Admin"
IP_CAMERA_URL = "http://192.168.4.50/stream"  # ESP32-CAM OV2640 MJPEG stream for attendance
esp32_ip = "http://192.168.4.50/time"

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
            if file_path == FIRED_FILE and data and isinstance(data[0], str):
                logger.warning(f"Converting legacy {file_path} format")
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
                save_data(file_path, new_data)
                data = new_data
            if parse_ts:
                for doc in data:
                    if "timestamp" in doc:
                        try:
                            doc["timestamp"] = datetime.datetime.fromisoformat(doc["timestamp"])
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Invalid timestamp in {file_path}: {doc.get('timestamp', 'None')}, error: {e}")
                            doc["timestamp"] = datetime.datetime.min
                    if "fired_timestamp" in doc:
                        try:
                            doc["fired_timestamp"] = datetime.datetime.fromisoformat(doc["fired_timestamp"])
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Invalid fired_timestamp in {file_path}: {doc.get('fired_timestamp', 'None')}, error: {e}")
                            doc["fired_timestamp"] = datetime.datetime.min
            return data
        return []
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []

def save_data(file_path, data):
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, cls=DateTimeEncoder, indent=2)
        logger.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save {file_path}: {e}")
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
                if embedding and len(embedding) == 512:
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
                        logger.warning(f"Invalid embedding shape for {doc['student_id']}")
                else:
                    logger.warning(f"Missing/invalid embedding for {doc['student_id']}")
        logger.info(f"Loaded {len(known_embeddings)} valid embeddings")
    except Exception as e:
        logger.error(f"Failed to load faces: {e}")
    return known_embeddings

def log_attendance(student_id, name, device=DEVICE_NAME, admin_name=ADMIN_USER):
    try:
        now = datetime.datetime.now()
        attendance = load_data(ATTENDANCE_FILE, parse_ts=True)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if any(rec["student_id"] == student_id and rec["timestamp"] >= today for rec in attendance):
            logger.info(f"Attendance already logged for {name} (ID: {student_id}) today")
            return False
        record = {
            "student_id": student_id,
            "name": name,
            "timestamp": now,
            "device": device,
            "admin_name": admin_name
        }
        attendance.insert(0, record)  # Insert at the beginning for newest first
        save_data(ATTENDANCE_FILE, attendance)
        logger.info(f"Attendance logged for {name} (ID: {student_id}) at {now.isoformat()}")
        return True
    except Exception as e:
        logger.error(f"Attendance logging failed for {name} (ID: {student_id}): {e}")
        return False

def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one number"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
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
is_registration_mode = False
last_unknown_face = None
last_unknown_frame = None

# ---------------------------
# Video Stream
# ---------------------------
def gen_frames_attendance():
    global current_name, known_embeddings, is_registration_mode, last_unknown_face, last_unknown_frame
    while True:
        try:
            with urllib.request.urlopen(IP_CAMERA_URL, timeout=5) as stream:
                bytes_data = b''
                while True:
                    bytes_data += stream.read(1024)
                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            logger.error("Failed to decode frame from ESP32")
                            continue

                        faces = app_face.get(frame)
                        logger.debug(f"Detected {len(faces)} faces in ESP32 frame")

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
                            if face.det_score < 0.7:
                                logger.debug(f"Low detection score: {face.det_score:.2f}")
                                current_name = "Poor Quality"
                                last_unknown_face = None
                                last_unknown_frame = None
                            else:
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
                                            # Use default admin_name since no request context
                                            if log_attendance(sid, data["name"], DEVICE_NAME, ADMIN_USER):
                                                logger.info(f"Successfully logged attendance for {data['name']} (ID: {sid})")
                                            else:
                                                logger.debug(f"Attendance not logged for {data['name']} (ID: {sid}) - already logged or failed")
                                        break
                                if not matched:
                                    current_name = "Unknown"
                                    last_unknown_face = face
                                    last_unknown_frame = frame.copy()
                                    logger.debug(f"Unknown face detected, det_score={face.det_score:.4f}")

                        for face in faces:
                            x1, y1, x2, y2 = face.bbox.astype(int)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, current_name if len(faces) == 1 else "Multiple", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        ret, buffer = cv2.imencode('.jpg', frame)
                        if not ret:
                            logger.error("Failed to encode frame")
                            continue
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        break
        except Exception as e:
            logger.error(f"Error fetching frame from ESP32 camera: {e}")
            time.sleep(1)

def gen_frames_registration():
    global current_name, known_embeddings, is_registration_mode, last_unknown_face, last_unknown_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open PC webcam")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame from PC webcam")
                time.sleep(0.1)
                continue

            faces = app_face.get(frame)
            logger.debug(f"Detected {len(faces)} faces in PC webcam")

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
                if face.det_score < 0.7:
                    logger.debug(f"Low detection score: {face.det_score:.2f}")
                    current_name = "Poor Quality"
                    last_unknown_face = None
                    last_unknown_frame = None
                else:
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
                            break
                    if not matched:
                        current_name = "Unknown"
                        last_unknown_face = face
                        last_unknown_frame = frame.copy()
                        logger.debug(f"Unknown face detected, det_score={face.det_score:.4f}")

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, current_name if len(faces) == 1 else "Multiple", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.error("Failed to encode frame")
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        logger.error(f"Error in PC webcam stream: {e}")
    finally:
        cap.release()

# ---------------------------
# Routes
# ---------------------------
@app.route("/video_feed")
def video_feed():
    global is_registration_mode
    is_registration_mode = False
    logger.info("Starting ESP32 video feed for attendance")
    return Response(gen_frames_attendance(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/register_video_feed")
def register_video_feed():
    global is_registration_mode
    is_registration_mode = True
    logger.info("Starting PC webcam feed for registration")
    return Response(gen_frames_registration(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    logger.info("Camera stop not required for streaming")
    return jsonify({"status": "success", "message": "Camera stop not required"})

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
        session["admin_name"] = "System Admin"  # Use a meaningful name instead of "admin"
        logger.info(f"Admin {username} logged in")
        return redirect(url_for("home"))

    registered = load_data(REGISTERED_FILE)
    admin = next((doc for doc in registered if doc.get("department") == ADMINS_DEPARTMENT and
                  (doc.get("student_id") == username or doc.get("email") == username)), None)

    if admin and "password" in admin and admin["password"]:
        try:
            if bcrypt.checkpw(password.encode('utf-8'), admin["password"].encode('utf-8')):
                session["admin"] = admin["student_id"]  # Store student_id instead of name
                session["admin_name"] = admin["name"]  # Store display name
                logger.info(f"Admin {admin['name']} logged in")
                return redirect(url_for("home"))
        except Exception as e:
            logger.error(f"Password verification failed for {username}: {e}")

    logger.warning(f"Invalid login attempt for {username}")
    return render_template("login.html", error="Invalid username or password")

@app.route("/home")
def home():
    if not session.get("admin"):
        logger.warning("Unauthorized access to home page")
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

    logger.info(f"Home page loaded with stats: {stats}")
    return render_template("home.html", stats=stats)

@app.route("/register_page")
def register_page():
    if not session.get("admin"):
        logger.warning("Unauthorized access to register page")
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/logout")
def logout():
    admin = session.get("admin", "Unknown")
    session.pop("admin", None)
    logger.info(f"Admin {admin} logged out")
    return redirect(url_for("login"))

@app.route("/admin_register_page")
def admin_register_page():
    if not session.get("admin"):
        logger.warning("Unauthorized access to admin register page")
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
        logger.error("Missing required fields for registration")
        return jsonify({"status": "error", "message": "All fields are required"}), 400

    if not validate_email(email):
        logger.error(f"Invalid email format: {email}")
        return jsonify({"status": "error", "message": "Invalid email address"}), 400

    if last_unknown_face is None or last_unknown_frame is None:
        logger.error("No unknown face detected for registration")
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
        logger.error(f"Image save failed for {student_id}: {e}")
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

    phone_regex = r'^\+251\d{9}$'
    if not re.match(phone_regex, phone):
        logger.error(f"Invalid phone format: {phone}")
        return jsonify({"status": "error", "message": "Phone number must be in format +2519xxxxxxxx"}), 400

    is_valid, password_error = validate_password(password)
    if not is_valid:
        logger.error(f"Password validation failed: {password_error}")
        return jsonify({"status": "error", "message": password_error}), 400

    if last_unknown_face is None or last_unknown_frame is None:
        logger.error("No unknown face detected for admin registration")
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

@app.route("/send_alert", methods=["POST"])
def send_alert():
    email = request.form["email"]
    name = request.form["name"]

    try:
        sender_email = "tarikushemsu3@gmail.com"
        sender_password = "kztm conz kqxe tctz"  # App password
        subject = "Attendance Alert"

        msg = EmailMessage()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = subject
        msg.set_content(f"""
Dear {name},

Our system detected that you have been absent for 3 or more consecutive days.
Please contact your department immediately.

Regards,
Attendance Admin
""")

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        logger.info(f"Alert email sent to {name} ({email})")
        return redirect(url_for("notifications"))
    except Exception as e:
        logger.error(f"Failed to send email to {name} ({email}): {e}")
        return redirect(url_for("notifications"))

@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if not session.get("admin"):
        logger.warning("Unauthorized access to attendance page")
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
    late = []
    present_ids = set()
    late_threshold = datetime.datetime(selected_year, selected_month, selected_day, 9, 30, 0)

    registered = load_data(REGISTERED_FILE, parse_ts=True)

    for rec in attendance_records:
        student = next((doc for doc in registered if doc["student_id"] == rec["student_id"]), None)
        if student:
            student = student.copy()
            student["timestamp"] = rec["timestamp"]
            student["admin_name"] = rec.get("admin_name", ADMIN_USER)
            if rec["timestamp"] > late_threshold:
                late.append(student)
            else:
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

    logger.info(f"Attendance page loaded for {selected_year}-{selected_month}-{selected_day}: {len(present)} present, {len(late)} late, {len(absent)} absent")
    return render_template("attendance.html",
                           present=present,
                           late=late,
                           absent=absent,
                           fired=fired,
                           years=years,
                           months=months,
                           selected_year=selected_year,
                           selected_month=selected_month,
                           selected_day=selected_day)

@app.route("/attendance_data", methods=["GET"])
def attendance_data():
    year = request.args.get("year", type=int)
    month = request.args.get("month", type=int)
    day = request.args.get("day", type=int)

    if not all([year, month, day]):
        today = datetime.datetime.now().date()
        year = year or today.year
        month = month or today.month
        day = day or today.day

    start = datetime.datetime(year, month, day, 0, 0, 0)
    end = datetime.datetime(year, month, day, 23, 59, 59)

    fired = load_data(FIRED_FILE, parse_ts=True)
    fired_ids = set(doc["student_id"] for doc in fired)
    attendance_data = load_data(ATTENDANCE_FILE, parse_ts=True)
    attendance_records = [rec for rec in attendance_data if start <= rec["timestamp"] <= end and rec["student_id"] not in fired_ids]

    present = []
    late = []
    present_ids = set()
    late_threshold = datetime.datetime(year, month, day, 9, 30, 0)

    registered = load_data(REGISTERED_FILE, parse_ts=True)

    for rec in attendance_records:
        student = next((doc for doc in registered if doc["student_id"] == rec["student_id"]), None)
        if student:
            student = student.copy()
            student["timestamp"] = rec["timestamp"].isoformat()
            student["admin_name"] = rec.get("admin_name", ADMIN_USER)
            if rec["timestamp"] > late_threshold:
                late.append(student)
            else:
                present.append(student)
            present_ids.add(rec["student_id"])

    absent = [doc for doc in registered if doc["student_id"] not in present_ids and doc["student_id"] not in fired_ids]

    response = {
        "present": present,
        "late": late,
        "absent": absent,
        "stats": {
            "total_registered": len([doc for doc in registered if doc["student_id"] not in fired_ids]),
            "present_today": len(present),
            "late_today": len(late),
            "absent_today": len(absent)
        }
    }
    logger.info(f"Attendance data fetched for {year}-{month}-{day}: {response['stats']}")
    return jsonify(response)

@app.route("/notifications")
def notifications():
    if not session.get("admin"):
        logger.warning("Unauthorized access to notifications page")
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

    logger.info(f"Notifications page loaded with {len(absents)} absent staff")
    return render_template("notifications.html", absents=absents)

@app.route("/registered")
def registered():
    if not session.get("admin"):
        logger.warning("Unauthorized access to registered page")
        return redirect(url_for("login"))
    registered = load_data(REGISTERED_FILE, parse_ts=True)
    fired = load_data(FIRED_FILE, parse_ts=True)
    fired_ids = set(doc["student_id"] for doc in fired)
    students = sorted(
        [doc for doc in registered if doc["student_id"] not in fired_ids], 
        key=lambda x: x.get("timestamp", datetime.datetime.min), 
        reverse=True
    )
    fired = sorted(
        fired,
        key=lambda x: x.get("fired_timestamp", datetime.datetime.min),
        reverse=True
    )
    logger.info(f"Registered page loaded with {len(students)} staff, excluded {len(fired_ids)} fired")
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
            logger.info(f"Detected {len(faces)} faces in uploaded frame")
            return jsonify({"status": "error", "message": f"Detected {len(faces)} faces, expected 1"}), 400
        embedding = faces[0].embedding.reshape(1, -1)
        for student_id, data in known_embeddings.items():
            similarity = cosine_similarity(embedding, data["embedding"].reshape(1, -1))[0][0]
            logger.debug(f"Upload frame comparison with {student_id} ({data['name']}): similarity={similarity:.4f}")
            if similarity > THRESHOLD:
                today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                attendance_data = load_data(ATTENDANCE_FILE, parse_ts=True)
                if not any(rec["student_id"] == student_id and rec["timestamp"] >= today for rec in attendance_data):
                    if log_attendance(student_id, data["name"], admin_name=session.get("admin", ADMIN_USER)):
                        logger.info(f"Attendance logged via upload for {data['name']} (ID: {student_id})")
                        return jsonify({"status": "success", "message": f"Attendance logged for {data['name']}"})
                    else:
                        logger.info(f"Attendance not logged for {data['name']} (ID: {student_id}) - already logged or failed")
                        return jsonify({"status": "success", "message": f"Attendance already logged for {data['name']} today"})
                else:
                    logger.info(f"Attendance already logged for {student_id} today")
                    return jsonify({"status": "success", "message": f"Attendance already logged for {data['name']} today"})
        logger.info("No matching face found in uploaded frame")
        return jsonify({"status": "error", "message": "No matching face found"})
    except Exception as e:
        logger.error(f"Error processing uploaded frame: {e}")
        return jsonify({"status": "error", "message": f"Error processing frame: {str(e)}"}), 500

@app.route("/fire_staff", methods=["POST"])
def fire_staff():
    global known_embeddings
    data = request.json
    student_id = data.get("student_id")
    password = data.get("password")

    if not student_id or not password:
        logger.error("Missing student_id or password for fire staff")
        return jsonify({"status": "error", "message": "Student ID and password are required"}), 400

    registered = load_data(REGISTERED_FILE)
    admin = None
    if password == ADMIN_PASS:
        admin = {"name": ADMIN_USER}
    else:
        admin = next((doc for doc in registered if doc.get("department") == ADMINS_DEPARTMENT and 
                      bcrypt.checkpw(password.encode('utf-8'), doc.get("password", "").encode('utf-8'))), None)
    
    if not admin:
        logger.error("Invalid admin password for fire staff")
        return jsonify({"status": "error", "message": "Invalid admin password"}), 401

    staff = next((doc for doc in registered if doc["student_id"] == student_id), None)
    if not staff:
        logger.error(f"Staff with ID {student_id} not found")
        return jsonify({"status": "error", "message": f"Staff with ID {student_id} not found"}), 404

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

    try:
        registered = [doc for doc in registered if doc["student_id"] != student_id]
        save_data(REGISTERED_FILE, registered)
        logger.info(f"Removed {student_id} from registered.json")
    except Exception as e:
        logger.error(f"Failed to update registered.json for {student_id}: {e}")
        return jsonify({"status": "error", "message": "Failed to update registered list due to server error"}), 500

    known_embeddings = load_known_faces()

    image_path = f'static/images/{student_id}.jpg'
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
            logger.info(f"Removed image for {student_id}")
    except Exception as e:
        logger.warning(f"Failed to remove image for {student_id}: {e}")

    logger.info(f"Staff {staff['name']} (ID: {student_id}) fired successfully")
    return jsonify({"status": "success", "message": f"Staff {staff['name']} fired successfully"})

@app.route("/delete_staff", methods=["POST"])
def delete_staff():
    global known_embeddings
    data = request.json
    student_id = data.get("student_id")
    password = data.get("password")

    if not student_id or not password:
        logger.error("Missing student_id or password for delete staff")
        return jsonify({"status": "error", "message": "Student ID and password are required"}), 400

    registered = load_data(REGISTERED_FILE)
    admin = None
    if password == ADMIN_PASS:
        admin = {"name": ADMIN_USER}
    else:
        admin = next((doc for doc in registered if doc.get("department") == ADMINS_DEPARTMENT and 
                      bcrypt.checkpw(password.encode('utf-8'), doc.get("password", "").encode('utf-8'))), None)
    
    if not admin:
        logger.error("Invalid admin password for delete staff")
        return jsonify({"status": "error", "message": "Invalid admin password"}), 401

    fired = load_data(FIRED_FILE)
    staff = next((doc for doc in fired if doc["student_id"] == student_id), None)
    if not staff:
        logger.error(f"Fired staff with ID {student_id} not found")
        return jsonify({"status": "error", "message": f"Fired staff with ID {student_id} not found"}), 404

    try:
        fired = [doc for doc in fired if doc["student_id"] != student_id]
        save_data(FIRED_FILE, fired)
        logger.info(f"Removed {student_id} from fired.json")
    except Exception as e:
        logger.error(f"Failed to remove from fired.json for {student_id}: {e}")
        return jsonify({"status": "error", "message": "Failed to delete staff from fired list due to server error"}), 500

    image_path = f'static/images/{student_id}.jpg'
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
            logger.info(f"Removed image for {student_id}")
    except Exception as e:
        logger.warning(f"Failed to remove image for {student_id}: {e}")

    logger.info(f"Staff {staff['name']} (ID: {student_id}) permanently deleted")
    return jsonify({"status": "success", "message": f"Staff {staff['name']} permanently deleted from system"})


# ---------------------------
if __name__ == "__main__":
    try:
        logger.info("Starting Flask app")
        logger.info("Webpage: http://127.0.0.1:5000")
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Flask app failed to start: {e}")