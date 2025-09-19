from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
from pymongo import MongoClient
import cv2, numpy as np, datetime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Config
# ---------------------------
MONGO_URI = "mongodb+srv://beamlakbekele197_db_user:m5pG8eSXLFxM0Y9e@cluster0.mrqsfdq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "attendance_system"
THRESHOLD = 0.55
DEVICE_NAME = "Device-01"
ADMIN_USER = "admin"
ADMIN_PASS = "1234"   # ⚠️ You should hash in production!

# ---------------------------
# MongoDB
# ---------------------------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
students_col = db["students"]
attendance_col = db["attendance"]

# ---------------------------
# FaceAnalysis
# ---------------------------
app_face = FaceAnalysis(providers=['CPUExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))

# ---------------------------
# Flask
# ---------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------------------------
# Load known embeddings
# ---------------------------
def load_known_faces():
    known_embeddings = {}
    for doc in students_col.find():
        known_embeddings[doc["student_id"]] = {
            "name": doc["name"],
            "embedding": np.array(doc["embedding"])
        }
    return known_embeddings

known_embeddings = load_known_faces()
current_name = "No Face"

def log_attendance(student_id, name, device=DEVICE_NAME):
    today = datetime.date.today().isoformat()
    if not attendance_col.find_one({"student_id": student_id, "date": today}):
        attendance_col.insert_one({
            "student_id": student_id,
            "name": name,
            "date": today,
            "timestamp": datetime.datetime.now().isoformat(),
            "device": device
        })

# ---------------------------
# Video Stream
# ---------------------------
cap = cv2.VideoCapture(0)

def gen_frames():
    global current_name, known_embeddings
    while True:
        success, frame = cap.read()
        if not success:
            break

        faces = app_face.get(frame)
        current_name = "Unknown"

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = face.embedding.reshape(1, -1)

            for sid, data in known_embeddings.items():
                sim = cosine_similarity(emb, data["embedding"].reshape(1, -1))[0][0]
                if sim > THRESHOLD:
                    current_name = data["name"]
                    log_attendance(sid, data["name"], DEVICE_NAME)
                    break

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, current_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------------------
# Auth Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == ADMIN_USER and password == ADMIN_PASS:
            session["admin"] = True
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("login"))

# ---------------------------
# Admin Pages
# ---------------------------
def admin_required(f):
    def wrapper(*args, **kwargs):
        if "admin" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@app.route("/home")
@admin_required
def home():
    return render_template("home.html")

@app.route("/register_page")
@admin_required
def register_page():
    return render_template("index.html")

@app.route("/attendance_page")
@admin_required
def attendance_page():
    today = datetime.date.today().isoformat()
    all_students = list(students_col.find({}, {"_id": 0}))
    present = list(attendance_col.find({"date": today}, {"_id": 0}))
    present_ids = {p["student_id"] for p in present}
    absent = [s for s in all_students if s["student_id"] not in present_ids]
    return render_template("attendance.html", present=present, absent=absent)

@app.route("/notifications_page")
@admin_required
def notifications_page():
    # count absences
    all_students = list(students_col.find({}, {"_id": 0}))
    absences = {}
    today = datetime.date.today()

    for s in all_students:
        sid = s["student_id"]
        attended_dates = {a["date"] for a in attendance_col.find({"student_id": sid})}
        total_days = (today - datetime.date(2025, 1, 1)).days + 1
        abs_count = total_days - len(attended_dates)
        absences[sid] = {"name": s["name"], "absences": abs_count}

    alerts = [v for v in absences.values() if v["absences"] >= 3]
    return render_template("notifications.html", alerts=alerts)

# ---------------------------
# API for recognition + registration
# ---------------------------
@app.route("/recognize_status")
def recognize_status():
    return jsonify({"name": current_name})

@app.route("/register_unknown", methods=["POST"])
def register_unknown():
    global known_embeddings
    data = request.json
    student_id = data["student_id"]
    name = data["name"]

    ret, frame = cap.read()
    faces = app_face.get(frame)
    if len(faces) == 0:
        return jsonify({"status": "error", "message": "No face detected"}), 400

    embedding = faces[0].embedding.tolist()
    students_col.insert_one({
        "student_id": student_id,
        "name": name,
        "embedding": embedding
    })
    known_embeddings = load_known_faces()
    return jsonify({"status": "success", "message": f"{name} registered successfully!"})

# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
