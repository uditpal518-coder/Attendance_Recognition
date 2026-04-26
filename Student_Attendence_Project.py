import streamlit as st
import cv2
import numpy as np
import os
import joblib
import random
import pandas as pd
from datetime import datetime
import pytz
import time
import shutil
import sqlite3
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout='wide', page_title="Smart Attendance System", page_icon="🎓")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #1f4037, #99f2c8);
}
h1, h2, h3 {
    color: #ffffff;
    text-align: center;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.stTextInput>div>div>input {
    border-radius: 10px;
}
.block-container {
    padding: 2rem;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)


BASE_DIR = "students"
os.makedirs(BASE_DIR, exist_ok=True)
HAAR_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'page' not in st.session_state:
    st.session_state.page = "Home"


def get_connection():
    """Single reusable connection helper."""
    return sqlite3.connect("attendance.db", check_same_thread=False)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            date TEXT,
            time TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_name TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()

init_db()


def save_attendance_to_db(name, date, time_str):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM attendance_records WHERE student_name = ? AND date = ?",
            (name, date)
        )
        if cursor.fetchone():
            conn.close()
            return False
        cursor.execute(
            "INSERT INTO attendance_records (student_name, date, time) VALUES (?, ?, ?)",
            (name, date, time_str)
        )
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False

def users_info(name,password):
    try:
        with sqlite3.connect('attendance.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
            "INSERT INTO users (user_name,password) VALUES (?,?)",
            (name,password)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(name,password): 
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users_info WHERE user_name = ? and password = ?",
            (name,password)
        )
        data = cursor.fetchone()
        conn.close()

        if data:
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False
    
def stu_info(name):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO students_info (student_name) VALUES (?)", (name,))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False


def delete_student(student_id, student_name):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM students_info WHERE id = ? AND student_name = ?",
            (student_id, student_name)
        )
        deleted = cursor.rowcount
        conn.commit()

        if deleted == 0:
            return False

        folder_path = os.path.join(BASE_DIR, student_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        return True
    except Exception as e:
        st.error(f"Delete Error: {e}")
        return False


def save_data(name, frame, faces):
    path = os.path.join(BASE_DIR, name)
    os.makedirs(path, exist_ok=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 51):
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))
            brightness = random.uniform(0.8, 1.2)
            face_img = cv2.convertScaleAbs(face_img, alpha=brightness, beta=0)
            cv2.imwrite(f"{path}/{name}_{i}.jpg", face_img)

        progress_bar.progress(i / 50)
        status_text.text(f"Saving Image: {i}/50")
        time.sleep(0.05)

    st.success(f"Successfully! ✅ {name} data saved.")


def train_system():
    with st.sidebar:
        X, y = [], []
        folders = [
            f for f in os.listdir(BASE_DIR)
            if os.path.isdir(os.path.join(BASE_DIR, f))
        ]

        if not folders:
            st.error("No student folders found for training.")
            return

        if len(folders) < 2:
            st.warning("Minimum 2 students required for training!")
            return

        placeholder = st.empty()
        placeholder.info("Please wait! Model is training...")

        for name in folders:
            for img_file in os.listdir(os.path.join(BASE_DIR, name)):
                img_path = os.path.join(BASE_DIR, name, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    X.append(gray_img.flatten() / 255.0)
                    y.append(name)

        if len(X) == 0:
            st.error("No valid image data found for training.")
            return

        pca = PCA(0.95)
        X_pca = pca.fit_transform(X)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_pca, y)

        joblib.dump(pca, "pca_model.pkl")
        joblib.dump(model, "lr_model.pkl")

        # ✅ Clear cached models so next load picks up new files
        load_models.clear()

        placeholder.success("Model trained successfully! ✅")


# ✅ FIX: cache_resource with better error handling
@st.cache_resource
def load_models():
    if not os.path.exists("pca_model.pkl") or not os.path.exists("lr_model.pkl"):
        return None, None
    try:
        pca = joblib.load("pca_model.pkl")
        model = joblib.load("lr_model.pkl")
        return pca, model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None



def dashboard():
    try:
        conn = get_connection()
        df = pd.read_sql(
            "SELECT COUNT(*) as count FROM attendance_records WHERE date=DATE('now', 'localtime')",
            conn
        )
        conn.close()
        return int(df['count'][0]) if len(df) > 0 else 0
    except Exception:
        return 0


def dashboard_total():
    try:
        folders = [
            f for f in os.listdir(BASE_DIR)
            if os.path.isdir(os.path.join(BASE_DIR, f))
        ]
        return len(folders)
    except Exception:
        return 0


def login_page():
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        st.markdown("<h1 style='text-align:center;'>🔐Welcome</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align:center;'>Student Attendance System </h1>", unsafe_allow_html=True)
        choice = st.radio("Select Option", ["Login", "Sign Up"], horizontal=True)

        if choice == "Login":
            st.subheader("👤 Login")
            with st.form("login_form"):
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                login_btn = st.form_submit_button("🔐Login")
            if login_btn:
                log = login_user(username,password)
                if log:
                    st.session_state.logged_in = True
                    st.rerun()   
                else:
                    st.error("Incorrect Username or Password")

        elif choice == "Sign Up":
            st.subheader("📝 Create Account")
            with st.form("singup_form"):
                new_user = st.text_input("Username", key="signup_mail")
                new_password = st.text_input("Password", type="password", key="signup_pass")
                submit_btn = st.form_submit_button("Sing Up")
            if submit_btn:
                if new_user and new_password:
                    success = users_info(new_user,new_password)
                    if success:
                        st.success("Account Create Successfully! 🎉")
                    else:
                        st.error("Username already exists. Please choose different one.")                
                else:
                    st.warning("Please enter both ausername and password.") 


if st.session_state.logged_in:

    st.sidebar.title("Main Menu")

    if st.sidebar.button("🏠 Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("➕ Add New Student"):
        st.session_state.page = "Add Students"
    if st.sidebar.button("📸 Mark Attendance"):
        st.session_state.page = "Attendance"
    if st.sidebar.button("👤 Total Students"):
        st.session_state.page = "TotalStudents"
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()  # ✅ rerun so logout takes effect

    st.sidebar.markdown("---")
    if st.sidebar.button("⚙️ System Train"):
        train_system()

    # ── HOME ──────────────────────────────────
    if st.session_state.page == "Home":
        st.title("📊 SMART ATTENDANCE DASHBOARD")
        st.write("Welcome Back! 👤 Admin..")
        st.markdown("---")

        total = dashboard_total()
        today_count = dashboard()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Students", value=total, delta="5 New")
        with col2:
            # ✅ FIX: ZeroDivisionError — check total before dividing
            if total > 0:
                pct = f"{round((today_count * 100 / total), 2)}%"
            else:
                pct = "0%"
            st.metric(label="Attendance Today", value=today_count, delta=pct)
        with col3:
            st.metric(label="Student Status", value="Ready", delta="normal")

        st.subheader("📋 Recent Attendance")
        conn = get_connection()
        df = pd.read_sql("""
            SELECT student_name, time, date
            FROM attendance_records
            WHERE date = DATE('now', 'localtime')
            ORDER BY id DESC
            LIMIT 5
        """, conn)
        conn.close()

        if not df.empty:
            df['time'] = pd.to_datetime(df['time']).dt.strftime("%I:%M %p")
            st.table(df)
        else:
            st.info("No attendance marked for today yet!")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="attendance_today.csv",
            mime="text/csv",
        )
        st.markdown("---")
        st.warning("⚠️ Data resets on cloud restart (Streamlit Cloud uses temporary storage)")

    # ── ADD STUDENT ───────────────────────────
    elif st.session_state.page == "Add Students":
        st.title("👤 REGISTER NEW STUDENT")
        with st.form("add_student_detail", clear_on_submit=True):
            name_input = st.text_input("Enter Student Name").capitalize()
            camera_img = st.camera_input("Take Photo!")
            submitted = st.form_submit_button("Save Data")

            if submitted:
                if camera_img is not None and name_input.strip() != "":
                    file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, 1)
                    face_model = cv2.CascadeClassifier(HAAR_FILE)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_model.detectMultiScale(gray, minNeighbors=10, scaleFactor=1.1)

                    if len(faces) > 0:
                        for (x, y, w, h) in faces:
                            face_img = frame[y:y+h, x:x+w]
                            face_img = cv2.resize(face_img, (100, 100))
                            st.image(face_img, channels="BGR", width=300, caption="Face Detected")
                        save_data(name_input, frame, faces)
                        stu_info(name_input)
                        st.toast(f"{name_input} added 🎉")
                        st.balloons()
                    else:
                        st.warning("Face not detected! Please try again.")
                elif camera_img is None:
                    st.info("Please take a photo first!")
                else:
                    st.warning("Enter Student Name!")

    # ── ATTENDANCE ────────────────────────────
    elif st.session_state.page == "Attendance":
        st.title("👤 ATTENDANCE RECOGNITION")
        pca, model = load_models()

        if pca is None or model is None:
            st.error("Model not trained yet! Please train the model first from the sidebar.")
        else:
            test_img = st.camera_input("Scan Face")
            if test_img is not None:
                file_bytes = np.asarray(bytearray(test_img.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_model = cv2.CascadeClassifier(HAAR_FILE)
                faces = face_model.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        face_img = frame[y:y+h, x:x+w]
                        face_img = cv2.resize(face_img, (100, 100))
                        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                        flat_img = gray_face.flatten().astype('float32') / 255.0
                        x_pca = pca.transform([flat_img])
                        probs = model.predict_proba(x_pca)[0]
                        max_prob = np.max(probs)
                        predicted_class = model.classes_[np.argmax(probs)]

                        confidence = max_prob * 100
                        name = predicted_class if confidence >= 95 else "Unknown"

                        if name != "Unknown":
                            now = datetime.now(pytz.timezone('Asia/Kolkata'))
                            current_date = now.strftime("%Y-%m-%d")
                            current_time = now.strftime("%I:%M:%S %p")

                            if save_attendance_to_db(name, current_date, current_time):
                                st.success(f"✅ {name} Marked Present!")
                            else:
                                st.warning(f"⚠️ {name} already marked present today.")

                            st.success(f"Recognized: {name}")
                            st.success(f"Confidence Score: {confidence:.2f}%")
                            st.success(f"Date: {current_date}")
                            st.success(f"Time: {current_time}")
                            st.image(face_img, caption=name, width=150)
                            st.balloons()
                        else:
                            st.error("Unknown Person — Access Denied ❌")
                else:
                    st.warning("Face not detected! Please try again.")

    # ── TOTAL STUDENTS ────────────────────────
    elif st.session_state.page == "TotalStudents":
        st.title("📋 Total Registered Students")
        st.subheader("Student Information")

        conn = get_connection()
        df = pd.read_sql("SELECT * FROM students_info", conn)
        conn.close()

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            stu_id = st.text_input("Student ID")
        with col2:
            
            stu_name = st.text_input("Student Name")
            stu_name = stu_name.capitalize()
        with col3:
            st.write("")
            st.write("")
            if st.button("❌ Delete"):
                if stu_name.strip() == "" or stu_id.strip() == "":
                    st.warning("Please enter both ID and Name before deleting!")
                else:
                    result = delete_student(stu_id, stu_name)
                    if result:
                        st.success(f"✅ Successfully deleted {stu_name}")
                        st.rerun()
                    else:
                        st.error("No student found with this ID/Name")

        if not df.empty:
            df.insert(0, 'S.No', range(1, 1 + len(df)))
            df['S.No'] = df['S.No'].astype(str)
            df['id'] = df['id'].astype(str)
            st.dataframe(df, hide_index=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="students_data.csv",
                mime="text/csv",
            )
        else:
            st.warning("No student data found!")

else:
    login_page()
