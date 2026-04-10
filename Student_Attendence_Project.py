import streamlit as st
import cv2
import numpy as np
import os
import joblib
import random
import pandas as pd
from datetime import datetime
import pytz
import sqlite3
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# --- SETTINGS ---
BASE_DIR = "students"
os.makedirs(BASE_DIR, exist_ok=True)
HAAR_FILE = "haarcascade_frontalface_default.xml"

st.set_page_config(layout='wide', page_title="Smart Attendance System", page_icon="🎓")



if 'logged_in' not in  st.session_state:
    st.session_state.logged_in = False


if 'page' not in st.session_state:
    st.session_state.page = "Home"



def login_page():
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        if not st.session_state.logged_in:
            st.title("👤Admin Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("🔐Login"):
                if username == "admin" and password == "123":
                    st.session_state.logged_in = True
                    st.rerun()

                else:
                    st.error("Invalid username and password")


def init_db():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT,
        date TEXT,
        time TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

def save_attendance_to_db(name, date, time):
    try:
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()

        check_query = "SELECT * FROM attendance_records WHERE student_name = ? AND date = ?"
        cursor.execute(check_query, (name, date))
        result = cursor.fetchone()

        if result:
            return False

        query = "INSERT INTO attendance_records (student_name, date, time) VALUES (?, ?, ?)"
        cursor.execute(query, (name, date, time))
        conn.commit()
        conn.close()

        return True

    except Exception as e:
        st.error(f"Database Error: {e}")
        return False
    
    

        

# --- FUNCTIONS ---

def save_data(name, frame, faces):
    path = os.path.join(BASE_DIR, name)
    os.makedirs(path, exist_ok=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(1,201):
        for (x,y,w,h) in faces:
            face_img = frame[y:y+h,x:x+w]
            face_img = cv2.resize(face_img,(200,200))

            brightness = random.uniform(0.8, 1.2)
            face_img = cv2.convertScaleAbs(face_img, alpha=brightness, beta=0)

            cv2.imwrite(f"{path}/{name}_{i}.jpg",face_img)

        progress_bar.progress(i / 200)
        status_text.text(f"Saving Image: {i}/200")

    st.success(f"Successfully! ✅ {name} data save..")

def train_system():
    X=[]
    y=[]
    folders = [folder_name for folder_name in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, folder_name))]

    if not folders:
        st.sidebar.error("Do not have any folder for Training")
        return
    
    st.sidebar.info("Please Wait! Model are Train new Data....  ")

    for name in folders:
        for img in os.listdir(os.path.join(BASE_DIR,name)):
            img_path = os.path.join(BASE_DIR,name,img)
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray_img is not None:
                X.append(gray_img.flatten() / 255.0)
                y.append(name)

    if len(X) > 0:
        pca = PCA(.9999)
        X_pca = pca.fit_transform(X)

        model = LogisticRegression(max_iter=200)
        model.fit(X_pca, y)

        joblib.dump(pca, "pca_model.pkl")
        joblib.dump(model, "lr_model.pkl")
        st.sidebar.success(" Model Trained SuccessFully!")   
    else:
        st.sidebar.error("Data Lessthan for Training Perpose")
    

def load_models():
    try:
        pca = joblib.load("pca_model.pkl")
        model = joblib.load("lr_model.pkl")
        return pca, model
    except:
        return None, None
    
def dashboard():
    conn = sqlite3.connect("attendance.db")
    df = pd.read_sql("SELECT COUNT(*) as count FROM attendance_records WHERE date=DATE('now')", conn)
    if len(df) > 0:
        return df['count'][0]
    return 0

def dashboard_total():
    folders = [folder_name for folder_name in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, folder_name))]
    count = len(folders)
    return count
    
  

# --- SIDEBAR NAVIGATION ---

if st.session_state.logged_in:

    st.sidebar.title("Main Menu")

    if st.sidebar.button("🏠Home"):
        st.session_state.page = "Home"

    if st.sidebar.button("➕Add New Student"):
        st.session_state.page = "AddStudent"

    if st.sidebar.button("📸Mark Attendance"):
        st.session_state.page = "Attendance"

    if st.sidebar.button("🚪Logout"):
            st.session_state.logged_in = False
            st.rerun()


    st.sidebar.markdown("---")
    if st.sidebar.button("⚙️System Train"):
        train_system()

    # --- PAGE LOGIC --
    if st.session_state.page =="Home":
        st.title("📊SMART ATTENDANCE DASHBOARD")
        st.write("welcome Back!👤Admin..")
        st.markdown("---")
        
        col1,col2,col3 = st.columns(3)

        with col1:
            st.metric(label="Total Student", value=dashboard_total(),delta="5 New")
        with col2:
            st.metric(label="Attendance Today", value=dashboard(),delta=f"{round((dashboard() * 100 / dashboard_total()),2)}%")
        with col3:
            st.metric(label="Student Status", value="Ready",delta="normal")

        st.subheader(" RECENT ATTENDANCE")
        conn = sqlite3.connect("attendance.db")
        df = pd.read_sql("""
        SELECT student_name, time, date 
        FROM attendance_records 
        ORDER BY id DESC 
        LIMIT 5
        """, conn)
        df['time'] = df['time'].astype(str).map(lambda x: x.split()[-1])
        df['time'] = pd.to_datetime(df['time'].astype(str)).dt.strftime('%I:%M%p')
        st.table(df)
        st.markdown("---")
        st.warning("⚠️ Data resets on cloud restart")

    elif st.session_state.page == "AddStudent":
        st.title("👤REGISTRATION NEW STUDENT")

        name_input = st.text_input("Enter Student Name")
        camera_img = st.camera_input("Take Photo!")

        if camera_img is not None and name_input:
            file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            face_model=cv2.CascadeClassifier(HAAR_FILE)
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_model.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                #st.image(frame, channels="BGR", width=300, caption="Face Detected")
                if st.button("Save Data"):
                    save_data(name_input, frame, faces)

            else:
                st.warning("Face Not Detect! Please Try Again...")

        elif camera_img is not None and not name_input:
            st.warning("Enter Student Name!")
        else:
            st.info("Please enter a Student Name and Take a Photo!")

            
    elif st.session_state.page == "Attendance":
        st.title("👤ATTENDANCE RECOGNITION")
        pca, model = load_models()

        if pca is None or model is None:
            st.error("Model are not Train! firstly Train Model...")

        else:
            test_img = st.camera_input("Scan Face")
            if test_img is not None:
                file_bytes = np.asarray(bytearray(test_img.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_model = cv2.CascadeClassifier(HAAR_FILE)
                faces = face_model.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    for (x,y,w,h) in faces:
                        face_img = frame[y:y+h, x:x+w]
                        face_img = cv2.resize(face_img, (200, 200))
                        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                        flat_img = gray_face.flatten().astype('float32') / 255.0
                        x_pca = pca.transform([flat_img])
                        probs = model.predict_proba(x_pca)[0]
                        max_prob = np.max(probs)
                        predicted_class = model.classes_[np.argmax(probs)]

                        confidence = max_prob * 100
                        if confidence >= 75:
                            name = predicted_class
                        else:
                            name = "Unknown"
                        #name = model.predict(x_pca)[0]
                        if name != "Unknown":
                            now = datetime.now(pytz.timezone('Asia/Kolkata'))
                            current_date =now.strftime("%Y-%m-%d")
                            current_time = now.strftime("%H:%M:%S")
    
                            if save_attendance_to_db(name, current_date, current_time):
                                st.success(f"{name} Marked Present ✅!")
    
                            else:
                                st.warning(f"You have Already marke ✅ Today Attendance {name}")
    
                            st.success(f"Recognized: {name}") 
                            st.success(f"Confidence Score: {confidence}") 
                            st.success(f"Date: {current_date}") 
                            st.success(f"Time: {current_time}")
                            st.image(face_img, caption=name, width=150)
                        else:
                            st.error("Unknown Person")
                else: 
                    st.warning("Face not Detect! Please Try Again...")    
else:
    login_page()
        


