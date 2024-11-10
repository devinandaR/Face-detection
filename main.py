import cv2
import numpy as np
import os
import pickle
import sqlite3
from datetime import datetime
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import glob

def create_model(num_classes):
    """Create a more complex CNN model for face recognition"""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Fourth Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_face_recognition(training_images_path, model_file="face_model.h5"):
    print("Starting training process...")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Get student folders
    student_folders = [f for f in os.listdir(training_images_path) 
                      if os.path.isdir(os.path.join(training_images_path, f))]
    
    if not student_folders:
        print("No student folders found!")
        return None, face_cascade
    
    print(f"Found students: {student_folders}")
    
    # Initialize data structures
    X = []
    y = []
    
    # Process each student's images
    for class_idx, student_name in enumerate(student_folders):
        student_path = os.path.join(training_images_path, student_name)
        print(f"\nProcessing {student_name}'s images...")
        
        # Get all images for this student
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(student_path, ext)))
        
        if not image_files:
            print(f"No images found for {student_name}")
            continue
            
        # Process each image
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Detect face
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                continue
                
            # Process the largest face in the image
            x, y_, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face = img[y_:y_+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = face.astype('float32') / 255.0
            
            X.append(face)
            y.append(class_idx)  # Use numerical index for classes
            
        print(f"Processed {len([i for i in y if i == class_idx])} images for {student_name}")
    
    if not X:
        print("No faces detected in any images!")
        return None, face_cascade
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Create label encoder with explicit mapping
    le = LabelEncoder()
    le.fit(range(len(student_folders)))  # Fit with numerical indices
    le.classes_ = np.array(student_folders)  # Set class names explicitly
    
    # Save the label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Create and train model
    model = create_model(len(student_folders))
    print("\nTraining model...")
    
    # Train with validation split
    history = model.fit(
        X, y,
        validation_split=0.2,
        epochs=50,
        batch_size=16,
        verbose=1
    )
    
    # Save model
    model.save(model_file)
    print(f"\nModel saved as {model_file}")
    
    return model, face_cascade

def recognize_faces():
    """Perform real-time face recognition"""
    print("Starting face recognition...")
    
    # Load the saved model and label encoder
    try:
        model = load_model("face_model.h5")
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("Model and encoder loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the system first")
        return
    
    # Initialize video capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open video capture device")
        return
    
    print("Starting video capture... Press 'c' to capture attendance, 'q' to quit")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        current_names = []  # Store names of people in current frame
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            try:
                face_roi = frame[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (224, 224))
                face_roi = face_roi.astype('float32') / 255.0
                face_roi = np.expand_dims(face_roi, axis=0)
                
                predictions = model.predict(face_roi)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class]
                
                predicted_name = le.inverse_transform([predicted_class])[0]
                
                if confidence > 0.95:
                    cv2.putText(frame, predicted_name, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    current_names.append(predicted_name)
                else:
                    cv2.putText(frame, "Unknown", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
            except Exception as e:
                print(f"Error processing face: {e}")
        
        cv2.putText(frame, "Press 'c' to capture attendance", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Face Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if current_names:
                print("\nLogging attendance for:", current_names)
                for name in current_names:
                    log_attendance(name)
                print("Attendance logged successfully!")
            else:
                print("No recognized faces to log attendance")
    
    video_capture.release()
    cv2.destroyAllWindows()

def setup_database():
    """Create and populate the database with student information from CSV"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            roll_number TEXT PRIMARY KEY,
            name TEXT NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT,
            login_time DATETIME,
            FOREIGN KEY (roll_number) REFERENCES students(roll_number)
        )
    ''')
    
    try:
        df = pd.read_csv('students.csv')
        df.to_sql('students', conn, if_exists='replace', index=False)
        print("Database initialized with student data from CSV")
    except Exception as e:
        print(f"Error loading student data: {str(e)}")
    finally:
        conn.close()

def log_attendance(name):
    """Log attendance when a face is recognized"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT roll_number FROM students WHERE name = ?', (name,))
        result = cursor.fetchone()
        
        if result:
            roll_number = result[0]
            current_time = datetime.now()
            cursor.execute('''
                INSERT INTO attendance_logs (roll_number, login_time)
                VALUES (?, ?)
            ''', (roll_number, current_time))
            conn.commit()
            print(f"Logged attendance for {name} (Roll: {roll_number}) at {current_time}")
        else:
            print(f"No roll number found for {name}")
    finally:
        conn.close()

def view_attendance_logs():
    """View all attendance logs"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT s.name, s.roll_number, a.login_time 
            FROM attendance_logs a 
            JOIN students s ON a.roll_number = s.roll_number 
            ORDER BY a.login_time DESC
        ''')
        logs = cursor.fetchall()
        
        print("\nAttendance Logs:")
        print("Name | Roll Number | Login Time")
        print("-" * 50)
        for log in logs:
            print(f"{log[0]} | {log[1]} | {log[2]}")
    finally:
        conn.close()

if __name__ == '__main__':
    print("Main.py is being run directly")
else:
    print("Main.py is being imported as a module")
