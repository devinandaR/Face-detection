import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                            QVBoxLayout, QWidget, QTableWidget, 
                            QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
import sqlite3
import os

# Import main module functions
try:
    from main import recognize_faces, view_attendance_logs, train_face_recognition
    print("Successfully imported from main.py")
except ImportError as e:
    print(f"Failed to import from main.py: {e}")

class AttendanceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Attendance System")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()
        
    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create buttons
        self.login_button = QPushButton("Take Attendance")
        self.view_button = QPushButton("View Attendance")
        self.train_button = QPushButton("Train System")
        
        # Style buttons
        button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 16px;
                min-height: 50px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        
        # Apply style and add buttons to layout
        for button in [self.login_button, self.view_button, self.train_button]:
            button.setStyleSheet(button_style)
            layout.addWidget(button)
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Name', 'Roll Number', 'Login Time'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        
        # Connect buttons to functions
        self.login_button.clicked.connect(self.start_recognition)
        self.view_button.clicked.connect(self.show_attendance)
        self.train_button.clicked.connect(self.train_system)
        
        # Show the window
        self.show()
    
    def start_recognition(self):
        try:
            # Hide the main window before starting recognition
            self.hide()
            QApplication.processEvents()  # Process any pending events
            
            # Start face recognition
            recognize_faces()
            
            # Show the window again after recognition ends
            self.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during recognition: {str(e)}")
            self.show()
    
    def show_attendance(self):
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.name, s.roll_number, a.login_time 
                FROM attendance_logs a 
                JOIN students s ON a.roll_number = s.roll_number 
                ORDER BY a.login_time DESC
            ''')
            data = cursor.fetchall()
            
            self.table.setRowCount(len(data))
            for row, (name, roll, time) in enumerate(data):
                self.table.setItem(row, 0, QTableWidgetItem(name))
                self.table.setItem(row, 1, QTableWidgetItem(roll))
                self.table.setItem(row, 2, QTableWidgetItem(str(time)))
            
            conn.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading attendance data: {str(e)}")
    
    def train_system(self):
        try:
            # Get absolute path to training_images directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            training_dir = os.path.join(current_dir, "training_images")
            
            print(f"Looking for training images in: {training_dir}")  # Debug print
            
            if not os.path.exists(training_dir):
                os.makedirs(training_dir)
                QMessageBox.information(
                    self,
                    "Directory Created",
                    f"Training directory created at: {training_dir}\n"
                    "Please add student folders with face images before training."
                )
                return
                
            # Check for student folders and images
            student_folders = [f for f in os.listdir(training_dir) 
                             if os.path.isdir(os.path.join(training_dir, f))]
            
            if not student_folders:
                QMessageBox.warning(
                    self,
                    "No Student Folders",
                    f"No student folders found in: {training_dir}\n"
                    "Please create folders named with student names and add their photos."
                )
                return
                
            # Count total images
            total_images = 0
            for student in student_folders:
                student_path = os.path.join(training_dir, student)
                images = [f for f in os.listdir(student_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                total_images += len(images)
                
            if total_images == 0:
                QMessageBox.warning(
                    self,
                    "No Images",
                    f"No images found in student folders in: {training_dir}\n"
                    "Please add .jpg, .jpeg, or .png images to the student folders."
                )
                return
                
            print(f"Found {len(student_folders)} students with {total_images} total images")
            
            QMessageBox.information(
                self, 
                "Training", 
                f"Starting training process...\n"
                f"Found {len(student_folders)} students with {total_images} images.\n"
                "This may take several minutes."
            )
            
            model, _ = train_face_recognition(training_dir)
            
            if model is not None:
                QMessageBox.information(
                    self, 
                    "Success", 
                    "Training completed successfully!"
                )
            else:
                QMessageBox.warning(
                    self, 
                    "Warning", 
                    "Training process did not complete successfully.\n"
                    "Please check the console for error messages."
                )
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Error during training: {str(e)}\n"
                f"Looking in directory: {training_dir}"
            )

def main():
    # Create application
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = AttendanceApp()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()