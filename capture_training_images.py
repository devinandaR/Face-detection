import cv2
import os

def capture_training_images():
    # Get student name
    student_name = input("Enter student name (no spaces, use underscore): ").strip()
    
    # Create directory for student
    student_dir = os.path.join("training_images", student_name)
    if not os.path.exists(student_dir):
        os.makedirs(student_dir)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    image_count = 0
    max_images = 20  # Number of images to capture per student
    
    print("\nPress 'c' to capture an image")
    print("Press 'q' to quit")
    
    while image_count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Capture Training Images', frame)
        
        # Check for keypress
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c') and len(faces) > 0:
            # Save the image
            image_path = os.path.join(student_dir, f"{student_name}_{image_count}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Captured image {image_count + 1}/{max_images}")
            image_count += 1
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nCaptured {image_count} images for {student_name}")
    print(f"Images saved in: {student_dir}")

def main():
    print("Face Recognition Training Image Capture Tool")
    print("===========================================")
    
    # Create training_images directory if it doesn't exist
    if not os.path.exists("training_images"):
        os.makedirs("training_images")
    
    while True:
        print("\n1. Capture images for new student")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == '1':
            capture_training_images()
        elif choice == '2':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 