import cv2
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import sqlite3

# SQLite database connection
conn = sqlite3.connect('face_database.db')
c = conn.cursor()

# Create the 'users' table if not exists
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        photo BLOB
    )
''')
conn.commit()


class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super(FaceRecognitionApp, self).__init__()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.init_ui()

        # OpenCV configuration
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)

        # Timer for face recognition
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.detect_face)
        self.matched_user = None

    def init_ui(self):
        layout = QVBoxLayout(self.central_widget)

        self.register_button = QPushButton('Register', self)
        self.register_button.clicked.connect(self.show_register_dialog)
        layout.addWidget(self.register_button)

        self.find_user_button = QPushButton('Find User', self)
        self.find_user_button.clicked.connect(self.start_face_recognition)
        layout.addWidget(self.find_user_button)

        self.camera_label = QLabel(self)
        layout.addWidget(self.camera_label)

    def show_register_dialog(self):
        name, ok_pressed = QInputDialog.getText(self, "Register", "Enter your name:")
        if ok_pressed and name:
            self.register_user(name)

    def register_user(self, name):
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = gray[y:y + h, x:x + w]
            _, photo = cv2.imencode('.png', face_roi)

            # Save user to the database
            c.execute('INSERT INTO users (name, photo) VALUES (?, ?)', (name, photo.tobytes()))
            conn.commit()

            QMessageBox.information(self, 'Success', 'User registered successfully!')
        else:
            QMessageBox.warning(self, 'Error', 'No face detected. Please try again.')

    def start_face_recognition(self):
        self.matched_user = None
        self.timer.start(1000)  # Start face recognition every second

    def detect_face(self):
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Search for a matching user in the database
            for row in c.execute('SELECT * FROM users'):
                user_id, name, stored_photo = row
                stored_photo = bytearray(stored_photo)

                stored_face = cv2.imdecode(np.array(stored_photo), cv2.IMREAD_GRAYSCALE)
                current_face = gray[y:y + h, x:x + w]

                if cv2.norm(stored_face, current_face) < 50:
                    self.matched_user = name
                    break

        if self.matched_user:
            cv2.putText(frame, f'MATCHED: {self.matched_user}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
        else:
            cv2.putText(frame, 'NO MATCH', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap(q_image)
        self.camera_label.setPixmap(pixmap)

    def closeEvent(self, event):
        # Release the camera when closing the application
        self.cap.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = FaceRecognitionApp()
    main_win.show()
    sys.exit(app.exec_())
