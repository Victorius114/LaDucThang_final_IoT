import os
import numpy as np
import cv2
import sqlite3
import json
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
from scipy.spatial.distance import cosine
import pandas as pd
import onnxruntime
from werkzeug.security import check_password_hash

app = Flask(__name__)

# Kết nối với cơ sở dữ liệu SQLite
def connect_to_db():
    conn = sqlite3.connect(r'db_mysql/Nhan_dien.db', check_same_thread=False)
    return conn

# Lấy danh sách MSSV và thông tin từ SQLite
def fetch_mssv_list(conn):
    query = "SELECT * FROM in4_sv"
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    mssv_list = [row[0] for row in rows]
    ten_list = [row[1] for row in rows]
    lop_list = [row[2] for row in rows]
    khoa_list = [row[3] for row in rows]
    return mssv_list, ten_list, lop_list, khoa_list

@app.route('/get_time', methods=['GET'])
def get_time():
    query = "SELECT gio, phut FROM Thoigian_tiet LIMIT 1"
    cursor = conn.cursor()
    cursor.execute(query)
    row = cursor.fetchone()

    if row:
        gio, phut = row
        return jsonify({"gio": gio, "phut": phut})
    else:
        return jsonify({"gio": 0, "phut": 0})  # Nếu không có dữ liệu

@app.route('/save_time', methods=['GET', 'POST'])
def save_time():
    data = request.get_json()
    hour = int(data.get('hour', 0))
    minute = int(data.get('minute', 0))

    try:
        # Cập nhật giá trị giờ và phút trong bảng Thoigian_tiet
        query = "UPDATE Thoigian_tiet SET gio = ?, phut = ? WHERE id = 1"
        cursor = conn.cursor()
        cursor.execute(query, (hour, minute))
        conn.commit()

        # Kiểm tra xem có cập nhật dòng không
        if cursor.rowcount > 0:
            return jsonify({'message': 'Thời gian đã được lưu thành công.'})
        else:
            return jsonify({'message': 'Không có dòng nào được cập nhật.'}), 400

    except Exception as e:
        # In ra lỗi chi tiết
        return jsonify({'message': f'Lỗi khi lưu thời gian: {str(e)}'}), 500


@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'fail', 'message': 'Request không chứa JSON'}), 400

        user_id = data.get('id')
        password = data.get('pass')

        if not user_id or not password:
            return jsonify({'status': 'fail', 'message': 'Thiếu ID hoặc mật khẩu'}), 400

        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT pass, loai FROM Login WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[0], password):
            loai = user[1]
            path = '/index_gv' if loai == 'gv' else '/index_sv'
            return jsonify({'status': 'success', 'type': loai, 'redirect': path})
        else:
            return jsonify({'status': 'fail', 'message': 'Sai ID hoặc mật khẩu'}), 401
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# Khởi tạo mô hình ONNX
onnx_session = onnxruntime.InferenceSession(r'face_recognition_model.onnx')

# Đọc dữ liệu từ thư mục và lưu embeddings
dataset_dir = r'dataset'
dataset_embeddings = []
dataset_labels = []

for label in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, label)
    if not os.path.isdir(person_dir):
        continue
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        face = cv2.imread(image_path)
        if face is not None:
            face = cv2.resize(face, (160, 160))
            face = face.astype('float32')
            face = (face / 255.0 - 0.5) / 0.5  # Normalize
            face = np.transpose(face, (2, 0, 1))  # HWC to CHW
            face = np.expand_dims(face, axis=0)  # Add batch dimension

            # Inference với ONNX
            input_name = onnx_session.get_inputs()[0].name
            face_embedding = onnx_session.run(None, {input_name: face})[0].flatten()
            dataset_embeddings.append(face_embedding)
            dataset_labels.append(label)

dataset_embeddings = np.array(dataset_embeddings)
dataset_labels = np.array(dataset_labels)

# Kết nối tới SQLite
conn = connect_to_db()
mssv_list, ten_list, lop_list, khoa_list = fetch_mssv_list(conn)

detected_label = None
current_frame = None  # Biến toàn cục để lưu frame hiện tại

# Hàm nhận diện khuôn mặt từ webcam
def detect_face():
    global detected_label, current_frame
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        current_frame = frame  # Lưu frame hiện tại

        # Phát hiện khuôn mặt bằng MTCNN (có thể thay thế bằng OpenCV DNN nếu cần)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            detected_label = None
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (160, 160))
                face = face.astype('float32')
                face = (face / 255.0 - 0.5) / 0.5  # Normalize
                face = np.transpose(face, (2, 0, 1))  # HWC to CHW
                face = np.expand_dims(face, axis=0)  # Add batch dimension

                # Inference với ONNX
                input_name = onnx_session.get_inputs()[0].name
                face_embedding = onnx_session.run(None, {input_name: face})[0].flatten()

                distances = [cosine(face_embedding, stored_embedding) for stored_embedding in dataset_embeddings]
                min_distance_idx = np.argmin(distances)
                min_distance = distances[min_distance_idx]
                label = dataset_labels[min_distance_idx] if min_distance < 0.4 else "Unknown"
                detected_label = label
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/')
def loginscr():
    return render_template('login.html')

@app.route('/index_gv')
def index_gv():
    return render_template('index_gv.html')

@app.route('/index_sv')
def index_sv():
    return render_template('index_sv.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_face(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/diemdanh', methods=['POST'])
def diemdanh():
    global detected_label, current_frame
    if detected_label:
        data = request.get_json()
        hour = int(data.get('hour', 0))
        minute = int(data.get('minute', 0))

        query = "SELECT gio, phut FROM Thoigian_tiet LIMIT 1"
        cursor = conn.cursor()
        cursor.execute(query)
        row = cursor.fetchone()

        if row:
            gio, phut = row
            if gio != hour or phut != minute:
                update_query = "UPDATE Thoigian_tiet SET gio = ?, phut = ? WHERE gio = ? AND phut = ?"
                cursor.execute(update_query, (hour, minute, gio, phut))
                conn.commit()

        # Lấy thời gian hiện tại và thời gian điểm danh
        time_now = datetime.now()
        diemdanh_time = time_now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        # Kiểm tra xem có đến muộn hay không
        state = "Đúng giờ" if time_now < diemdanh_time else "Đến muộn"

        index = mssv_list.index(detected_label)
        time_str = time_now.strftime("%Y-%m-%d %H:%M:%S")

        check_query = "SELECT gio, phut FROM Thoigian_tiet LIMIT 1"
        cursor.execute(check_query)
        lesson_time = cursor.fetchone()

        if lesson_time:
            lesson_hour, lesson_minute = lesson_time  # Giờ và phút của tiết học

            # Lấy lần điểm danh gần nhất của sinh viên
            check_query = """
                SELECT [Thời gian điểm danh] FROM Diem_danh 
                WHERE MSSV = ? 
                ORDER BY [Thời gian điểm danh] DESC LIMIT 1
            """
            cursor.execute(check_query, (mssv_list[index],))
            last_diemdanh = cursor.fetchone()

            if last_diemdanh:
                last_diemdanh_time = datetime.strptime(last_diemdanh[0], "%Y-%m-%d %H:%M:%S")

                # Tạo thời gian bắt đầu tiết học
                lesson_start_time = last_diemdanh_time.replace(hour=lesson_hour, minute=lesson_minute, second=0)

                # Tính chênh lệch thời gian (phút)
                time_diff = (last_diemdanh_time - lesson_start_time).total_seconds() / 60

                if 0 <= time_diff <= 50:  # Nếu điểm danh trong vòng 50 phút kể từ tiết học
                    return jsonify({
                        'status': 'fail',
                        'message': f'{mssv_list[index]} đã điểm danh vào lúc {last_diemdanh[0]}'
                    })

        # Nếu chưa điểm danh trong 50 phút, lưu vào database
        query = "INSERT INTO Diem_danh (MSSV, [Thời gian điểm danh], [Trạng thái]) VALUES (?, ?, ?)"
        try:
            cursor.execute(query, (mssv_list[index], time_str, state))
            conn.commit()
            inserted_id = cursor.lastrowid
            return jsonify({
                'id': inserted_id,
                'mssv': mssv_list[index],
                'status': 'success',
                'message': f'Đã điểm danh - {state}'
            })
        except sqlite3.Error as e:
            conn.rollback()
            return jsonify({'status': 'fail', 'message': f'Lỗi khi chèn dữ liệu: {str(e)}'})

    return jsonify({'status': 'fail', 'message': 'Không nhận diện được khuôn mặt'})


@app.route('/diemdanh_list')
def diemdanh_list():
    query = (f"SELECT in4_sv.MSSV, in4_sv.[Họ và tên], Diem_danh.[Thời gian điểm danh], Diem_danh.[Trạng thái] "
             f"FROM in4_sv "
             f"INNER JOIN Diem_danh ON in4_sv.MSSV = Diem_danh.MSSV "
             f"ORDER BY id ASC")
    df = pd.read_sql(query, conn)
    table_html = df.to_html(classes='table table-bordered', index=False).replace('\n', '')
    return render_template('diemdanh_list.html', tables=table_html)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)