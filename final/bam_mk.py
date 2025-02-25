from flask import Flask, request, jsonify
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)


# Kết nối SQLite
def get_db_connection():
    conn = sqlite3.connect(r"db_mysql/Nhan_dien.db")
    conn.row_factory = sqlite3.Row
    return conn

# Hàm đăng ký trực tiếp trong Python
def register_user(username, password, loai="gv"):
    hashed_password = generate_password_hash(password)

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO Login (id, pass, loai) VALUES (?, ?, ?)",
                       (username, hashed_password, loai))
        conn.commit()
        print("Đăng ký thành công")
    except sqlite3.IntegrityError:
        print("Tên người dùng đã tồn tại")
    finally:
        conn.close()

# Gọi trực tiếp từ Python
register_user("test1", "123", "sv")