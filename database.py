import sqlite3
from config import DATABASE_PATH


def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            gradcam_path TEXT,
            report_path TEXT,
            rgb_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Migration: add rgb_path column to existing DBs that don't have it
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN rgb_path TEXT")
    except Exception:
        pass  # Column already exists

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction_id INTEGER,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    """)

    conn.commit()
    conn.close()


def get_user_by_email(email):
    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return user


def create_user(name, email, password):
    conn = get_connection()
    conn.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password))
    conn.commit()
    conn.close()


def save_prediction(user_id, file_name, file_type, prediction, confidence, gradcam_path, report_path, rgb_path=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (user_id, file_name, file_type, prediction, confidence, gradcam_path, report_path, rgb_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, file_name, file_type, prediction, confidence, gradcam_path, report_path, rgb_path))
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id


def get_user_predictions(user_id):
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC
    """, (user_id,)).fetchall()
    conn.close()
    return rows


def save_chat_message(user_id, prediction_id, role, message):
    conn = get_connection()
    conn.execute("""
        INSERT INTO chat_messages (user_id, prediction_id, role, message)
        VALUES (?, ?, ?, ?)
    """, (user_id, prediction_id, role, message))
    conn.commit()
    conn.close()


def get_chat_history(user_id, prediction_id):
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM chat_messages WHERE user_id = ? AND prediction_id = ? ORDER BY created_at ASC
    """, (user_id, prediction_id)).fetchall()
    conn.close()
    return rows


def get_prediction_by_id(prediction_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    conn.close()
    return row