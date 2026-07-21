from flask import Flask, render_template, redirect, url_for
from config import SECRET_KEY, TEMP_FOLDER
from database import init_db
from routes.auth_routes import auth_bp
from routes.chat_routes import chat_bp
from routes.history_routes import history_bp
import os

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Register custom Jinja2 filters
app.jinja_env.filters['basename'] = os.path.basename

app.register_blueprint(auth_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(history_bp)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    init_db()
    print("[INFO] Database initialized.")
    print("[INFO] Starting Deepfake Detection server...")
    app.run(debug=True, host="0.0.0.0", port=5000)