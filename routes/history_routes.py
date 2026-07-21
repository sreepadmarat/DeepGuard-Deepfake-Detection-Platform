import os
from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify, send_file
from database import get_user_predictions, get_prediction_by_id
from config import TEMP_FOLDER

history_bp = Blueprint("history", __name__)


def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


@history_bp.route("/history")
@login_required
def history_page():
    predictions = get_user_predictions(session["user_id"])
    records = [dict(p) for p in predictions]
    return render_template("history.html", predictions=records, user_name=session.get("user_name"))


@history_bp.route("/history/prediction/<int:prediction_id>")
@login_required
def get_history_prediction(prediction_id):
    """Return JSON data for a single historical prediction (used by the View modal)."""
    user_id = session["user_id"]
    prediction = get_prediction_by_id(prediction_id)

    if not prediction or prediction["user_id"] != user_id:
        return jsonify({"error": "Not found"}), 404

    p = dict(prediction)
    # Normalize image paths to URL-friendly static paths
    def to_static_url(path):
        if not path:
            return None
        return "/static/temp/" + os.path.basename(path)

    return jsonify({
        "id": p["id"],
        "file_name": p["file_name"],
        "file_type": p["file_type"],
        "prediction": p["prediction"],
        "confidence": p["confidence"],
        "gradcam_url": to_static_url(p.get("gradcam_path")),
        "rgb_url": to_static_url(p.get("rgb_path")),
        "has_report": bool(p.get("report_path")),
        "created_at": p["created_at"],
    })


@history_bp.route("/download/report/history/<path:report_path>")
@login_required
def download_history_report(report_path):
    full_path = os.path.join(os.path.dirname(TEMP_FOLDER), report_path)
    if not os.path.exists(full_path):
        return jsonify({"error": "Report not found"}), 404
    return send_file(full_path, as_attachment=True, download_name="deepfake_report.pdf")
