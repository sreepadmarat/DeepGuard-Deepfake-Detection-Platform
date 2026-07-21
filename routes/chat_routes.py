from flask import Blueprint, request, jsonify, render_template, session, redirect, url_for, send_file
from functools import wraps
from database import (
    get_connection, save_prediction, get_user_predictions,
    save_chat_message, get_chat_history, get_prediction_by_id
)
from services.image_service import predict_image
from services.video_service import predict_video
from services.gemini_service import chat_with_gemini
from services.report_service import generate_report
from services.email_service import send_fake_alert
from questions import FAKE_QUESTIONS, REAL_QUESTIONS
from config import TEMP_FOLDER, BASE_DIR
import os
import uuid

chat_bp = Blueprint('chat', __name__)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated


@chat_bp.route('/chat', methods=['GET'])
@login_required
def chat():
    user_id = session['user_id']

    # Normal session chat: collect ALL prediction IDs uploaded this session
    active_ids = session.get('active_prediction_ids', [])

    # Backward-compat: if old single-id key exists and new list is empty, migrate it
    legacy_id = session.get('active_prediction_id')
    if legacy_id and not active_ids:
        active_ids = [legacy_id]
        session['active_prediction_ids'] = active_ids

    predictions_list = []
    for pid in active_ids:
        pred = get_prediction_by_id(pid)
        if pred and pred['user_id'] == user_id:
            msgs = get_chat_history(user_id, pid)
            fqs = FAKE_QUESTIONS if pred['prediction'] == 'Fake' else REAL_QUESTIONS
            predictions_list.append({
                'prediction': pred,
                'chat_messages': msgs,
                'follow_up_questions': fqs
            })

    # Keep legacy single-prediction variables populated for backward compat
    prediction = predictions_list[-1]['prediction'] if predictions_list else None
    chat_messages = predictions_list[-1]['chat_messages'] if predictions_list else []
    follow_up_questions = predictions_list[-1]['follow_up_questions'] if predictions_list else []

    return render_template(
        'chat.html',
        prediction=prediction,
        predictions_list=predictions_list,
        chat_messages=chat_messages,
        follow_up_questions=follow_up_questions
    )


@chat_bp.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return redirect(url_for('chat.chat'))

    file = request.files['file']
    file_type = request.form.get('file_type', 'image')

    if file.filename == '':
        return redirect(url_for('chat.chat'))

    user_id = session['user_id']

    # Save uploaded file to temp
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    ext = os.path.splitext(file.filename)[1].lower()
    temp_filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(TEMP_FOLDER, temp_filename)
    file.save(temp_path)

    # Run prediction
    if file_type == 'image':
        result = predict_image(temp_path)
    else:
        result = predict_video(temp_path)

    prediction_label = result['prediction']
    confidence = result['confidence']
    gradcam_path = result.get('gradcam_path', '')
    rgb_path = result.get('rgb_path', '')

    # Generate PDF report
    report_path = generate_report(
        user_name=session.get('user_name', 'User'),
        file_name=file.filename,
        file_type=file_type,
        prediction=prediction_label,
        confidence=confidence,
        gradcam_path=gradcam_path,
        rgb_path=rgb_path
    )

    # Save prediction to DB
    prediction_id = save_prediction(
        user_id=user_id,
        file_name=file.filename,
        file_type=file_type,
        prediction=prediction_label,
        confidence=confidence,
        gradcam_path=gradcam_path,
        report_path=report_path,
        rgb_path=rgb_path
    )

    # Track the active prediction for THIS session only.
    session['active_prediction_id'] = prediction_id  # legacy compat
    # Append to list so all session uploads stay visible in chat
    active_ids = session.get('active_prediction_ids', [])
    if prediction_id not in active_ids:
        active_ids.append(prediction_id)
    session['active_prediction_ids'] = active_ids

    # Save initial AI result message to chat
    verdict = 'a Deepfake' if prediction_label == 'Fake' else 'Real'
    note = 'This media appears to be AI-generated or manipulated.' if prediction_label == 'Fake' else 'This media appears to be authentic.'
    intro_msg = f"Analysis complete. The uploaded {file_type} {file.filename} has been classified as {verdict} with a confidence of {confidence:.2f}%. {note}"

    save_chat_message(user_id, prediction_id, 'assistant', intro_msg)

    # Send email alert if fake
    if prediction_label == 'Fake':
        try:
            send_fake_alert(
                receiver_email=session.get('user_email'),
                user_name=session.get('user_name', 'User'),
                file_name=file.filename,
                confidence=confidence
            )
        except Exception as e:
            print(f"[EMAIL ERROR in route] {str(e)}")

    # Clean up temp upload file (keep gradcam & report)
    try:
        os.remove(temp_path)
    except Exception:
        pass

    return redirect(url_for('chat.chat'))


@chat_bp.route('/chat', methods=['POST'])
@login_required
def chat_message():
    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'response': 'Please enter a message.'})

    user_id = session['user_id']

    # Use the session-scoped active prediction (not the latest in DB).
    active_id = session.get('active_prediction_id')
    prediction = get_prediction_by_id(active_id) if active_id else None

    prediction_id = None
    prediction_context = {'prediction': 'Unknown', 'confidence': 0, 'file_type': 'unknown', 'file_name': 'unknown'}
    if prediction and prediction['user_id'] == user_id:
        prediction_id = prediction['id']
        prediction_context = {
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'file_type': prediction['file_type'],
            'file_name': prediction['file_name']
        }

    # Save user message
    save_chat_message(user_id, prediction_id, 'user', user_message)

    # Build message history for Gemini
    raw_history = get_chat_history(user_id, prediction_id) if prediction_id else []
    messages = [{'role': m['role'], 'message': m['message']} for m in raw_history]
    messages.append({'role': 'user', 'message': user_message})

    # Get Gemini response
    ai_response = chat_with_gemini(messages=messages, prediction_context=prediction_context)

    # Save AI response
    save_chat_message(user_id, prediction_id, 'assistant', ai_response)

    return jsonify({'response': ai_response})


@chat_bp.route('/clear_chat', methods=['POST'])
@login_required
def clear_chat():
    """Clear only the current session's chat history (does NOT delete DB records)."""
    session.pop('active_prediction_ids', None)
    session.pop('active_prediction_id', None)
    return jsonify({'status': 'ok'})


@chat_bp.route('/download_report/<int:prediction_id>')
@login_required
def download_report(prediction_id):
    user_id = session['user_id']
    prediction = get_prediction_by_id(prediction_id)

    if not prediction or prediction['user_id'] != user_id:
        return redirect(url_for('chat.chat'))

    report_full_path = prediction['report_path']
    if not os.path.isabs(report_full_path):
        filename = os.path.basename(report_full_path)
        report_full_path = os.path.join(BASE_DIR, 'static', 'temp', filename)

    return send_file(
        report_full_path,
        as_attachment=True,
        download_name=f"DeepGuard_Report_{prediction_id}.pdf"
    )
