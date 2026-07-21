import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import SMTP_HOST, SMTP_PORT, SMTP_EMAIL, SMTP_PASSWORD


def send_fake_alert(receiver_email, user_name, file_name, confidence):
    if not receiver_email:
        print("[EMAIL ERROR] No receiver email provided.")
        return False

    subject = "⚠️ Deepfake Alert — Fake Content Detected by DeepGuard"

    body = f"""Hello {user_name},

DeepGuard has flagged the following file as a deepfake:

  File Name  : {file_name}
  Prediction : FAKE
  Confidence : {confidence:.2f}%

This content has been identified as AI-generated or manipulated media.
Please review and take appropriate action.

This is an automated alert from DeepGuard Deepfake Detection System.
"""

    msg = MIMEMultipart()
    msg["From"] = f"DeepGuard <{SMTP_EMAIL}>"
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, receiver_email, msg.as_string())
        server.quit()
        print(f"[EMAIL] Alert sent successfully to {receiver_email}")
        return True
    except smtplib.SMTPAuthenticationError:
        print("[EMAIL ERROR] Authentication failed. Check SMTP_EMAIL and SMTP_PASSWORD in config.py.")
        return False
    except smtplib.SMTPException as e:
        print(f"[EMAIL ERROR] SMTP error: {str(e)}")
        return False
    except Exception as e:
        print(f"[EMAIL ERROR] Unexpected error: {str(e)}")
        return False