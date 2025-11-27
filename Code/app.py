# app.py
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, make_response
import os
import pymysql
import re
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
load_dotenv()

import secrets
from datetime import datetime, timedelta
import uuid
import traceback
import time

# Import the processing function from main.py
import importlib
import main as processor
importlib.reload(processor)

# Optional imports for email (SMTP)
import smtplib, ssl
from email.message import EmailMessage

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "yoursecretkey")
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# --- Database connection (unchanged) ---
db = pymysql.connect(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD", "Vedu@9970"),
    database=os.getenv("DB_NAME", "salescope"),
    cursorclass=pymysql.cursors.DictCursor
)
cursor = db.cursor()

ALLOWED_EXTENSIONS = {"csv", "xls", "xlsx"}
def allowed_file_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Password reset configuration ---
RESET_TOKEN_EXPIRY_MINUTES = int(os.getenv("RESET_TOKEN_EXPIRY_MINUTES", "60"))

# --- Password reset helpers ---
def create_password_reset_token(user_id):
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRY_MINUTES)
    try:
        cursor.execute(
            "INSERT INTO password_resets (user_id, token, expires_at, used) VALUES (%s, %s, %s, 0)",
            (user_id, token, expires_at)
        )
        db.commit()
        return token
    except Exception:
        db.rollback()
        return None

def get_password_reset_record(token):
    cursor.execute("SELECT * FROM password_resets WHERE token=%s AND used=0", (token,))
    return cursor.fetchone()

def mark_token_used(token):
    try:
        cursor.execute("UPDATE password_resets SET used=1 WHERE token=%s", (token,))
        db.commit()
    except Exception:
        db.rollback()

def invalidate_old_tokens_for_user(user_id):
    try:
        cursor.execute("UPDATE password_resets SET used=1 WHERE user_id=%s", (user_id,))
        db.commit()
    except Exception:
        db.rollback()

def send_reset_email(to_email, reset_url):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    mail_from = os.getenv("MAIL_FROM", smtp_user)
    use_ssl = os.getenv("SMTP_USE_SSL", "0") == "1"

    # If SMTP not configured, return False (no-op)
    if not smtp_host or not smtp_user or not smtp_pass or not mail_from:
        return False

    subject = "SaleScope — Password reset request"
    body = f"""Hello,

We received a request to reset your SaleScope password. 
Click below to reset (valid for {RESET_TOKEN_EXPIRY_MINUTES} minutes):

{reset_url}

If you didn’t request it, ignore this email.
"""

    msg = EmailMessage()
    msg["From"] = mail_from
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        if use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls(context=ssl.create_default_context())
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        return True
    except Exception:
        return False

# ----------------- ROUTES -----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        company = request.form["company"]
        mobile = request.form["mobile"]
        owner = request.form["owner"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if not re.match(r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$', email):
            flash("Invalid email format!", "danger")
            return redirect(url_for("signup"))

        if not re.match(r'^[0-9]{10}$', mobile):
            flash("Mobile number must be 10 digits!", "danger")
            return redirect(url_for("signup"))

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password)

        try:
            cursor.execute(
                "INSERT INTO users (email, company, mobile, owner, password) VALUES (%s, %s, %s, %s, %s)",
                (email, company, mobile, owner, hashed_password)
            )
            db.commit()
            flash("Signup successful! Please login.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            db.rollback()
            flash(str(e), "danger")

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        mobile = request.form["mobile"]
        password = request.form["password"]

        cursor.execute("SELECT * FROM users WHERE mobile=%s", (mobile,))
        user = cursor.fetchone()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["owner"] = user["owner"]
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "danger")

    return render_template("login.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("login"))

    try:
        importlib.reload(processor)
    except Exception:
        print("Failed to reload processor:", traceback.format_exc())

    charts = []
    results = None
    error_message = None
    success_message = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error_message = "No file selected. Please choose a CSV/XLS/XLSX file."
            return render_template("dashboard.html", results=None, error_message=error_message)

        filename = secure_filename(file.filename)
        if not allowed_file_extension(filename):
            error_message = "Only CSV, XLS, or XLSX files are allowed."
            return render_template("dashboard.html", results=None, error_message=error_message)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        dest_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_{filename}")
        try:
            file.save(dest_path)
        except Exception as e:
            error_message = f"Failed to save file: {e}"
            print("DEBUG: save failed", traceback.format_exc())
            return render_template("dashboard.html", results=None, error_message=error_message)

        try:
            proc = processor.process_data(dest_path)
        except Exception as e:
            error_message = f"Processing failed: {e}"
            print("DEBUG: processing exception", traceback.format_exc())
            try:
                os.remove(dest_path)
            except Exception:
                pass
            return render_template("dashboard.html", results=None, error_message=error_message)

        try:
            os.remove(dest_path)
        except Exception:
            pass

        if proc.get("error"):
            missing = proc.get("missing_columns") or proc.get("missing") or []
            if missing:
                cols_html = ", ".join(f"<strong>{c}</strong>" for c in missing)
                error_message = f"Uploaded file is missing required column(s): {cols_html}"
            else:
                error_message = proc.get("message", "Processing failed.")
            return render_template("dashboard.html", results=None, error_message=error_message)

        results = proc
        ts = int(time.time())
        for k, v in proc.get('graphs', {}).items():
            if v:
                try:
                    chart_url = url_for('static_files', filename=v)
                except Exception:
                    chart_url = url_for('static', filename=v)
                chart_url = f"{chart_url}?v={ts}"
                charts.append(chart_url)

        success_message = "File processed successfully!"

    return render_template(
        "dashboard.html",
        charts=charts,
        owner=session.get("owner"),
        results=results,
        error_message=error_message,
        success_message=success_message,
        **(results or {})
    )

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ----------------- Password Reset -----------------
# Ensure these endpoints exist and match the names used in templates (forgot_password)
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        if user:
            invalidate_old_tokens_for_user(user["id"])
            token = create_password_reset_token(user["id"])
            if token:
                reset_url = url_for("reset_password", token=token, _external=True)
                # Try to send email if SMTP configured; otherwise show link in dev mode
                sent = send_reset_email(email, reset_url)
                flash("If SMTP is configured, a reset link was sent to your email.", "info")
                return render_template("forgot_password_sent.html", reset_url=reset_url, email=email, sent=sent)
        # Always show a neutral message so attackers can't enumerate emails
        flash("If the email exists, a link will be sent.", "info")
        return redirect(url_for("login"))

    return render_template("forgot_password.html")

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    rec = get_password_reset_record(token)
    if not rec:
        flash("Invalid or expired link.", "danger")
        return redirect(url_for("login"))

    if request.method == "POST":
        password = request.form.get("password")
        confirm = request.form.get("confirm_password")

        if password != confirm:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("reset_password", token=token))

        hashed = generate_password_hash(password)
        try:
            cursor.execute("UPDATE users SET password=%s WHERE id=%s", (hashed, rec["user_id"]))
            mark_token_used(token)
            db.commit()
            flash("Password reset successful. Please login.", "success")
            return redirect(url_for("login"))
        except Exception:
            db.rollback()
            flash("Failed to reset password. Try again.", "danger")
            return redirect(url_for("reset_password", token=token))

    return render_template("reset_password.html", token=token)

# Serve static files (images) with no-cache headers to help ensure browsers fetch latest files
@app.route('/static/<path:filename>')
def static_files(filename):
    STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    resp = make_response(send_from_directory(STATIC_DIR, filename))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Debug: print registered endpoints so you can confirm forgot_password, reset_password exist
    print("Registered endpoints:")
    for rule in app.url_map.iter_rules():
        print(rule.endpoint, "->", rule)

    app.run(debug=True)
