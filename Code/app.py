from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, make_response
import os
import pymysql
import re
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
load_dotenv()

import uuid
import traceback
import time

# Import the processing function from main.py
import importlib
import main as processor
importlib.reload(processor)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "yoursecretkey")

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# ----------------- DATABASE CONNECTION -----------------

db = pymysql.connect(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD", "Vedu@9970"),
    database=os.getenv("DB_NAME", "salescope"),
    cursorclass=pymysql.cursors.DictCursor
)

cursor = db.cursor()

# ----------------- FILE VALIDATION -----------------

ALLOWED_EXTENSIONS = {"csv", "xls", "xlsx"}

def allowed_file_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------- ROUTES -----------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

# ----------------- SIGNUP -----------------

@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        email = request.form["email"]
        company = request.form["company"]
        mobile = request.form["mobile"]
        owner = request.form["owner"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        # Email validation
        if not re.match(r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$', email):

            flash("Invalid email format!", "danger")

            return redirect(url_for("signup"))

        # Mobile validation
        if not re.match(r'^[0-9]{10}$', mobile):

            flash("Mobile number must be 10 digits!", "danger")

            return redirect(url_for("signup"))

        # Password validation
        if password != confirm_password:

            flash("Passwords do not match!", "danger")

            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password)

        try:

            cursor.execute(
                """
                INSERT INTO users (email, company, mobile, owner, password)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (email, company, mobile, owner, hashed_password)
            )

            db.commit()

            flash("Signup successful! Please login.", "success")

            return redirect(url_for("login"))

        except Exception as e:

            db.rollback()

            flash(str(e), "danger")

    return render_template("signup.html")

# ----------------- LOGIN -----------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        mobile = request.form["mobile"]
        password = request.form["password"]

        cursor.execute(
            "SELECT * FROM users WHERE mobile=%s",
            (mobile,)
        )

        user = cursor.fetchone()

        if user and check_password_hash(user["password"], password):

            session["user_id"] = user["id"]
            session["owner"] = user["owner"]

            return redirect(url_for("dashboard"))

        flash("Invalid credentials", "danger")

    return render_template("login.html")

# ----------------- FORGOT PASSWORD -----------------

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():

    if request.method == "POST":

        email = request.form.get("email")

        cursor.execute(
            "SELECT * FROM users WHERE email=%s",
            (email,)
        )

        user = cursor.fetchone()

        if user:

            return redirect(
                url_for(
                    "reset_password",
                    user_id=user["id"]
                )
            )

        flash("Email not found!", "danger")

    return render_template("forgot_password.html")

# ----------------- RESET PASSWORD -----------------

@app.route('/reset-password/<int:user_id>', methods=['GET', 'POST'])
def reset_password(user_id):

    if request.method == "POST":

        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if password != confirm_password:

            flash("Passwords do not match!", "danger")

            return redirect(
                url_for(
                    "reset_password",
                    user_id=user_id
                )
            )

        hashed_password = generate_password_hash(password)

        try:

            cursor.execute(
                "UPDATE users SET password=%s WHERE id=%s",
                (hashed_password, user_id)
            )

            db.commit()

            flash("Password reset successful!", "success")

            return redirect(url_for("login"))

        except Exception as e:

            db.rollback()

            flash(str(e), "danger")

    return render_template("reset_password.html")

# ----------------- DASHBOARD -----------------

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():

    if "user_id" not in session:

        flash("Please login first!", "warning")

        return redirect(url_for("login"))

    try:
        importlib.reload(processor)

    except Exception:

        print("Failed to reload processor:")
        print(traceback.format_exc())

    charts = []
    results = None
    error_message = None
    success_message = None

    if request.method == "POST":

        file = request.files.get("file")

        if not file or file.filename == "":

            error_message = "No file selected. Please choose a CSV/XLS/XLSX file."

            return render_template(
                "dashboard.html",
                results=None,
                error_message=error_message
            )

        filename = secure_filename(file.filename)

        if not allowed_file_extension(filename):

            error_message = "Only CSV, XLS, or XLSX files are allowed."

            return render_template(
                "dashboard.html",
                results=None,
                error_message=error_message
            )

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        dest_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            f"{uuid.uuid4().hex}_{filename}"
        )

        try:
            file.save(dest_path)

        except Exception as e:

            error_message = f"Failed to save file: {e}"

            print(traceback.format_exc())

            return render_template(
                "dashboard.html",
                results=None,
                error_message=error_message
            )

        try:
            proc = processor.process_data(dest_path)

        except Exception as e:

            error_message = f"Processing failed: {e}"

            print(traceback.format_exc())

            try:
                os.remove(dest_path)
            except Exception:
                pass

            return render_template(
                "dashboard.html",
                results=None,
                error_message=error_message
            )

        # Delete uploaded file after processing
        try:
            os.remove(dest_path)
        except Exception:
            pass

        if proc.get("error"):

            missing = proc.get("missing_columns") or proc.get("missing") or []

            if missing:

                cols_html = ", ".join(
                    f"<strong>{c}</strong>" for c in missing
                )

                error_message = (
                    f"Uploaded file is missing required column(s): {cols_html}"
                )

            else:

                error_message = proc.get(
                    "message",
                    "Processing failed."
                )

            return render_template(
                "dashboard.html",
                results=None,
                error_message=error_message
            )

        results = proc

        ts = int(time.time())

        for k, v in proc.get('graphs', {}).items():

            if v:

                try:
                    chart_url = url_for(
                        'static_files',
                        filename=v
                    )

                except Exception:
                    chart_url = url_for(
                        'static',
                        filename=v
                    )

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

# ----------------- LOGOUT -----------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("home"))

# ----------------- STATIC FILES -----------------

@app.route('/static/<path:filename>')
def static_files(filename):

    STATIC_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'static'
    )

    resp = make_response(
        send_from_directory(STATIC_DIR, filename)
    )

    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'

    return resp

# ----------------- MAIN -----------------

if __name__ == "__main__":

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(debug=True)
