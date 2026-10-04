import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, flash
from mock_app.seed import get_db_path, init_db, reset_db

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "centralign-mock-app-secret-key-12345")

DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def get_db_connection():
    db_path = get_db_path()
    if not db_path.exists():
        init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            flash("Please log in to access this page.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def index():
    if session.get("logged_in"):
        return redirect(url_for("list_bills"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        if username == "admin" and password == "admin":
            session["logged_in"] = True
            session["user"] = "admin"
            flash("Logged in successfully.", "success")
            return redirect(url_for("list_bills"))
        else:
            return render_template("login.html", error="Invalid username or password. Please use admin/admin.")
            
    if session.get("logged_in"):
        return redirect(url_for("list_bills"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

@app.route("/bills", methods=["GET"])
@login_required
def list_bills():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, vendor, invoice_no, amount, due_date, status, created_at FROM bills ORDER BY id ASC")
    bills = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return render_template("bills.html", bills=bills)

@app.route("/bills/new", methods=["GET", "POST"])
@login_required
def new_bill():
    form_data = {}
    if request.method == "POST":
        vendor = request.form.get("vendor", "").strip()
        invoice_no = request.form.get("invoice_no", "").strip()
        amount_raw = request.form.get("amount", "").strip().replace("$", "").replace(",", "")
        due_date = request.form.get("due_date", "").strip()
        
        form_data = {
            "vendor": vendor,
            "invoice_no": invoice_no,
            "amount": amount_raw,
            "due_date": due_date,
        }

        # 1. Validation for empty fields
        if not vendor or not invoice_no or not amount_raw or not due_date:
            return render_template(
                "bill_new.html",
                form_data=form_data,
                error="All fields are required. Please fill in all fields.",
            )

        # 2. Strict Date format validation (only YYYY-MM-DD)
        if not DATE_REGEX.match(due_date):
            return render_template(
                "bill_new.html",
                form_data=form_data,
                error=f"Invalid date format: '{due_date}'. Due date must strictly be YYYY-MM-DD (e.g. 2026-10-25).",
            )
        try:
            datetime.strptime(due_date, "%Y-%m-%d")
        except ValueError:
            return render_template(
                "bill_new.html",
                form_data=form_data,
                error=f"Invalid calendar date: '{due_date}'. Please enter a valid date in YYYY-MM-DD format.",
            )

        # 3. Amount validation
        try:
            amount = float(amount_raw)
            if amount <= 0:
                return render_template(
                    "bill_new.html",
                    form_data=form_data,
                    error="Amount must be a positive number greater than 0.",
                )
        except ValueError:
            return render_template(
                "bill_new.html",
                form_data=form_data,
                error=f"Invalid amount value: '{amount_raw}'. Amount must be a valid number.",
            )

        # 4. Duplicate invoice_no check
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM bills WHERE invoice_no = ?", (invoice_no,))
        existing = cursor.fetchone()
        if existing:
            conn.close()
            return render_template(
                "bill_new.html",
                form_data=form_data,
                error=f"Invoice number '{invoice_no}' already exists in the system. Duplicate invoices are rejected.",
            )

        # 5. Insert new bill (supports FAULT_SILENT_SAVE fault injection)
        if os.getenv("FAULT_SILENT_SAVE") == "1":
            conn.close()
            flash(f"Bill #999 for {vendor} created successfully.", "success")
            return redirect(url_for("list_bills"))

        try:
            cursor.execute(
                """
                INSERT INTO bills (vendor, invoice_no, amount, due_date, status)
                VALUES (?, ?, ?, ?, 'pending')
                """,
                (vendor, invoice_no, amount, due_date),
            )
            conn.commit()
            new_id = cursor.lastrowid
        finally:
            conn.close()

        flash(f"Bill #{new_id} for {vendor} created successfully.", "success")
        return redirect(url_for("list_bills"))

    return render_template("bill_new.html", form_data=form_data)

@app.route("/bills/<int:bill_id>", methods=["GET"])
@login_required
def view_bill(bill_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, vendor, invoice_no, amount, due_date, status, created_at FROM bills WHERE id = ?", (bill_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        flash(f"Bill #{bill_id} not found.", "error")
        return redirect(url_for("list_bills"))
    return render_template("bill_detail.html", bill=dict(row))

@app.route("/bills/<int:bill_id>/pay", methods=["POST"])
@login_required
def mark_bill_paid(bill_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, status FROM bills WHERE id = ?", (bill_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        flash(f"Bill #{bill_id} not found.", "error")
        return redirect(url_for("list_bills"))
    
    cursor.execute("UPDATE bills SET status = 'paid' WHERE id = ?", (bill_id,))
    conn.commit()
    conn.close()
    flash(f"Bill #{bill_id} has been marked as paid.", "success")
    return redirect(url_for("view_bill", bill_id=bill_id))

@app.route("/bills/delete_all", methods=["POST"])
@login_required
def delete_all_bills():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM bills")
    conn.commit()
    conn.close()
    flash("All bills have been deleted.", "success")
    return redirect(url_for("list_bills"))

if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
