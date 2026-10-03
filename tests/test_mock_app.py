import os
import re
from pathlib import Path
import pytest
from mock_app.app import app
from mock_app.seed import reset_db, get_all_bills, get_db_path, SEED_BILLS

@pytest.fixture(autouse=True)
def clean_db():
    reset_db()
    yield
    reset_db()

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_reset_db_is_repeatable():
    # Initial state
    bills_initial = get_all_bills()
    assert len(bills_initial) == len(SEED_BILLS)
    assert bills_initial[0]["vendor"] == "Globex"
    assert bills_initial[2]["invoice_no"] == "UMB-2026-001"

    # Mutate DB directly: add a custom row and modify a row
    db_path = get_db_path()
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO bills (vendor, invoice_no, amount, due_date, status) VALUES (?, ?, ?, ?, ?)",
        ("Test Corp", "TEST-999", 999.00, "2026-12-31", "pending"),
    )
    cursor.execute("UPDATE bills SET status = 'paid' WHERE invoice_no = 'UMB-2026-001'")
    conn.commit()
    conn.close()

    mutated_bills = get_all_bills()
    assert len(mutated_bills) == len(SEED_BILLS) + 1

    # Call reset_db() and verify exact restoration
    reset_db()
    restored_bills = get_all_bills()
    assert len(restored_bills) == len(SEED_BILLS)
    # Ensure test row is gone and Umbrella is back to pending
    umb = next(b for b in restored_bills if b["invoice_no"] == "UMB-2026-001")
    assert umb["status"] == "pending"

def test_login_flow(client):
    # Unauthenticated access to /bills redirects to /login
    resp = client.get("/bills", follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]

    # Invalid login credentials
    bad_login = client.post("/login", data={"username": "wrong", "password": "bad"}, follow_redirects=True)
    assert bad_login.status_code == 200
    html = bad_login.get_data(as_text=True)
    assert 'role="alert"' in html
    assert "Invalid username or password" in html

    # Valid login credentials
    good_login = client.post("/login", data={"username": "admin", "password": "admin"}, follow_redirects=True)
    assert good_login.status_code == 200
    html_good = good_login.get_data(as_text=True)
    assert "All Bills" in html_good
    assert 'id="bills-table"' in html_good

def test_new_bill_bad_date_error(client):
    # Log in
    client.post("/login", data={"username": "admin", "password": "admin"})

    # Attempt to submit DD/MM/YYYY date
    resp = client.post("/bills/new", data={
        "vendor": "Initech",
        "invoice_no": "INT-2026-404",
        "amount": "3400.00",
        "due_date": "25/11/2026",
    })
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert 'role="alert"' in html
    assert "Due date must strictly be YYYY-MM-DD" in html

    # Attempt to submit text format date
    resp_text_date = client.post("/bills/new", data={
        "vendor": "Acme Corporation",
        "invoice_no": "ACM-2026-202",
        "amount": "5200.00",
        "due_date": "15 Nov 2026",
    })
    assert resp_text_date.status_code == 200
    html_text = resp_text_date.get_data(as_text=True)
    assert 'role="alert"' in html_text
    assert "Due date must strictly be YYYY-MM-DD" in html_text

def test_new_bill_duplicate_invoice_error(client):
    # Log in
    client.post("/login", data={"username": "admin", "password": "admin"})

    # UMB-2026-001 already exists in seed data
    resp = client.post("/bills/new", data={
        "vendor": "Umbrella",
        "invoice_no": "UMB-2026-001",
        "amount": "1450.00",
        "due_date": "2026-08-15",
    })
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert 'role="alert"' in html
    assert "already exists in the system" in html
    assert "Duplicate invoices are rejected" in html

def test_new_bill_success_and_detail(client):
    # Log in
    client.post("/login", data={"username": "admin", "password": "admin"})

    # Submit valid bill
    resp = client.post("/bills/new", data={
        "vendor": "Globex",
        "invoice_no": "GLX-2026-099",
        "amount": "7850.00",
        "due_date": "2026-10-25",
    }, follow_redirects=True)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "GLX-2026-099" in html
    assert "$7850.00" in html

    # Verify directly via get_all_bills()
    all_bills = get_all_bills()
    created = next(b for b in all_bills if b["invoice_no"] == "GLX-2026-099")
    assert created["vendor"] == "Globex"
    assert created["amount"] == 7850.00
    assert created["due_date"] == "2026-10-25"
    assert created["status"] == "pending"

    # View detail page
    detail_resp = client.get(f"/bills/{created['id']}")
    assert detail_resp.status_code == 200
    detail_html = detail_resp.get_data(as_text=True)
    assert 'id="mark-paid-btn"' in detail_html

    # Mark as paid
    pay_resp = client.post(f"/bills/{created['id']}/pay", follow_redirects=True)
    assert pay_resp.status_code == 200
    pay_html = pay_resp.get_data(as_text=True)
    assert "has been marked as paid" in pay_html

    # Verify updated status
    updated = next(b for b in get_all_bills() if b["invoice_no"] == "GLX-2026-099")
    assert updated["status"] == "paid"

def test_invoices_directory_and_files():
    invoices_dir = Path(__file__).parent.parent / "invoices"
    assert invoices_dir.exists()
    files = list(invoices_dir.glob("*.txt"))
    assert len(files) >= 8

    # Verify Globex has 2 invoices for comparison
    globex_files = [f for f in files if "globex" in f.name.lower()]
    assert len(globex_files) >= 2

    # Verify Umbrella duplicate invoice file exists
    umbrella_file = next(f for f in files if "umbrella" in f.name.lower())
    assert "UMB-2026-001" in umbrella_file.read_text(encoding="utf-8")

    # Verify Cyberdyne anomalous invoice exists
    cyber_file = next(f for f in files if "cyberdyne" in f.name.lower())
    content = cyber_file.read_text(encoding="utf-8")
    assert "95,000" in content

    # Verify Acme ambiguous vendor invoice exists
    acme_file = next(f for f in files if f.name == "acme_2026_10_15.txt")
    acme_content = acme_file.read_text(encoding="utf-8")
    assert "15 Oct 2026" in acme_content
