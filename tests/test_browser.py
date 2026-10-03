import threading
import time
import socket
import pytest
import requests
from werkzeug.serving import make_server
from mock_app.app import app
from mock_app.seed import reset_db
from agent.browser import BrowserSession

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

class ServerThread(threading.Thread):
    def __init__(self, app, port):
        super().__init__(daemon=True)
        self.port = port
        self.server = make_server("127.0.0.1", port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

@pytest.fixture(scope="module")
def app_server():
    reset_db()
    port = get_free_port()
    server_thread = ServerThread(app, port)
    server_thread.start()

    base_url = f"http://127.0.0.1:{port}"
    # Wait for server to be responsive
    for _ in range(50):
        try:
            r = requests.get(f"{base_url}/login", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)

    yield base_url
    server_thread.shutdown()

@pytest.fixture
def browser():
    session = BrowserSession(headless=True)
    yield session
    session.close()

def find_element(elements: list[dict], *, label_substr: str = "", tag: str = "", role: str = "") -> dict:
    for el in elements:
        label_match = label_substr.lower() in el.get("label", "").lower() if label_substr else True
        tag_match = el.get("tag", "").lower() == tag.lower() if tag else True
        role_match = el.get("role", "").lower() == role.lower() if role else True
        if label_match and tag_match and role_match:
            return el
    raise AssertionError(f"Element not found with label_substr='{label_substr}', tag='{tag}', role='{role}'. Available: {elements}")

def test_browser_flow_with_bad_date_and_stale_ids(app_server, browser):
    # 1. Navigate to login
    nav_res = browser.goto(f"{app_server}/login")
    assert nav_res["ok"] is True

    # 2. Snapshot login page
    snap1 = browser.snapshot()
    assert snap1["ok"] is True
    assert "User Login" in snap1["text"]

    username_el = find_element(snap1["elements"], label_substr="username")
    password_el = find_element(snap1["elements"], label_substr="password")
    login_btn = find_element(snap1["elements"], label_substr="log in", tag="button")

    # 3. Enter credentials and log in
    assert browser.type(username_el["id"], "admin")["ok"] is True
    assert browser.type(password_el["id"], "admin")["ok"] is True
    assert browser.click(login_btn["id"])["ok"] is True

    # 4. Snapshot landing page and navigate to new form
    snap2 = browser.snapshot()
    assert snap2["ok"] is True
    assert "All Bills" in snap2["text"]

    new_btn = find_element(snap2["elements"], label_substr="new bill")
    assert browser.click(new_btn["id"])["ok"] is True

    # 5. Snapshot form page
    snap3 = browser.snapshot()
    assert snap3["ok"] is True
    assert "Enter New Bill" in snap3["text"]

    vendor_el = find_element(snap3["elements"], label_substr="vendor", tag="select")
    inv_el = find_element(snap3["elements"], label_substr="invoice number")
    amount_el = find_element(snap3["elements"], label_substr="amount")
    date_el = find_element(snap3["elements"], label_substr="due date")
    save_btn = find_element(snap3["elements"], label_substr="save bill", tag="button")

    # Describe test
    desc = browser.describe(save_btn["id"])
    assert desc["ok"] is True
    assert desc["element"]["tag"] == "button"
    assert "save bill" in desc["element"]["label"].lower()

    # Fill form with bad date format (DD/MM/YYYY)
    assert browser.select(vendor_el["id"], "Initech")["ok"] is True
    assert browser.type(inv_el["id"], "TEST-BAD-DATE-001")["ok"] is True
    assert browser.type(amount_el["id"], "1234.56")["ok"] is True
    assert browser.type(date_el["id"], "25/11/2026")["ok"] is True

    # Click save
    assert browser.click(save_btn["id"])["ok"] is True

    # 6. Verify error alert appears in the next snapshot
    snap4 = browser.snapshot()
    assert snap4["ok"] is True
    assert len(snap4["alerts"]) > 0
    assert "Due date must strictly be YYYY-MM-DD" in snap4["alerts"][0]

    # 7. Test stale and unknown ids return safe error responses without raising
    stale_click = browser.click("99999")
    assert stale_click["ok"] is False
    assert "stale" in stale_click["error"].lower() or "not found" in stale_click["error"].lower()

    stale_type = browser.type("99999", "sample")
    assert stale_type["ok"] is False
    assert "stale" in stale_type["error"].lower() or "not found" in stale_type["error"].lower()

    stale_select = browser.select("99999", "Initech")
    assert stale_select["ok"] is False
    assert "stale" in stale_select["error"].lower() or "not found" in stale_select["error"].lower()

    stale_describe = browser.describe("99999")
    assert stale_describe["ok"] is False
    assert "stale" in stale_describe["error"].lower() or "not found" in stale_describe["error"].lower()
