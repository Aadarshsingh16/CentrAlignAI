import tempfile
from pathlib import Path
import pytest
from agent.browser import BrowserSession
from agent.policy import Policy, make_approval_guard
from agent.state import AgentState
from agent.tools import Tool, ToolRegistry

SAMPLE_FORM_HTML = """
<!DOCTYPE html>
<html>
<head><title>Policy Test Page</title></head>
<body>
    <form id="test-form">
        <label for="username">User Name</label>
        <input type="text" id="username" name="username" value="test_user" />

        <label for="role-select">User Role</label>
        <select id="role-select" name="role">
            <option value="admin" selected>Admin</option>
            <option value="viewer">Viewer</option>
        </select>

        <a href="#cancel" id="btn-cancel">Cancel</a>
        <button type="button" id="btn-save">Save Settings</button>
        <button type="button" id="btn-delete">Delete Account</button>
        <button type="button" id="btn-export">Export Records</button>
    </form>
</body>
</html>
"""

@pytest.fixture
def local_page(tmp_path):
    html_file = tmp_path / "page.html"
    html_file.write_text(SAMPLE_FORM_HTML, encoding="utf-8")
    return html_file.as_uri()

@pytest.fixture
def browser():
    session = BrowserSession(headless=True)
    yield session
    session.close()

def test_non_risky_click_runs_without_asking(browser, local_page):
    browser.goto(local_page)
    snap = browser.snapshot()

    cancel_el = next(el for el in snap["elements"] if "cancel" in el["label"].lower())

    state = AgentState()
    policy = Policy()
    approval_called = []

    def fake_approval(context):
        approval_called.append(context)
        return {"approved": True, "note": ""}

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="browser_click",
            description="Click element",
            parameters={"type": "object", "properties": {"id": {"type": "string"}}},
            fn=lambda id: browser.click(id),
        )
    )
    guard = make_approval_guard(browser, policy, fake_approval, state)
    registry.guards.append(guard)

    res = registry.execute("browser_click", {"id": cancel_el["id"]})
    assert res["ok"] is True
    assert len(approval_called) == 0  # Not risky, no approval prompt
    assert state.approvals == 0
    assert state.denials == 0

def test_risky_click_asks_and_runs_on_approval(browser, local_page):
    browser.goto(local_page)
    snap = browser.snapshot()

    save_el = next(el for el in snap["elements"] if "save settings" in el["label"].lower())

    state = AgentState()
    policy = Policy()
    approval_called = []

    def fake_approval(context):
        approval_called.append(context)
        return {"approved": True, "note": "Authorized by admin"}

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="browser_click",
            description="Click element",
            parameters={"type": "object", "properties": {"id": {"type": "string"}}},
            fn=lambda id: browser.click(id),
        )
    )
    guard = make_approval_guard(browser, policy, fake_approval, state)
    registry.guards.append(guard)

    res = registry.execute("browser_click", {"id": save_el["id"]})
    assert res["ok"] is True
    assert len(approval_called) == 1
    assert "save settings" in approval_called[0]["label"].lower()
    assert state.approvals == 1
    assert state.denials == 0

def test_denial_blocks_and_returns_note(browser, local_page):
    browser.goto(local_page)
    snap = browser.snapshot()

    delete_el = next(el for el in snap["elements"] if "delete account" in el["label"].lower())

    state = AgentState()
    policy = Policy()

    def fake_deny(context):
        return {"approved": False, "note": "Deletion forbidden"}

    clicked = []
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="browser_click",
            description="Click element",
            parameters={"type": "object", "properties": {"id": {"type": "string"}}},
            fn=lambda id: clicked.append(id) or {"ok": True, "observation": "clicked"},
        )
    )
    guard = make_approval_guard(browser, policy, fake_deny, state)
    registry.guards.append(guard)

    res = registry.execute("browser_click", {"id": delete_el["id"]})
    assert res["ok"] is False
    assert "denied by user: Deletion forbidden" in res["error"]
    assert len(clicked) == 0  # Ensure tool was NOT executed
    assert state.denials == 1
    assert state.approvals == 0

def test_guard_shows_form_values(browser, local_page):
    browser.goto(local_page)
    snap = browser.snapshot()

    save_el = next(el for el in snap["elements"] if "save settings" in el["label"].lower())

    captured_context = []

    def check_context(context):
        captured_context.append(context)
        return {"approved": True, "note": ""}

    state = AgentState()
    policy = Policy()
    guard = make_approval_guard(browser, policy, check_context, state)

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="browser_click",
            description="Click",
            parameters={"type": "object", "properties": {"id": {"type": "string"}}},
            fn=lambda id: {"ok": True, "observation": "clicked"},
        )
    )
    registry.guards.append(guard)

    registry.execute("browser_click", {"id": save_el["id"]})

    assert len(captured_context) == 1
    form_vals = captured_context[0]["form_values"]
    assert form_vals.get("User Name") == "test_user"
    assert form_vals.get("User Role") == "Admin"

def test_element_ids_unchanged_after_guard_check(browser, local_page):
    browser.goto(local_page)
    snap_before = browser.snapshot()
    ids_before = [el["id"] for el in snap_before["elements"]]

    save_el = next(el for el in snap_before["elements"] if "save settings" in el["label"].lower())

    state = AgentState()
    policy = Policy()
    guard = make_approval_guard(browser, policy, lambda ctx: {"approved": True, "note": ""}, state)

    # Run guard check without taking a new snapshot
    guard_res = guard("browser_click", {"id": save_el["id"]})
    assert guard_res is None  # approved

    # Check that data-agent-id attributes in the DOM are still intact
    page = browser.page
    tagged_ids = page.evaluate("() => Array.from(document.querySelectorAll('[data-agent-id]')).map(el => el.getAttribute('data-agent-id'))")
    assert tagged_ids == ids_before

def test_custom_policy_rule_without_code_changes(tmp_path, browser, local_page):
    # Create custom policy YAML with a rule targeting 'export'
    custom_yaml = tmp_path / "custom_policy.yaml"
    custom_yaml.write_text("rules:\n  - '(?i)export'\n", encoding="utf-8")

    policy = Policy(config_path=custom_yaml)
    assert len(policy.rules) == 1

    browser.goto(local_page)
    snap = browser.snapshot()
    export_el = next(el for el in snap["elements"] if "export records" in el["label"].lower())

    asked = []
    guard = make_approval_guard(
        browser,
        policy,
        lambda ctx: asked.append(ctx) or {"approved": True, "note": "custom approved"},
        AgentState(),
    )

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="browser_click",
            description="Click",
            parameters={"type": "object", "properties": {"id": {"type": "string"}}},
            fn=lambda id: {"ok": True, "observation": "clicked"},
        )
    )
    registry.guards.append(guard)

    res = registry.execute("browser_click", {"id": export_el["id"]})
    assert res["ok"] is True
    assert len(asked) == 1
    assert "export records" in asked[0]["label"].lower()
