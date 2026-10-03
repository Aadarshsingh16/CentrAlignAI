from pathlib import Path
import pytest
from agent.state import AgentState
from agent.trace import Trace
from agent.tools import (
    Tool,
    ToolRegistry,
    build_default_registry,
    format_snapshot,
)

def test_path_traversal_blocked(tmp_path):
    # Setup dummy workspace
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("classified data", encoding="utf-8")

    dummy_file = workspace / "public.txt"
    dummy_file.write_text("public data", encoding="utf-8")

    state = AgentState()
    registry = build_default_registry(browser=None, state=state, workspace_dir=workspace)

    # 1. list_files path traversal
    res_list = registry.execute("list_files", {"dir": "../"})
    assert res_list["ok"] is False
    assert "access denied" in res_list["error"].lower()

    # 2. read_file path traversal
    res_read_traversal = registry.execute("read_file", {"path": "../secret.txt"})
    assert res_read_traversal["ok"] is False
    assert "access denied" in res_read_traversal["error"].lower()

    # 3. read_file valid access
    res_read_valid = registry.execute("read_file", {"path": "public.txt"})
    assert res_read_valid["ok"] is True
    assert "public data" in res_read_valid["observation"]

def test_unknown_tool_and_bad_args(tmp_path):
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"required_arg": {"type": "string"}},
                "required": ["required_arg"],
            },
            fn=lambda required_arg: f"result: {required_arg}",
        )
    )

    # Unknown tool
    res_unknown = registry.execute("non_existent_tool", {})
    assert res_unknown["ok"] is False
    assert "Unknown tool 'non_existent_tool'" in res_unknown["error"]
    assert "test_tool" in res_unknown["error"]

    # Missing required argument
    res_missing = registry.execute("test_tool", {})
    assert res_missing["ok"] is False
    assert "Missing required argument" in res_missing["error"]
    assert "required_arg" in res_missing["error"]

def test_tool_that_raises_returns_error():
    registry = ToolRegistry()
    def failing_fn():
        raise RuntimeError("Controlled explosion")

    registry.register(
        Tool(
            name="failing_tool",
            description="A failing tool",
            parameters={"type": "object", "properties": {}},
            fn=failing_fn,
        )
    )

    res = registry.execute("failing_tool", {})
    assert res["ok"] is False
    assert "Controlled explosion" in res["error"]
    assert "Tool execution error" in res["observation"]

def test_guard_can_block():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="guarded_tool",
            description="Guarded tool",
            parameters={"type": "object", "properties": {}},
            fn=lambda: "ran successfully",
        )
    )

    def security_guard(name, args):
        if name == "guarded_tool":
            return {"block": "Action denied by security policy"}
        return None

    registry.guards.append(security_guard)

    res = registry.execute("guarded_tool", {})
    assert res["ok"] is False
    assert "Action denied by security policy" in res["error"]
    assert "Blocked by policy" in res["observation"]

def test_remember_updates_state(tmp_path):
    state = AgentState()
    registry = build_default_registry(browser=None, state=state, workspace_dir=tmp_path)

    res = registry.execute("remember", {"key": "favorite_color", "value": "indigo"})
    assert res["ok"] is True
    assert "indigo" in res["observation"]
    assert state.facts.get("favorite_color") == "indigo"

    prompt_text = state.to_prompt()
    assert "favorite_color: indigo" in prompt_text

    state_dict = state.to_dict()
    assert state_dict["facts"]["favorite_color"] == "indigo"

def test_trace_read_and_write(tmp_path):
    trace_file = tmp_path / "test_trace.jsonl"
    trace = Trace(trace_file)

    rec1 = trace.append(
        step=1,
        thought="Need to check directory",
        action="list_files",
        args={"dir": "."},
        observation="file1.txt\nfile2.txt",
        state={"facts": {}, "steps_done": ["initialised"], "open_questions": []},
    )
    assert rec1["step"] == 1

    rec2 = trace.append(
        step=2,
        thought="Done",
        action="finish",
        args={"claim": "Completed task"},
        observation="Finished successfully",
        state={"facts": {"done": True}, "steps_done": ["initialised", "finished"], "open_questions": []},
    )
    assert rec2["step"] == 2

    # Read back
    records = Trace.read(trace_file)
    assert len(records) == 2
    assert records[0]["action"] == "list_files"
    assert records[1]["action"] == "finish"
    assert records[1]["thought"] == "Done"
    assert records[1]["state"]["facts"]["done"] is True

def test_ask_user_returns_scripted_answer(tmp_path):
    scripted_answers = {
        "Please specify your department": "Engineering"
    }

    def mock_input(prompt: str) -> str:
        return scripted_answers.get(prompt, "Default Answer")

    state = AgentState()
    registry = build_default_registry(
        browser=None,
        state=state,
        workspace_dir=tmp_path,
        input_fn=mock_input,
    )

    res = registry.execute("ask_user", {"question": "Please specify your department"})
    assert res["ok"] is True
    assert "User answer: Engineering" in res["observation"]

def test_format_snapshot():
    sample_snapshot = {
        "url": "http://localhost:5000/form",
        "title": "Registration Form",
        "alerts": ["Validation Error: Invalid code", "Session expiring soon"],
        "elements": [
            {"id": "1", "tag": "input", "role": "textbox", "label": "Full Name", "value": "", "disabled": False},
            {"id": "2", "tag": "button", "role": "button", "label": "Submit", "value": "", "disabled": True},
        ],
        "text": "Please register your details carefully before submitting.",
    }

    rendered = format_snapshot(sample_snapshot)
    # Check alert text appears
    assert "Validation Error: Invalid code" in rendered
    assert "Session expiring soon" in rendered
    assert "[ALERT]" in rendered

    # Check element IDs and details appear
    assert "[1]" in rendered
    assert "Full Name" in rendered
    assert "[2]" in rendered
    assert "Submit" in rendered
    assert "(disabled)" in rendered

    # Check URL and title
    assert "http://localhost:5000/form" in rendered
    assert "Registration Form" in rendered
