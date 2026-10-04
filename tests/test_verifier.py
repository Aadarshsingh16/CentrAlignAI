import copy
import pytest
from typing import Any, Optional
from agent.env import Environment
from agent.llm import LLMResponse
from agent.loop import run_agent, AgentResult
from agent.state import AgentState
from agent.tools import Tool, ToolRegistry, build_default_registry
from agent.trace import Trace
from agent.verifier import compute_diff, verify, Verdict, VERIFIER_SYSTEM_PROMPT
from mock_app.app import app
from mock_app.seed import reset_db, get_all_bills, DEFAULT_DB_PATH
from mock_app.env import MockAppEnvironment

class FakeLLM:
    """Scripted LLM for unit tests without network calls."""

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.history_received = []
        self.systems_received = []

    def chat(self, system: str, messages: list[dict], tool_schemas: list[dict]) -> LLMResponse:
        self.systems_received.append(system)
        self.history_received.append(copy.deepcopy(messages))
        if not self.script:
            return LLMResponse(text="", tool_call=None, raw=None)
        next_call = self.script.pop(0)
        return LLMResponse(
            text=next_call.get("text", "thinking"),
            tool_call=next_call.get("tool_call"),
            raw=None,
        )

class FakeEnvironment(Environment):
    """Configurable environment for testing."""
    name: str = "fake_env"

    def __init__(self, initial_state: Any):
        self.state = copy.deepcopy(initial_state)

    def ground_truth(self) -> Any:
        return copy.deepcopy(self.state)

def test_compute_diff_added_changed_removed():
    # 1. Added
    before = [{"id": 1, "title": "A", "val": 10}]
    after = [{"id": 1, "title": "A", "val": 10}, {"id": 2, "title": "B", "val": 20}]
    diff = compute_diff(before, after)
    assert len(diff["added"]) == 1
    assert diff["added"][0]["id"] == 2
    assert diff["removed"] == []
    assert diff["changed"] == []

    # 2. Removed
    before = [{"id": 1, "title": "A", "val": 10}, {"id": 2, "title": "B", "val": 20}]
    after = [{"id": 1, "title": "A", "val": 10}]
    diff = compute_diff(before, after)
    assert diff["added"] == []
    assert len(diff["removed"]) == 1
    assert diff["removed"][0]["id"] == 2
    assert diff["changed"] == []

    # 3. Changed
    before = [{"id": 1, "title": "A", "val": 10}]
    after = [{"id": 1, "title": "A", "val": 99}]
    diff = compute_diff(before, after)
    assert diff["added"] == []
    assert diff["removed"] == []
    assert len(diff["changed"]) == 1
    assert diff["changed"][0]["id"] == 1
    assert diff["changed"][0]["changes"]["val"] == {"old": 10, "new": 99}
    assert diff["changed"][0]["val"] == {"old": 10, "new": 99}

    # 4. Combined: added, removed, changed
    before = [
        {"id": 1, "title": "Item 1", "status": "draft"},
        {"id": 2, "title": "Item 2", "status": "active"},
    ]
    after = [
        {"id": 1, "title": "Item 1", "status": "published"},
        {"id": 3, "title": "Item 3", "status": "active"},
    ]
    diff = compute_diff(before, after)
    assert len(diff["added"]) == 1
    assert diff["added"][0]["id"] == 3
    assert len(diff["removed"]) == 1
    assert diff["removed"][0]["id"] == 2
    assert len(diff["changed"]) == 1
    assert diff["changed"][0]["id"] == 1
    assert diff["changed"][0]["changes"]["status"] == {"old": "draft", "new": "published"}

def test_compute_diff_generic_dict():
    before = {"a": 1, "nested": {"x": 10}}
    after = {"a": 1, "b": 2, "nested": {"x": 20}}
    diff = compute_diff(before, after)
    assert diff["added"] == {"b": 2}
    assert "nested" in diff["changed"]

def test_verifier_passes_when_diff_matches_claim():
    llm = FakeLLM([
        {
            "tool_call": {
                "name": "report_verdict",
                "args": {
                    "achieved": True,
                    "claim_accurate": True,
                    "reasons": "Ground-truth diff shows 1 item added matching the claim.",
                },
            }
        }
    ])

    before = [{"id": 1, "code": "A"}]
    after = [{"id": 1, "code": "A"}, {"id": 2, "code": "B"}]
    diff = compute_diff(before, after)

    verdict = verify(
        task="Create item B",
        claim="Created item B with code B",
        evidence="Observed confirmation message",
        before=before,
        after=after,
        diff=diff,
        run_summary={"approvals": 1, "denials": 0, "asks": 0, "denial_notes": []},
        llm=llm,
    )

    assert verdict.achieved is True
    assert verdict.claim_accurate is True
    assert "matching the claim" in verdict.reasons
    assert len(llm.history_received) == 1
    # Verify fresh call with no prior history and correct verifier system prompt
    assert llm.systems_received[0] == VERIFIER_SYSTEM_PROMPT

def test_verifier_flags_claim_unsupported_by_diff():
    llm = FakeLLM([
        {
            "tool_call": {
                "name": "report_verdict",
                "args": {
                    "achieved": False,
                    "claim_accurate": False,
                    "reasons": "Agent claims record was created, but ground-truth diff is empty.",
                },
            }
        }
    ])

    before = [{"id": 1, "code": "A"}]
    after = [{"id": 1, "code": "A"}]
    diff = compute_diff(before, after)

    verdict = verify(
        task="Create item B",
        claim="Successfully created item B",
        evidence="Form submitted",
        before=before,
        after=after,
        diff=diff,
        run_summary={"approvals": 1, "denials": 0, "asks": 0, "denial_notes": []},
        llm=llm,
    )

    assert verdict.achieved is False
    assert verdict.claim_accurate is False
    assert "ground-truth diff is empty" in verdict.reasons

def test_verifier_user_denied_action_honest_claim():
    llm = FakeLLM([
        {
            "tool_call": {
                "name": "report_verdict",
                "args": {
                    "achieved": False,
                    "claim_accurate": True,
                    "reasons": "Agent honestly reported that action was denied by user; diff reflects no changes.",
                },
            }
        }
    ])

    before = [{"id": 1, "code": "A"}]
    after = [{"id": 1, "code": "A"}]
    diff = compute_diff(before, after)

    verdict = verify(
        task="Delete item A",
        claim="Operation was denied by user; no items were deleted.",
        evidence="User clicked deny on confirmation",
        before=before,
        after=after,
        diff=diff,
        run_summary={"approvals": 0, "denials": 1, "asks": 0, "denial_notes": ["User denied deletion"]},
        llm=llm,
    )

    assert verdict.achieved is False
    assert verdict.claim_accurate is True
    assert "honestly reported" in verdict.reasons

def test_loop_feeds_mismatch_back_and_continues(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace = Trace(trace_path)
    state = AgentState()
    registry = build_default_registry(browser=None, state=state, workspace_dir=tmp_path)

    env = FakeEnvironment([{"id": 1, "val": "initial"}])

    # Script:
    # 1. Agent calls finish (prematurely)
    # 2. Verifier called -> returns mismatch (achieved=False, claim_accurate=False)
    # 3. Agent receives observation of mismatch and calls remember
    # 4. Agent calls finish (2nd attempt)
    # 5. Verifier called -> returns pass (achieved=True, claim_accurate=True)
    script = [
        # 1. Agent turn: premature finish
        {
            "tool_call": {
                "name": "finish",
                "args": {"thought": "Claiming early completion", "claim": "Done", "evidence": "None"},
            }
        },
        # 2. Verifier LLM call: mismatch
        {
            "tool_call": {
                "name": "report_verdict",
                "args": {
                    "achieved": False,
                    "claim_accurate": False,
                    "reasons": "Diff is empty; nothing was accomplished.",
                },
            }
        },
        # 3. Agent turn after getting mismatch observation: adapts and records fact
        {
            "tool_call": {
                "name": "remember",
                "args": {"thought": "Learning from mismatch", "key": "retry", "value": "true"},
            }
        },
        # 4. Agent turn: finish again
        {
            "tool_call": {
                "name": "finish",
                "args": {"thought": "Retrying finish with proper claim", "claim": "Done accurately", "evidence": "Recorded"},
            }
        },
        # 5. Verifier LLM call: pass
        {
            "tool_call": {
                "name": "report_verdict",
                "args": {
                    "achieved": True,
                    "claim_accurate": True,
                    "reasons": "Ground-truth matches updated claim.",
                },
            }
        },
    ]

    llm = FakeLLM(script)
    res = run_agent(
        task="Test recovery task",
        registry=registry,
        llm=llm,
        state=state,
        trace=trace,
        env=env,
        max_steps=10,
    )

    assert res.status == "finished"
    assert res.verified is True
    assert res.verdict is not None
    assert res.verdict["achieved"] is True
    assert res.verdict["claim_accurate"] is True
    assert res.steps == 3  # finish(failed), remember, finish(passed)

    # Check trace contains the mismatch observation
    records = Trace.read(trace_path)
    assert any("Verification mismatch (round 1/2)" in r.get("observation", "") for r in records)

def test_loop_stops_after_two_failed_rounds(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace = Trace(trace_path)
    state = AgentState()
    registry = build_default_registry(browser=None, state=state, workspace_dir=tmp_path)

    env = FakeEnvironment([{"id": 1, "val": "initial"}])

    # Script:
    # Round 1 finish -> verifier fails
    # Round 2 finish -> verifier fails again -> loop aborts and reports verified=False
    script = [
        # Agent finish 1
        {
            "tool_call": {
                "name": "finish",
                "args": {"thought": "Attempt 1", "claim": "Done 1", "evidence": "None"},
            }
        },
        # Verifier 1 (fail)
        {
            "tool_call": {
                "name": "report_verdict",
                "args": {
                    "achieved": False,
                    "claim_accurate": False,
                    "reasons": "Round 1 failure: diff empty",
                },
            }
        },
        # Agent finish 2
        {
            "tool_call": {
                "name": "finish",
                "args": {"thought": "Attempt 2", "claim": "Done 2", "evidence": "None"},
            }
        },
        # Verifier 2 (fail)
        {
            "tool_call": {
                "name": "report_verdict",
                "args": {
                    "achieved": False,
                    "claim_accurate": False,
                    "reasons": "Round 2 failure: diff still empty",
                },
            }
        },
    ]

    llm = FakeLLM(script)
    res = run_agent(
        task="Test two failed rounds",
        registry=registry,
        llm=llm,
        state=state,
        trace=trace,
        env=env,
        max_steps=10,
    )

    assert res.status == "finished"
    assert res.verified is False
    assert res.verdict is not None
    assert res.verdict["achieved"] is False
    assert res.verdict["claim_accurate"] is False
    assert "Round 2 failure" in res.verdict["reasons"]

def test_fault_injection_prevents_db_write(monkeypatch, tmp_path):
    # Setup test DB
    test_db = tmp_path / "test_bills.db"
    reset_db(test_db)
    monkeypatch.setenv("MOCK_APP_DB_PATH", str(test_db))

    client = app.test_client()

    # Login first
    client.post("/login", data={"username": "admin", "password": "admin"}, follow_redirects=True)

    bills_before = get_all_bills(test_db)
    initial_count = len(bills_before)

    # 1. Direct call with FAULT_SILENT_SAVE=1
    monkeypatch.setenv("FAULT_SILENT_SAVE", "1")

    post_data = {
        "vendor": "Globex",
        "invoice_no": "GLX-FAULT-TEST-999",
        "amount": "1250.00",
        "due_date": "2026-11-20",
    }
    resp = client.post("/bills/new", data=post_data, follow_redirects=True)
    assert resp.status_code == 200

    # Ensure flash success is visible to user
    assert b"created successfully" in resp.data

    # Verify no row was inserted into SQLite
    bills_after = get_all_bills(test_db)
    assert len(bills_after) == initial_count
    assert not any(b["invoice_no"] == "GLX-FAULT-TEST-999" for b in bills_after)

    # 2. Turn off fault injection -> row should actually be written
    monkeypatch.setenv("FAULT_SILENT_SAVE", "0")
    resp2 = client.post("/bills/new", data=post_data, follow_redirects=True)
    assert resp2.status_code == 200
    assert b"created successfully" in resp2.data

    bills_normal = get_all_bills(test_db)
    assert len(bills_normal) == initial_count + 1
    assert any(b["invoice_no"] == "GLX-FAULT-TEST-999" for b in bills_normal)

def test_mock_app_environment(tmp_path, monkeypatch):
    test_db = tmp_path / "mock_env.db"
    reset_db(test_db)
    env = MockAppEnvironment(db_path=str(test_db))

    ground_truth = env.ground_truth()
    assert isinstance(ground_truth, list)
    assert len(ground_truth) == 6
    assert ground_truth[0]["vendor"] == "Globex"
