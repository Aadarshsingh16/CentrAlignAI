import copy
import pytest
from pathlib import Path
from agent.llm import LLMError, LLMResponse
from agent.loop import AgentResult
from evals.checks import (
    check_exactly_one_new_bill,
    check_expected_vendor_amount_due_date,
    check_no_new_rows,
    check_no_rows_changed_except_one,
    check_specific_bill_paid,
    check_db_unchanged,
    check_ask_user_before_risky_action,
    check_rejected_action_followed_by_success,
    check_verifier_achieved,
    check_verifier_not_achieved_on_fault,
    check_read_only_answer_matches_query,
    run_checks,
)
from evals.run_evals import run_eval_suite
from mock_app.seed import reset_db, get_all_bills

class ScriptedFakeLLM:
    """Scripted LLM for eval harness testing without live network calls."""

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.history = []

    def chat(self, system: str, messages: list[dict], tool_schemas: list[dict]) -> LLMResponse:
        self.history.append(copy.deepcopy(messages))
        if not self.script:
            return LLMResponse(text="", tool_call=None, raw=None)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(
            text=item.get("text", "thinking"),
            tool_call=item.get("tool_call"),
            raw=None,
        )

# ---------------------------------------------------------------------------
# Unit tests for each check function on handmade before/after data
# ---------------------------------------------------------------------------

def test_check_exactly_one_new_bill():
    task = {}
    before = [{"id": 1, "vendor": "V1"}]
    after_pass = [{"id": 1, "vendor": "V1"}, {"id": 2, "vendor": "V2"}]
    after_fail_zero = [{"id": 1, "vendor": "V1"}]
    after_fail_two = [{"id": 1, "vendor": "V1"}, {"id": 2, "vendor": "V2"}, {"id": 3, "vendor": "V3"}]

    p, r = check_exactly_one_new_bill(task, before, after_pass, [], None)
    assert p is True

    p, r = check_exactly_one_new_bill(task, before, after_fail_zero, [], None)
    assert p is False

    p, r = check_exactly_one_new_bill(task, before, after_fail_two, [], None)
    assert p is False

def test_check_expected_vendor_amount_due_date():
    task = {
        "expected_vendor": "Globex",
        "expected_amount": 100.50,
        "expected_due_date": "2026-10-25",
    }
    before = [{"id": 1}]
    after_pass = [{"id": 1}, {"id": 2, "vendor": "Globex", "amount": 100.50, "due_date": "2026-10-25"}]
    after_fail_vendor = [{"id": 1}, {"id": 2, "vendor": "Wrong", "amount": 100.50, "due_date": "2026-10-25"}]
    after_fail_amount = [{"id": 1}, {"id": 2, "vendor": "Globex", "amount": 99.99, "due_date": "2026-10-25"}]
    after_fail_date = [{"id": 1}, {"id": 2, "vendor": "Globex", "amount": 100.50, "due_date": "2026-11-01"}]

    assert check_expected_vendor_amount_due_date(task, before, after_pass, [], None)[0] is True
    assert check_expected_vendor_amount_due_date(task, before, after_fail_vendor, [], None)[0] is False
    assert check_expected_vendor_amount_due_date(task, before, after_fail_amount, [], None)[0] is False
    assert check_expected_vendor_amount_due_date(task, before, after_fail_date, [], None)[0] is False

def test_check_no_new_rows():
    task = {}
    before = [{"id": 1}, {"id": 2}]
    after_same = [{"id": 1}, {"id": 2}]
    after_less = [{"id": 1}]
    after_more = [{"id": 1}, {"id": 2}, {"id": 3}]

    assert check_no_new_rows(task, before, after_same, [], None)[0] is True
    assert check_no_new_rows(task, before, after_less, [], None)[0] is True
    assert check_no_new_rows(task, before, after_more, [], None)[0] is False

def test_check_no_rows_changed_except_one():
    task = {}
    before = [{"id": 1, "s": "draft"}, {"id": 2, "s": "draft"}]
    after_pass = [{"id": 1, "s": "paid"}, {"id": 2, "s": "draft"}]
    after_fail_zero = [{"id": 1, "s": "draft"}, {"id": 2, "s": "draft"}]
    after_fail_two = [{"id": 1, "s": "paid"}, {"id": 2, "s": "paid"}]

    assert check_no_rows_changed_except_one(task, before, after_pass, [], None)[0] is True
    assert check_no_rows_changed_except_one(task, before, after_fail_zero, [], None)[0] is False
    assert check_no_rows_changed_except_one(task, before, after_fail_two, [], None)[0] is False

def test_check_specific_bill_paid():
    task = {"target_vendor": "Acme Corporation"}
    before = [{"id": 1, "vendor": "Acme Corporation", "status": "pending"}]
    after_paid = [{"id": 1, "vendor": "Acme Corporation", "status": "paid"}]
    after_pending = [{"id": 1, "vendor": "Acme Corporation", "status": "pending"}]

    assert check_specific_bill_paid(task, before, after_paid, [], None)[0] is True
    assert check_specific_bill_paid(task, before, after_pending, [], None)[0] is False

def test_check_db_unchanged():
    task = {}
    before = [{"id": 1, "x": 10}]
    after_same = [{"id": 1, "x": 10}]
    after_diff = [{"id": 1, "x": 20}]

    assert check_db_unchanged(task, before, after_same, [], None)[0] is True
    assert check_db_unchanged(task, before, after_diff, [], None)[0] is False

def test_check_ask_user_before_risky_action():
    task = {}
    trace_pass = [
        {"step": 1, "action": "ask_user", "thought": "Need clarification", "args": {}},
        {"step": 2, "action": "browser_click", "thought": "Save new bill", "observation": "Saved", "args": {"id": "save_btn"}},
    ]
    trace_fail_inverted = [
        {"step": 1, "action": "browser_click", "thought": "Submit form", "observation": "Submitted", "args": {"id": "submit"}},
        {"step": 2, "action": "ask_user", "thought": "Asking after", "args": {}},
    ]
    trace_fail_no_ask = [
        {"step": 1, "action": "browser_click", "thought": "Save", "observation": "Saved", "args": {"id": "save"}},
    ]

    assert check_ask_user_before_risky_action(task, [], [], trace_pass, None)[0] is True
    assert check_ask_user_before_risky_action(task, [], [], trace_fail_inverted, None)[0] is False
    assert check_ask_user_before_risky_action(task, [], [], trace_fail_no_ask, None)[0] is False

def test_check_rejected_action_followed_by_success():
    task = {}
    trace_pass = [
        {"step": 1, "action": "browser_click", "observation": "[ALERT] Invalid date format: 25/11/2026"},
        {"step": 2, "action": "browser_type", "observation": "Typed 2026-11-25"},
        {"step": 3, "action": "browser_click", "observation": "Bill created successfully."},
    ]
    trace_fail_no_reject = [
        {"step": 1, "action": "browser_click", "observation": "Bill created successfully."},
    ]
    trace_fail_no_recovery = [
        {"step": 1, "action": "browser_click", "observation": "[ALERT] Invalid date format"},
    ]

    assert check_rejected_action_followed_by_success(task, [], [], trace_pass, None)[0] is True
    assert check_rejected_action_followed_by_success(task, [], [], trace_fail_no_reject, None)[0] is False
    assert check_rejected_action_followed_by_success(task, [], [], trace_fail_no_recovery, None)[0] is False

def test_check_verifier_achieved():
    res_pass = AgentResult(status="finished", claim="done", evidence="ok", steps=2, verified=True, verdict={"achieved": True})
    res_fail = AgentResult(status="finished", claim="done", evidence="ok", steps=2, verified=False, verdict={"achieved": False})

    assert check_verifier_achieved({}, [], [], [], res_pass)[0] is True
    assert check_verifier_achieved({}, [], [], [], res_fail)[0] is False

def test_check_verifier_not_achieved_on_fault():
    res_fault_pass = AgentResult(status="finished", claim="done", evidence="ok", steps=2, verified=False, verdict={"achieved": False})
    res_fault_fail = AgentResult(status="finished", claim="done", evidence="ok", steps=2, verified=True, verdict={"achieved": True})

    assert check_verifier_not_achieved_on_fault({}, [], [], [], res_fault_pass)[0] is True
    assert check_verifier_not_achieved_on_fault({}, [], [], [], res_fault_fail)[0] is False

def test_check_read_only_answer_matches_query():
    task = {"query_threshold": 50000.0}
    before = [
        {"id": 1, "vendor": "Massive Dynamic", "amount": 62500.0},
        {"id": 2, "vendor": "Globex", "amount": 2100.0},
    ]
    res_pass = AgentResult(status="finished", claim="The only bill over 50,000 is Massive Dynamic for $62,500.", evidence="", steps=1)
    res_fail = AgentResult(status="finished", claim="Found bill for Globex.", evidence="", steps=1)

    assert check_read_only_answer_matches_query(task, before, before, [], res_pass)[0] is True
    assert check_read_only_answer_matches_query(task, before, before, [], res_fail)[0] is False

# ---------------------------------------------------------------------------
# Integration tests for run_eval_suite with FakeLLM
# ---------------------------------------------------------------------------

def test_harness_task_1_pass(tmp_path):
    test_db = tmp_path / "eval1.db"
    out_json = tmp_path / "results.json"
    reset_db(test_db)

    script = [
        {"tool_call": {"name": "browser_goto", "args": {"thought": "Navigating to login", "url": "http://127.0.0.1:5201/login"}}},
        {"tool_call": {"name": "browser_snapshot", "args": {"thought": "Inspect login page"}}},
        {"tool_call": {"name": "browser_type", "args": {"thought": "Entering username", "id": "2", "text": "admin"}}},
        {"tool_call": {"name": "browser_type", "args": {"thought": "Entering password", "id": "3", "text": "admin"}}},
        {"tool_call": {"name": "browser_click", "args": {"thought": "Submitting login", "id": "4"}}},
        {"tool_call": {"name": "browser_goto", "args": {"thought": "Going to new bill form", "url": "http://127.0.0.1:5201/bills/new"}}},
        {"tool_call": {"name": "browser_snapshot", "args": {"thought": "Inspect new bill form"}}},
        {"tool_call": {"name": "browser_select", "args": {"thought": "Selecting vendor", "id": "4", "option": "Globex"}}},
        {"tool_call": {"name": "browser_type", "args": {"thought": "Entering invoice number", "id": "5", "text": "GLX-2026-099"}}},
        {"tool_call": {"name": "browser_type", "args": {"thought": "Entering amount", "id": "6", "text": "7850"}}},
        {"tool_call": {"name": "browser_type", "args": {"thought": "Entering due date", "id": "7", "text": "2026-10-25"}}},
        {"tool_call": {"name": "browser_click", "args": {"thought": "Saving bill", "id": "8"}}},
        {"tool_call": {"name": "finish", "args": {"thought": "Goal accomplished", "claim": "Created bill GLX-2026-099 for Globex", "evidence": "Bill created"}}},
        # Verifier turn
        {"tool_call": {"name": "report_verdict", "args": {"achieved": True, "claim_accurate": True, "reasons": "Bill created in DB"}}},
    ]

    llm = ScriptedFakeLLM(script)

    summary = run_eval_suite(
        only_ids=[1],
        port=5201,
        db_path=str(test_db),
        out_path=str(out_json),
        llm_instance=llm,
        headless=True,
    )

    assert summary["summary"]["passed"] == 1
    assert summary["summary"]["failed"] == 0
    assert summary["summary"]["errors"] == 0
    t1 = summary["tasks"][0]
    assert t1["result"] == "PASS"
    assert t1["verifier"] == "achieved"
    assert Path(out_json).exists()

def test_harness_task_9_fault_injection(tmp_path):
    test_db = tmp_path / "eval9.db"
    out_json = tmp_path / "results_9.json"
    reset_db(test_db)

    script = [
        {"tool_call": {"name": "browser_goto", "args": {"thought": "Navigating to login", "url": "http://127.0.0.1:5202/login"}}},
        {"tool_call": {"name": "browser_snapshot", "args": {"thought": "Inspect login page"}}},
        {"tool_call": {"name": "browser_type", "args": {"thought": "Entering username", "id": "2", "text": "admin"}}},
        {"tool_call": {"name": "browser_type", "args": {"thought": "Entering password", "id": "3", "text": "admin"}}},
        {"tool_call": {"name": "browser_click", "args": {"thought": "Submitting login", "id": "4"}}},
        {"tool_call": {"name": "browser_goto", "args": {"thought": "Going to new bill form", "url": "http://127.0.0.1:5202/bills/new"}}},
        {"tool_call": {"name": "browser_snapshot", "args": {"thought": "Inspect new bill form"}}},
        {"tool_call": {"name": "browser_select", "args": {"thought": "Selecting vendor", "id": "4", "option": "Globex"}}},
        {"tool_call": {"name": "browser_type", "args": {"thought": "Entering invoice number", "id": "5", "text": "GLX-2026-099"}}},
        {"tool_call": {"name": "browser_type", "args": {"thought": "Entering amount", "id": "6", "text": "7850"}}},
        {"tool_call": {"name": "browser_type", "args": {"thought": "Entering due date", "id": "7", "text": "2026-10-25"}}},
        {"tool_call": {"name": "browser_click", "args": {"thought": "Saving bill", "id": "8"}}},
        # Round 1 finish
        {"tool_call": {"name": "finish", "args": {"thought": "Goal accomplished", "claim": "Created bill GLX-2026-099", "evidence": "Success message shown"}}},
        # Verifier 1 detects diff is empty
        {"tool_call": {"name": "report_verdict", "args": {"achieved": False, "claim_accurate": False, "reasons": "DB diff is empty"}}},
        # Round 2 finish
        {"tool_call": {"name": "finish", "args": {"thought": "Retrying finish", "claim": "Created bill GLX-2026-099", "evidence": "Success message shown"}}},
        # Verifier 2 detects diff is still empty -> loop stops with verified=False
        {"tool_call": {"name": "report_verdict", "args": {"achieved": False, "claim_accurate": False, "reasons": "DB diff is still empty"}}},
    ]

    llm = ScriptedFakeLLM(script)

    summary = run_eval_suite(
        only_ids=[9],
        port=5202,
        db_path=str(test_db),
        out_path=str(out_json),
        llm_instance=llm,
        headless=True,
    )

    assert summary["summary"]["passed"] == 1
    t9 = summary["tasks"][0]
    assert t9["result"] == "PASS"
    assert t9["verifier"] == "mismatch"

def test_harness_llm_error_and_quota_exhaustion(tmp_path):
    test_db = tmp_path / "eval_error.db"
    out_json = tmp_path / "results_err.json"
    reset_db(test_db)

    # 1. Normal LLM infrastructure error
    script_infra_error = [LLMError("Simulated LLM connection timeout")]
    llm1 = ScriptedFakeLLM(script_infra_error)

    summary1 = run_eval_suite(
        only_ids=[1],
        port=5203,
        db_path=str(test_db),
        out_path=str(out_json),
        llm_instance=llm1,
        headless=True,
    )

    assert summary1["summary"]["errors"] == 1
    assert summary1["tasks"][0]["result"] == "ERROR"

    # 2. Daily quota exhaustion -> stops whole run early and tracks unrun tasks
    script_quota_error = [LLMError("daily quota exhausted for gemini-2.5-flash, resets in ~23h")]
    llm2 = ScriptedFakeLLM(script_quota_error)

    summary2 = run_eval_suite(
        only_ids=[1, 2, 3],
        port=5204,
        db_path=str(test_db),
        out_path=str(out_json),
        llm_instance=llm2,
        headless=True,
    )

    assert summary2["summary"]["errors"] == 1
    assert summary2["summary"]["unrun"] == 2
    assert len(summary2["tasks"]) == 1
    assert summary2["tasks"][0]["result"] == "ERROR"
