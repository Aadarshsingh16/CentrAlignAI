import argparse
import copy
import datetime
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional
import yaml
from werkzeug.serving import make_server

from agent.browser import BrowserSession
from agent.llm import make_llm, LLMError
from agent.loop import run_agent, AgentResult
from agent.policy import Policy, make_approval_guard
from agent.state import AgentState
from agent.tools import build_default_registry
from agent.trace import Trace
from evals.checks import run_checks
from mock_app.app import app
from mock_app.env import MockAppEnvironment
from mock_app.seed import reset_db, get_all_bills

class MockAppServer(threading.Thread):
    """Runs the Flask mock app in a dedicated background daemon thread."""

    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.port = port
        self.server = make_server("127.0.0.1", port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

def run_eval_suite(
    tasks_yaml_path: str = "evals/tasks.yaml",
    only_ids: Optional[list[int]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    headless: bool = True,
    out_path: str = "evals/results.json",
    port: int = 5055,
    db_path: Optional[str] = None,
    llm_instance: Optional[Any] = None,
    llm_factory: Optional[Callable[[dict], Any]] = None,
) -> dict[str, Any]:
    """
    Executes the full evaluation harness.
    Runs each task through the unchanged agent, evaluates ground truth and trace assertions,
    and produces structured results.
    """
    tasks_file = Path(tasks_yaml_path)
    if not tasks_file.exists():
        raise FileNotFoundError(f"Tasks definition file not found at: {tasks_yaml_path}")

    raw_yaml = yaml.safe_load(tasks_file.read_text(encoding="utf-8"))
    all_tasks = raw_yaml.get("tasks", [])

    if only_ids:
        tasks_to_run = [t for t in all_tasks if int(t.get("id")) in only_ids]
    else:
        tasks_to_run = all_tasks

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_traces_dir = Path("traces") / "evals" / timestamp
    eval_traces_dir.mkdir(parents=True, exist_ok=True)

    # Resolve database path
    target_db_path = str(Path(db_path).resolve()) if db_path else str(Path("mock_app/bills.db").resolve())
    os.environ["MOCK_APP_DB_PATH"] = target_db_path

    # Start mock app server on specified port
    server = MockAppServer(port=port)
    server.start()
    time.sleep(0.3)  # Short pause to let server bind

    # Base company context pointing to our eval server port
    base_context = ""
    context_file = Path("company_context.md")
    if context_file.exists():
        raw_context = context_file.read_text(encoding="utf-8")
        base_context = raw_context.replace("http://localhost:5000", f"http://127.0.0.1:{port}").replace(
            "http://127.0.0.1:5000", f"http://127.0.0.1:{port}"
        )

    task_results = []
    stopped_due_to_quota = False
    unrun_tasks = []

    try:
        for idx, task in enumerate(tasks_to_run):
            task_id = task.get("id")
            task_name = task.get("name", f"Task {task_id}")
            task_text = task.get("task", "")
            user_answers = list(task.get("user_answers", []))
            approvals_spec = copy.deepcopy(task.get("approvals", "approve"))
            env_flags = task.get("env_flags", {})
            check_names = task.get("checks", [])

            # Reset database before task run
            reset_db(target_db_path)
            before_bills = get_all_bills(target_db_path)

            # Set environment flags (e.g. FAULT_SILENT_SAVE=1)
            for k, v in env_flags.items():
                os.environ[k] = str(v)

            trace_file = eval_traces_dir / f"task_{task_id}.jsonl"
            trace = Trace(trace_file)
            state = AgentState()

            # Scripted ask_user input function
            def scripted_input(question: str) -> str:
                if user_answers:
                    return str(user_answers.pop(0))
                return ""

            # Scripted approval function
            def scripted_approval(ctx: dict[str, Any]) -> dict[str, Any]:
                nonlocal approvals_spec
                decision = "approve"
                if isinstance(approvals_spec, list):
                    if approvals_spec:
                        decision = approvals_spec.pop(0)
                elif isinstance(approvals_spec, str):
                    decision = approvals_spec

                if decision == "approve":
                    return {"approved": True, "note": ""}
                else:
                    return {"approved": False, "note": "Denied by simulated user"}

            browser = BrowserSession(headless=headless)
            registry = build_default_registry(
                browser=browser,
                state=state,
                workspace_dir=Path(".").resolve(),
                input_fn=scripted_input,
            )

            policy = Policy(config_path="policy.yaml" if Path("policy.yaml").exists() else None)
            approval_guard = make_approval_guard(
                browser=browser,
                policy=policy,
                approval_fn=scripted_approval,
                state=state,
            )
            registry.guards.append(approval_guard)

            # Determine LLM instance for this task
            if llm_factory:
                current_llm = llm_factory(task)
            elif llm_instance:
                current_llm = llm_instance
            else:
                current_llm = make_llm(provider=provider, model=model)

            env = MockAppEnvironment(db_path=target_db_path)

            status = "FAIL"
            failed_reasons = []
            checks_detail = []
            steps = 0
            verifier_label = "none"

            try:
                result = run_agent(
                    task=task_text,
                    registry=registry,
                    llm=current_llm,
                    state=state,
                    trace=trace,
                    context=base_context,
                    max_steps=40,
                    env=env,
                )

                steps = result.steps
                if result.verified is True:
                    verifier_label = "achieved"
                elif result.verified is False:
                    verifier_label = "mismatch"
                else:
                    verifier_label = "skipped"

                # Check for infrastructure LLM error
                if result.status == "llm_error":
                    status = "ERROR"
                    err_msg = str(result.evidence)
                    failed_reasons.append(f"LLM Infrastructure Error: {err_msg}")

                    if "daily quota" in err_msg.lower() or "perday" in err_msg.lower():
                        stopped_due_to_quota = True
                        unrun_tasks = [t.get("name") for t in tasks_to_run[idx + 1 :]]
                        task_results.append({
                            "id": task_id,
                            "name": task_name,
                            "result": status,
                            "steps": steps,
                            "asks": state.asks,
                            "approvals": state.approvals,
                            "denials": state.denials,
                            "verifier": verifier_label,
                            "trace_path": str(trace_file),
                            "failed_reasons": failed_reasons,
                            "checks": checks_detail,
                        })
                        break
                else:
                    # Run deterministic checks on ground truth and trace
                    after_bills = get_all_bills(target_db_path)
                    trace_records = Trace.read(trace_file)

                    check_outcomes = run_checks(check_names, task, before_bills, after_bills, trace_records, result)
                    all_passed = True
                    for c_name, c_passed, c_reason in check_outcomes:
                        checks_detail.append({"name": c_name, "passed": c_passed, "reason": c_reason})
                        if not c_passed:
                            all_passed = False
                            failed_reasons.append(f"[{c_name}] {c_reason}")

                    status = "PASS" if all_passed else "FAIL"

            except LLMError as e:
                status = "ERROR"
                err_msg = str(e)
                failed_reasons.append(f"LLM Error: {err_msg}")
                if "daily quota" in err_msg.lower() or "perday" in err_msg.lower():
                    stopped_due_to_quota = True
                    unrun_tasks = [t.get("name") for t in tasks_to_run[idx + 1 :]]
                    task_results.append({
                        "id": task_id,
                        "name": task_name,
                        "result": status,
                        "steps": steps,
                        "asks": state.asks,
                        "approvals": state.approvals,
                        "denials": state.denials,
                        "verifier": verifier_label,
                        "trace_path": str(trace_file),
                        "failed_reasons": failed_reasons,
                        "checks": checks_detail,
                    })
                    break
            except Exception as e:
                status = "ERROR"
                failed_reasons.append(f"Infrastructure Exception: {e}")
            finally:
                browser.close()
                # Clean up per-task environment flags
                for k in env_flags.keys():
                    os.environ.pop(k, None)

            task_results.append({
                "id": task_id,
                "name": task_name,
                "result": status,
                "steps": steps,
                "asks": state.asks,
                "approvals": state.approvals,
                "denials": state.denials,
                "verifier": verifier_label,
                "trace_path": str(trace_file),
                "failed_reasons": failed_reasons,
                "checks": checks_detail,
            })

    finally:
        server.shutdown()

    # Compile suite summary
    total = len(task_results)
    passed_count = sum(1 for r in task_results if r["result"] == "PASS")
    failed_count = sum(1 for r in task_results if r["result"] == "FAIL")
    error_count = sum(1 for r in task_results if r["result"] == "ERROR")

    summary_data = {
        "timestamp": timestamp,
        "summary": {
            "total_executed": total,
            "passed": passed_count,
            "failed": failed_count,
            "errors": error_count,
            "unrun": len(unrun_tasks),
        },
        "tasks": task_results,
    }

    # Write results.json
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # Print summary table formatted to EVALS.md specification
    print("\n" + "=" * 80)
    print(f"{'#':<3} {'Task':<30} {'Result':<8} {'Steps':<6} {'Asks':<5} {'Approvals':<10} {'Verifier':<10}")
    print("-" * 80)
    for r in task_results:
        print(
            f"{r['id']:<3} {r['name'][:30]:<30} {r['result']:<8} {r['steps']:<6} {r['asks']:<5} {r['approvals']:<10} {r['verifier']:<10}"
        )
    print("-" * 80)
    print(f"Total: {passed_count}/{total} passed (Errors: {error_count})")
    if stopped_due_to_quota:
        print(f"\n[ALERT] Run stopped early due to daily quota exhaustion.")
        print(f"Tasks not run: {', '.join(unrun_tasks)}")

    # Print failure details
    failures = [r for r in task_results if r["result"] in ("FAIL", "ERROR")]
    if failures:
        print("\n" + "=" * 80)
        print("FAILURE DETAILS:")
        for f in failures:
            print(f"\nTask {f['id']}: {f['name']} [{f['result']}]")
            print(f"  Trace: {f['trace_path']}")
            for reason in f["failed_reasons"]:
                print(f"  - {reason}")
        print("=" * 80)

    print(f"\nResults saved to: {out_file.resolve()}\n")
    return summary_data

def main():
    parser = argparse.ArgumentParser(description="CentrAlignAI Evaluation Harness")
    parser.add_argument("--tasks", type=str, default="evals/tasks.yaml", help="Path to tasks YAML file")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated task IDs to run (e.g. 1,4,5)")
    parser.add_argument("--model", type=str, default=None, help="LLM model override")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider: gemini or ollama")
    parser.add_argument("--headless", action="store_true", default=True, help="Run browser in headless mode")
    parser.add_argument("--out", type=str, default="evals/results.json", help="Path to write results.json")
    parser.add_argument("--port", type=int, default=5055, help="Port to host mock application")
    parser.add_argument("--db-path", type=str, default=None, help="Custom database path")

    args = parser.parse_args()

    only_ids = None
    if args.only:
        only_ids = [int(x.strip()) for x in args.only.split(",") if x.strip().isdigit()]

    run_eval_suite(
        tasks_yaml_path=args.tasks,
        only_ids=only_ids,
        provider=args.provider,
        model=args.model,
        headless=args.headless,
        out_path=args.out,
        port=args.port,
        db_path=args.db_path,
    )

if __name__ == "__main__":
    main()
