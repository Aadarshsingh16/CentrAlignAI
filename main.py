import argparse
import sys
from pathlib import Path
from agent.browser import BrowserSession
from agent.llm import LLM
from agent.loop import run_agent
from agent.policy import Policy, make_approval_guard, terminal_approval_fn
from agent.state import AgentState
from agent.tools import build_default_registry
from agent.trace import Trace
from mock_app.env import MockAppEnvironment

def main():
    parser = argparse.ArgumentParser(description="Autonomous AI Task Worker")
    parser.add_argument("task", type=str, help="Task description for the agent")
    parser.add_argument("--workspace", type=str, default=".", help="Workspace root directory (default: current directory)")
    parser.add_argument("--context", type=str, default="company_context.md", help="Path to context markdown file")
    parser.add_argument("--trace", type=str, default="traces/run.jsonl", help="Output path for JSONL trace log")
    parser.add_argument("--model", type=str, default=None, help="LLM model override")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")

    args = parser.parse_args()

    # Load context if present
    context_text = ""
    context_path = Path(args.context)
    if context_path.exists():
        context_text = context_path.read_text(encoding="utf-8")

    # Initialize components
    workspace_path = Path(args.workspace).resolve()
    browser = BrowserSession(headless=args.headless)
    state = AgentState()
    trace = Trace(args.trace)

    # Use real terminal input for ask_user
    def terminal_ask(question: str) -> str:
        print(f"\n[AGENT QUESTION]: {question}")
        return input("[YOUR ANSWER]: ").strip()

    registry = build_default_registry(
        browser=browser,
        state=state,
        workspace_dir=workspace_path,
        input_fn=terminal_ask,
    )

    policy = Policy(config_path="policy.yaml" if Path("policy.yaml").exists() else None)
    approval_guard = make_approval_guard(
        browser=browser,
        policy=policy,
        approval_fn=terminal_approval_fn,
        state=state,
    )
    registry.guards.append(approval_guard)

    llm = LLM(model=args.model)

    print(f"\nStarting Task: {args.task}")
    print(f"Model: {llm.model_name}")
    print(f"Trace log: {args.trace}\n" + "-" * 50)

    try:
        env = MockAppEnvironment()
        result = run_agent(
            task=args.task,
            registry=registry,
            llm=llm,
            state=state,
            trace=trace,
            context=context_text,
            max_steps=40,
            env=env,
        )

        print("\n" + "=" * 50)
        print("AGENT EXECUTION FINISHED")
        print(f"Status:   {result.status}")
        print(f"Steps:    {result.steps}")
        print(f"Claim:    {result.claim}")
        print(f"Evidence: {result.evidence}")
        if result.verified is not None:
            print(f"Verified: {result.verified}")
        if result.verdict is not None:
            print(f"Verdict:  {result.verdict}")
        if result.diff is not None:
            print(f"Diff:     {result.diff}")
        print("=" * 50 + "\n")
        return 0 if result.status == "finished" else 1

    finally:
        browser.close()

if __name__ == "__main__":
    sys.exit(main())
