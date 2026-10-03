# AGENTS.md — Rules for the coding agent

Read this file and PLAN.md before doing anything. EVALS.md defines "done".

## How to work
1. Do ONE step from PLAN.md at a time. When it is finished, STOP and summarize what you built in plain language. Wait for the human to say "next".
2. Write a test for each module. Run it. Show the output. Never say "done" without a passing run.
3. Make a small git commit after each working step (message: `step N: <what>`).
4. Keep code simple and readable. The human must be able to explain every line in an interview. No clever abstractions, no extra frameworks, no features not in PLAN.md.
5. If something in PLAN.md is unclear or wrong, ask. Do not silently change the design.

## Architecture rules (non-negotiable)
- The agent loop (`agent/`) and its tools must be TASK-AGNOSTIC. No word like "invoice", "bill", "vendor" anywhere in `agent/`. Only `mock_app/`, `invoices/`, `evals/` and the task text may be domain-specific.
- The LLM chooses every next action. Do NOT hardcode a step sequence or a workflow.
- Tool errors and page error messages are returned to the LLM as observations. Never swallow them.
- The approval gate is enforced IN CODE (tool executor), not by asking the LLM to be careful.
- The verifier checks ground truth from the environment (database), never just the agent's own claim.
- Every step is written to the trace log (thought, action, args, observation, state).
- Hard limits: max 40 steps per task, stop if the same action+args repeats 3 times.

## Other rules
- Python 3.11+, pytest for tests, Playwright for the browser, Flask + SQLite for the mock app.
- LLM access goes through ONE file, `agent/llm.py`, so the provider can be swapped.
- API keys come from environment variables / `.env`. Never commit keys. `.env` is in `.gitignore`.
- Use only mock data. No real credentials or third-party sites.
- Do not add CI, Docker, or docs beyond what PLAN.md lists.
