# PLAN.md — Autonomous AI Task Worker

## Goal
A task-agnostic agent that takes a plain-English task, works in a real browser and local files, recovers from errors, asks a human when unsafe or ambiguous, and proves the outcome with an independent verifier.

Demo task: "Find the latest invoice from Globex, extract the amount and due date, enter it in our internal system, and tell me when done."

## Repo layout
```
ai-worker/
  AGENTS.md  PLAN.md  EVALS.md  README.md  .env.example  requirements.txt
  mock_app/        # Flask + SQLite "internal system" (domain-specific OK)
    app.py  templates/  seed.py
  invoices/        # sample invoice .txt files (domain-specific OK)
  agent/           # TASK-AGNOSTIC
    llm.py         # single provider adapter: chat(messages, tools) -> text/tool_call
    loop.py        # the agent loop
    tools.py       # tool registry + executor (+ approval policy hook)
    browser.py     # Playwright wrapper: goto, snapshot, click, type
    state.py       # memory: facts, steps_done, open_questions (JSON)
    policy.py      # which actions need human approval (regex rules)
    verifier.py    # independent verification vs environment ground truth
    trace.py       # JSONL trace of every step
    env.py         # Environment interface: ground_truth() -> dict
  evals/
    tasks.yaml  run_evals.py   # table of pass/fail
  tests/
  main.py          # CLI: python main.py "task text"
```

## Core loop
```
task -> [LLM sees: task, state, last observations, tool list]
     -> LLM returns ONE tool call (+ short thought)
     -> executor runs it (policy check first; approval gate if risky)
     -> observation (result or error text) appended + state updated + trace written
     -> repeat until LLM calls finish(claim)
finish(claim) -> verifier reads ground truth, compares to task + claim
     -> pass: return summary + evidence;  fail: feed mismatch back to LLM, continue (max 2 rounds)
```

## Tools (generic)
| Tool | Purpose |
|---|---|
| list_files(dir) / read_file(path) | Local files (sandboxed to a workspace dir) |
| browser_goto(url) | Open a page |
| browser_snapshot() | Returns visible text + numbered list of interactive elements (id, role, label, value) + any error/alert text |
| browser_click(id) / browser_type(id, text) / browser_select(id, option) | Act on elements |
| remember(key, value) | Save a fact into state |
| ask_user(question) | Clarification; blocks until the human answers |
| finish(claim, evidence) | Declare done; triggers verifier |

## Approval policy (policy.py, in code)
- `browser_click` on an element whose label matches `save|submit|pay|delete|confirm|approve` => requires human approval, shown with context (current URL, form values from the last snapshot).
- Policy is data (a regex list), so it generalizes. Denied => agent receives "denied by user" as the observation and must adapt.

## Verifier (verifier.py)
- `env.ground_truth()` returns the app's real data (reads SQLite directly, not through the browser).
- A fresh LLM call (no agent history) gets: original task, agent's claim, ground truth before and after. It returns {achieved: bool, reasons, evidence}.
- Also records before/after diff of ground truth so the final summary can show exactly what changed.
- Eval harness additionally runs deterministic assertions on the DB (does not trust the LLM verifier alone).

## Mock app (mock_app/)
- Routes: `/login` (admin/admin), `/bills` (table), `/bills/new` (form: vendor dropdown, invoice_no, amount, due_date), `/bills/<id>` (detail, "Mark as paid" button).
- Traps built in:
  1. Date field accepts only `YYYY-MM-DD`; invoices use `DD/MM/YYYY` or `15 Oct 2026`. Error shown in a red alert.
  2. Duplicate `invoice_no` rejected with an error.
  3. Vendor list contains both "Acme Corp" and "Acme Corporation".
  4. One invoice has an amount ~10x that vendor's history.
- Seed data: 3 earlier bills so history exists.

## Invoices (invoices/)
~8 text files across vendors (Globex, Acme Corp, Acme Corporation, Initech, Umbrella) with different dates and formats. One vendor has two invoices so "latest" needs date comparison. One file has the anomalous amount.

## Build steps (stop after each; human runs and reads it)
1. **Setup**: repo, venv, requirements, `.env.example`, `.gitignore`, pytest runs.
2. **Mock app + seed data + invoice files**; start it, click through manually.
3. **browser.py**: goto/snapshot/click/type; test against the mock app (log in, open form).
4. **tools.py + state.py + trace.py**: registry, executor returning structured observations, JSONL trace; unit tests.
5. **llm.py + loop.py**: happy-path task works end to end (without approval/verify yet).
6. **policy.py + ask_user**: approval gate and clarification, enforced in executor.
7. **env.py + verifier.py**: independent verification; show a case where the agent claims success and the verifier catches a mismatch (e.g. temporarily break an action).
8. **Loop safety**: max steps, repeated-action detection, retry hints; test with the trap scenarios.
9. **evals/**: tasks.yaml + run_evals.py printing a pass/fail table (see EVALS.md).
10. **README + demo**: architecture, design decisions, limitations, next steps, assumptions, tools used; record 3-4 min demo.

## Time budget (~5h)
Steps 1-3: 1h15 | 4-5: 1h15 | 6-7: 1h | 8-9: 45m | 10: 45m

## If time runs short, cut in this order
PDF support (not planned) -> trap 4 -> eval count (min 5) -> polish. NEVER cut: verification, approval gate, trace, eval table.
