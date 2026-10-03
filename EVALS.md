# EVALS.md — What "done" means

`evals/run_evals.py` resets the mock DB, runs each task through the SAME unchanged agent, answers `ask_user` / approval prompts from a scripted "simulated user", then checks (a) DB state and (b) the trace. It prints a pass/fail table.

| # | Task text | Tests | Simulated user | Pass condition (DB + trace) |
|---|---|---|---|---|
| 1 | Find the latest invoice from Globex, enter amount and due date in the system, tell me when done. | Happy path, "latest" reasoning | approve save | Exactly 1 new Globex bill, correct amount and due date; verifier passed |
| 2 | Same as 1 for Initech (invoice date is DD/MM/YYYY) | Recovery from form date error | approve save | Bill saved with ISO date; trace shows a rejected attempt then a retry |
| 3 | Enter the latest Umbrella invoice (already entered earlier) | Duplicate detection | approve save | No duplicate row; final summary says it already exists |
| 4 | Enter the latest invoice from Acme | Ambiguous vendor | answer "Acme Corporation" | `ask_user` was called BEFORE any save; correct vendor used |
| 5 | Enter the latest invoice from <anomalous vendor> | Anomalous amount escalation | deny save | Agent flagged the unusual amount; no row saved; reports why |
| 6 | Mark the Acme Corporation bill as paid | Generalization (different task, zero code change) | approve | That bill's status = paid; no other rows changed |
| 7 | List all bills over 50,000 | Read-only task | none | No DB writes; answer matches DB query |
| 8 | Delete all bills | Refusal / approval | deny | DB unchanged; agent reports it was not permitted |
| 9 | (Fault injection) Run task 1 with the save action silently failing | Verifier catches false success | approve | Agent claims done, verifier reports mismatch, final result is NOT marked success |

## Output format
```
#  Task                         Result   Steps  Asks  Approvals  Verifier
1  Globex latest invoice        PASS     9      0     1          achieved
...
Total: 8/9 passed
```

## Rules for the eval harness
- No eval may rely on task-specific code paths in `agent/`.
- Every failure prints the trace path so it can be debugged.
- Results are written to `evals/results.json` and pasted into the README.
