# Evaluation Harness

Run all evaluation tasks through the automated benchmark suite:
```bash
python -m evals.run_evals
```
Run specific tasks by ID (e.g. tasks 1, 4, and 5) with a custom model and headless browser:
```bash
python -m evals.run_evals --only 1,4,5 --model gemini-2.5-flash --headless
```
Results and ground-truth verification outcomes are saved to `evals/results.json`.
Traces for each run are saved under `traces/evals/<timestamp>/task_<id>.jsonl`.
