import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

class Trace:
    """
    Appends structured step records into a JSONL trace log.
    Provides ground truth auditing and debugging logs for all agent actions.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        step: int,
        thought: str,
        action: str,
        args: dict[str, Any],
        observation: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Appends one JSON line representing a single step execution."""
        capped_obs = str(observation)
        if len(capped_obs) > 5000:
            capped_obs = capped_obs[:5000] + "... [trace observation truncated]"

        record = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "thought": thought,
            "action": action,
            "args": args,
            "observation": capped_obs,
            "state": state,
        }

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    @staticmethod
    def read(path: Union[str, Path]) -> list[dict[str, Any]]:
        """Reads all JSONL records from the specified trace file."""
        target = Path(path)
        if not target.exists():
            return []
        records = []
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records
