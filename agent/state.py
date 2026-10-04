from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentState:
    """
    Task-agnostic working memory for the agent.
    Tracks extracted facts, historical steps completed, and open questions.
    """
    facts: dict[str, Any] = field(default_factory=dict)
    steps_done: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    approvals: int = 0
    denials: int = 0
    asks: int = 0
    denial_notes: list[str] = field(default_factory=list)

    def remember(self, key: str, value: Any) -> None:
        """Store or update a key-value fact in memory."""
        self.facts[key] = value

    def add_step(self, summary: str) -> None:
        """Record a completed step summary."""
        self.steps_done.append(summary)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a serializable dictionary."""
        return {
            "facts": dict(self.facts),
            "steps_done": list(self.steps_done),
            "open_questions": list(self.open_questions),
            "approvals": self.approvals,
            "denials": self.denials,
            "asks": self.asks,
            "denial_notes": list(self.denial_notes),
        }

    def to_prompt(self, max_chars: int = 1500) -> str:
        """
        Generate a compact, readable prompt representation of memory.
        Capped in size to avoid consuming excessive context.
        """
        lines = []
        if self.facts:
            lines.append("Known Facts:")
            for k, v in self.facts.items():
                lines.append(f"  - {k}: {v}")

        if self.steps_done:
            lines.append("Steps Completed:")
            # Show the most recent steps if there are many
            recent_steps = self.steps_done[-10:]
            for s in recent_steps:
                lines.append(f"  - {s}")

        if self.open_questions:
            lines.append("Open Questions:")
            for q in self.open_questions:
                lines.append(f"  - {q}")

        text = "\n".join(lines) if lines else "No prior facts or steps recorded."
        if len(text) > max_chars:
            text = text[:max_chars] + "... [state truncated]"
        return text
