import re
from pathlib import Path
from typing import Any, Callable, Optional
import yaml
from agent.browser import BrowserSession
from agent.state import AgentState

DEFAULT_RULES = [
    r"(?i)\b(save|submit|pay|paid|delete|confirm|approve|send)\b",
]

class Policy:
    """
    Data-driven approval policy specifying regex rules for sensitive actions.
    """

    def __init__(self, rules: Optional[list[str]] = None, config_path: Optional[str | Path] = None):
        if rules is not None:
            self.rules = list(rules)
        else:
            self.rules = list(DEFAULT_RULES)

        if config_path:
            self.load_from_yaml(config_path)

    def load_from_yaml(self, path: str | Path) -> None:
        """Loads policy rules from a YAML configuration file."""
        p = Path(path)
        if p.exists():
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "rules" in data and isinstance(data["rules"], list):
                    self.rules = [str(r) for r in data["rules"]]
                elif isinstance(data, list):
                    self.rules = [str(r) for r in data]
            except Exception:
                pass

    def is_risky(self, action: str, label: str) -> bool:
        """Determines if an action on a given element label requires user approval."""
        if action != "browser_click":
            return False
        for rule in self.rules:
            if re.search(rule, label):
                return True
        return False

def terminal_approval_fn(context: dict[str, Any]) -> dict[str, Any]:
    """Interactive terminal approval prompt."""
    print("\n" + "!" * 50)
    print("ACTION REQUIRES HUMAN APPROVAL:")
    print(f"Action:       {context.get('action')}")
    print(f"Target Label: {context.get('label')}")
    print(f"Page URL:     {context.get('url')}")
    form_vals = context.get("form_values", {})
    if form_vals:
        print("Form Fields:")
        for k, v in form_vals.items():
            print(f"  - {k}: {v}")
    print("!" * 50)

    ans = input("Approve this action? (y/n): ").strip().lower()
    if ans in ("y", "yes"):
        note = input("Optional note (or press Enter): ").strip()
        return {"approved": True, "note": note}
    else:
        note = input("Reason / note for denial: ").strip()
        return {"approved": False, "note": note}

def make_approval_guard(
    browser: Optional[BrowserSession],
    policy: Policy,
    approval_fn: Callable[[dict[str, Any]], dict[str, Any]],
    state: AgentState,
) -> Callable[[str, dict[str, Any]], Optional[dict[str, Any]]]:
    """
    Creates a ToolRegistry guard evaluating approval rules on tool invocation.
    Inspects element labels and form values without mutating snapshot element ids.
    """
    def guard(name: str, args: dict[str, Any]) -> Optional[dict[str, Any]]:
        if name != "browser_click" or browser is None:
            return None

        element_id = str(args.get("id", ""))
        desc_res = browser.describe(element_id)
        if not desc_res.get("ok"):
            return None

        label = desc_res.get("element", {}).get("label", "")
        if not policy.is_risky("browser_click", label):
            return None

        # Gather read-only context without taking a new snapshot
        url = ""
        try:
            if browser.page and not browser.page.is_closed():
                url = browser.page.url
        except Exception:
            pass

        form_vals = {}
        if hasattr(browser, "form_values"):
            form_vals = browser.form_values()

        context = {
            "action": "browser_click",
            "element_id": element_id,
            "label": label,
            "url": url,
            "form_values": form_vals,
        }

        decision = approval_fn(context)
        is_approved = bool(decision.get("approved"))
        note = str(decision.get("note", "")).strip()

        if is_approved:
            state.approvals += 1
            return None
        else:
            state.denials += 1
            if note:
                state.denial_notes.append(note)
            reason = f"denied by user: {note}" if note else "denied by user"
            return {"block": reason}

    return guard
