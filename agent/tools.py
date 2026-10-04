import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from agent.browser import BrowserSession
from agent.state import AgentState

def format_snapshot(snapshot: dict[str, Any]) -> str:
    """
    Renders a browser snapshot compactly for an LLM prompt.
    Formats page URL, title, alerts first, then interactive elements, then trimmed text.
    """
    lines = [
        f"URL: {snapshot.get('url', '')}",
        f"Title: {snapshot.get('title', '')}",
    ]

    alerts = snapshot.get("alerts", [])
    if alerts:
        lines.append("ALERTS / ERRORS:")
        for alert in alerts:
            lines.append(f"  [ALERT] {alert}")

    lines.append("Interactive Elements:")
    elements = snapshot.get("elements", [])
    if not elements:
        lines.append("  (No interactive elements found)")
    else:
        for el in elements:
            val_part = f", value='{el['value']}'" if el.get("value") else ""
            dis_part = " (disabled)" if el.get("disabled") else ""
            lines.append(
                f"  [{el.get('id')}] <{el.get('tag')}> role={el.get('role')} label='{el.get('label')}'{val_part}{dis_part}"
            )

    text = snapshot.get("text", "").strip()
    if text:
        # Cap visible text in snapshot formatting
        if len(text) > 1000:
            text = text[:1000] + "... [text truncated]"
        lines.append("Page Text:")
        lines.append(text)

    return "\n".join(lines)

@dataclass
class Tool:
    """Task-agnostic definition of an agent tool."""
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

class ToolRegistry:
    """
    Registry for tools, schema generation, execution, and policy guard evaluation.
    Guarantees non-raising execution with structured observations and error reporting.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self.guards: list[Callable[[str, dict[str, Any]], Optional[dict[str, Any]]]] = []

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        """Returns JSON schema definitions for all registered tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, args: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Executes a registered tool by name with arguments.
        Enforces guards, validates required parameters, and catches all exceptions.
        """
        if args is None:
            args = {}

        if name not in self._tools:
            available = ", ".join(sorted(self._tools.keys()))
            err = f"Unknown tool '{name}'. Valid tools: {available}."
            return {"ok": False, "error": err, "observation": err}

        tool = self._tools[name]

        # 1. Evaluate guards (e.g. approval policies)
        for guard in self.guards:
            try:
                guard_result = guard(name, args)
                if guard_result and "block" in guard_result:
                    reason = guard_result["block"]
                    return {"ok": False, "error": reason, "observation": f"Blocked by policy: {reason}"}
            except Exception as e:
                err = f"Guard check failed: {e}"
                return {"ok": False, "error": err, "observation": err}

        # 2. Validate required parameters defined in schema
        schema = tool.parameters or {}
        required_params = schema.get("required", [])
        missing = [p for p in required_params if p not in args or args[p] is None]
        if missing:
            expected = ", ".join(schema.get("properties", {}).keys())
            err = f"Missing required argument(s) for '{name}': {', '.join(missing)}. Expected arguments: {expected}."
            return {"ok": False, "error": err, "observation": err}

        # 3. Execute tool implementation safely
        try:
            sig = inspect.signature(tool.fn)
            # Filter out extraneous arguments if tool does not accept **kwargs
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            call_args = args if has_var_keyword else {k: v for k, v in args.items() if k in sig.parameters}

            result = tool.fn(**call_args)

            if isinstance(result, dict):
                if "ok" not in result:
                    result["ok"] = True
                if "observation" not in result:
                    result["observation"] = str(result)
                return result
            elif isinstance(result, str):
                return {"ok": True, "observation": result}
            else:
                return {"ok": True, "observation": str(result)}
        except Exception as e:
            err = f"Tool execution error in '{name}': {e}"
            return {"ok": False, "error": err, "observation": err}

def build_default_registry(
    browser: BrowserSession,
    state: AgentState,
    workspace_dir: str | Path,
    input_fn: Callable[[str], str] = input,
) -> ToolRegistry:
    """
    Constructs and populates a default ToolRegistry with all standard tools.
    All tools are strictly task-agnostic.
    """
    registry = ToolRegistry()
    workspace_path = Path(workspace_dir).resolve()

    # Tool: list_files
    def list_files(dir: str = ".") -> dict[str, Any]:
        try:
            target = (workspace_path / dir).resolve()
            if not target.is_relative_to(workspace_path):
                return {"ok": False, "error": "Access denied: path is outside workspace directory."}
            if not target.exists():
                return {"ok": False, "error": f"Directory '{dir}' does not exist."}
            if not target.is_dir():
                return {"ok": False, "error": f"Path '{dir}' is not a directory."}

            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            items = []
            for item in entries[:100]:
                suffix = "/" if item.is_dir() else ""
                items.append(f"{item.name}{suffix}")
            
            summary = "\n".join(items) if items else "(Empty directory)"
            if len(entries) > 100:
                summary += f"\n... [showing first 100 of {len(entries)} items]"
            return {"ok": True, "observation": f"Files in {dir}:\n{summary}"}
        except Exception as e:
            return {"ok": False, "error": f"Failed to list directory: {e}"}

    registry.register(
        Tool(
            name="list_files",
            description="List files and directories inside a relative path within the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "dir": {"type": "string", "description": "Relative directory path (default is '.')."}
                },
            },
            fn=list_files,
        )
    )

    # Tool: read_file
    def read_file(path: str) -> dict[str, Any]:
        try:
            target = (workspace_path / path).resolve()
            if not target.is_relative_to(workspace_path):
                return {"ok": False, "error": "Access denied: path is outside workspace directory."}
            if not target.exists():
                return {"ok": False, "error": f"File '{path}' does not exist."}
            if target.is_dir():
                return {"ok": False, "error": f"Path '{path}' is a directory, not a file."}

            text = target.read_text(encoding="utf-8", errors="replace")
            if len(text) > 4000:
                text = text[:4000] + "\n... [file content truncated]"
            return {"ok": True, "observation": text}
        except Exception as e:
            return {"ok": False, "error": f"Failed to read file: {e}"}

    registry.register(
        Tool(
            name="read_file",
            description="Read the text content of a file at a relative path inside the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path to read."}
                },
                "required": ["path"],
            },
            fn=read_file,
        )
    )

    # Tool: browser_goto
    def browser_goto(url: str) -> dict[str, Any]:
        nav = browser.goto(url)
        if not nav["ok"]:
            return nav
        snap = browser.snapshot()
        obs = f"Navigated to {url}.\n" + format_snapshot(snap)
        return {"ok": True, "observation": obs}

    registry.register(
        Tool(
            name="browser_goto",
            description="Open a web page URL and return a snapshot of the rendered page.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to navigate to."}
                },
                "required": ["url"],
            },
            fn=browser_goto,
        )
    )

    # Tool: browser_snapshot
    def browser_snapshot() -> dict[str, Any]:
        snap = browser.snapshot()
        if not snap["ok"]:
            return snap
        return {"ok": True, "observation": format_snapshot(snap)}

    registry.register(
        Tool(
            name="browser_snapshot",
            description="Capture a new snapshot of current web page text, alerts, and interactive elements.",
            parameters={"type": "object", "properties": {}},
            fn=browser_snapshot,
        )
    )

    # Tool: browser_click
    def browser_click(id: str) -> dict[str, Any]:
        click_res = browser.click(str(id))
        if not click_res["ok"]:
            return click_res
        snap = browser.snapshot()
        obs = f"{click_res['observation']}\nCurrent page:\n" + format_snapshot(snap)
        return {"ok": True, "observation": obs}

    registry.register(
        Tool(
            name="browser_click",
            description="Click an interactive element identified by its snapshot numeric id.",
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Element id from the latest snapshot."}
                },
                "required": ["id"],
            },
            fn=browser_click,
        )
    )

    # Tool: browser_type
    def browser_type(id: str, text: str) -> dict[str, Any]:
        type_res = browser.type(str(id), text)
        return type_res

    registry.register(
        Tool(
            name="browser_type",
            description="Type text into an input or textarea element by its snapshot id.",
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Target input element id from the snapshot."},
                    "text": {"type": "string", "description": "Text to enter into the input."},
                },
                "required": ["id", "text"],
            },
            fn=browser_type,
        )
    )

    # Tool: browser_select
    def browser_select(id: str, option: str) -> dict[str, Any]:
        select_res = browser.select(str(id), option)
        return select_res

    registry.register(
        Tool(
            name="browser_select",
            description="Select an option in a dropdown element by its snapshot id.",
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Target select element id from the snapshot."},
                    "option": {"type": "string", "description": "Option label or value to select."},
                },
                "required": ["id", "option"],
            },
            fn=browser_select,
        )
    )

    # Tool: remember
    def remember(key: str, value: Any) -> dict[str, Any]:
        state.remember(key, value)
        return {"ok": True, "observation": f"Saved fact: {key} = {value}"}

    registry.register(
        Tool(
            name="remember",
            description="Store an extracted fact or intermediate finding in working memory.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Name or identifier of the fact."},
                    "value": {"type": "string", "description": "Value or content to record."},
                },
                "required": ["key", "value"],
            },
            fn=remember,
        )
    )

    # Tool: ask_user
    def ask_user(question: str) -> dict[str, Any]:
        state.asks += 1
        answer = input_fn(question)
        return {"ok": True, "observation": f"User answer: {answer}"}

    registry.register(
        Tool(
            name="ask_user",
            description="Ask the user a question for clarification or decision making.",
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to present to the user."}
                },
                "required": ["question"],
            },
            fn=ask_user,
        )
    )

    # Tool: finish
    def finish(claim: str, evidence: str = "") -> dict[str, Any]:
        if not claim:
            return {"ok": False, "error": "Claim parameter cannot be empty."}
        return {
            "ok": True,
            "finished": True,
            "claim": claim,
            "evidence": evidence,
            "observation": f"Task outcome declared: {claim}. Evidence: {evidence}",
        }

    registry.register(
        Tool(
            name="finish",
            description="Declare that the task is completed with a claim and supporting evidence.",
            parameters={
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "Summary statement describing what was accomplished."},
                    "evidence": {"type": "string", "description": "Concrete supporting evidence verifying the outcome."},
                },
                "required": ["claim"],
            },
            fn=finish,
        )
    )

    return registry
