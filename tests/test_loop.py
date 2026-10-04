import copy
from agent.llm import LLMResponse
from agent.loop import run_agent, AgentResult
from agent.state import AgentState
from agent.tools import Tool, ToolRegistry, build_default_registry
from agent.trace import Trace

class FakeLLM:
    """Scripted LLM for unit tests without network calls."""

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.history_received = []

    def chat(self, system: str, messages: list[dict], tool_schemas: list[dict]) -> LLMResponse:
        self.history_received.append(copy.deepcopy(messages))
        if not self.script:
            return LLMResponse(text="", tool_call=None, raw=None)
        next_call = self.script.pop(0)
        return LLMResponse(
            text=next_call.get("text", "thinking"),
            tool_call=next_call.get("tool_call"),
            raw=None,
        )

def test_finish_ends_run(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace = Trace(trace_path)
    state = AgentState()
    registry = build_default_registry(browser=None, state=state, workspace_dir=tmp_path)

    script = [
        {
            "tool_call": {
                "name": "remember",
                "args": {"thought": "Record info", "key": "status", "value": "ready"},
            }
        },
        {
            "tool_call": {
                "name": "finish",
                "args": {"thought": "All done", "claim": "Job finished successfully", "evidence": "Status is ready"},
            }
        },
    ]

    llm = FakeLLM(script)
    res = run_agent("Test task", registry, llm, state, trace, max_steps=10)

    assert res.status == "finished"
    assert res.claim == "Job finished successfully"
    assert res.evidence == "Status is ready"
    assert res.steps == 2

    # Verify trace
    records = Trace.read(trace_path)
    assert len(records) == 2
    assert records[0]["thought"] == "Record info"
    assert records[1]["action"] == "finish"

def test_max_steps_stops_loop(tmp_path):
    trace = Trace(tmp_path / "trace.jsonl")
    state = AgentState()
    registry = build_default_registry(browser=None, state=state, workspace_dir=tmp_path)

    # Endless different remember calls
    script = [
        {"tool_call": {"name": "remember", "args": {"thought": f"Step {i}", "key": f"k{i}", "value": f"v{i}"}}}
        for i in range(10)
    ]
    llm = FakeLLM(script)
    res = run_agent("Test task", registry, llm, state, trace, max_steps=4)

    assert res.status == "max_steps"
    assert res.steps == 4

def test_repeated_action_detection(tmp_path):
    trace = Trace(tmp_path / "trace.jsonl")
    state = AgentState()
    registry = build_default_registry(browser=None, state=state, workspace_dir=tmp_path)

    # Identical action called on identical state 3 times
    identical_call = {
        "tool_call": {
            "name": "remember",
            "args": {"thought": "Looping", "key": "stuck_key", "value": "same_val"},
        }
    }
    # Initial call sets the state, followed by 3 calls on identical state
    script = [identical_call, identical_call, identical_call, identical_call]

    llm = FakeLLM(script)
    res = run_agent("Test task", registry, llm, state, trace, max_steps=10)

    assert res.status == "stuck"
    assert "repeated identical actions" in res.claim.lower()
    assert res.steps == 4

    # Check that warning was injected on 2nd repeat on identical state (step 3)
    records = Trace.read(tmp_path / "trace.jsonl")
    assert "WARNING" in records[2]["observation"]

def test_same_click_on_different_pages_not_a_repeat(tmp_path):
    trace = Trace(tmp_path / "trace.jsonl")
    state = AgentState()
    registry = ToolRegistry()

    pages = ["Login View: button id=4", "Bills View: button id=4"]
    page_idx = [0]

    def mock_click(id: str):
        curr = pages[page_idx[0]]
        if page_idx[0] < len(pages) - 1:
            page_idx[0] += 1
        return f"Clicked element {id} on {curr}. Current view: {pages[page_idx[0]]}"

    registry.register(
        Tool(
            name="browser_click",
            description="Click element",
            parameters={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
            fn=mock_click,
        )
    )
    registry.register(
        Tool(
            name="finish",
            description="Finish",
            parameters={
                "type": "object",
                "properties": {"claim": {"type": "string"}, "evidence": {"type": "string"}},
                "required": ["claim"],
            },
            fn=lambda claim, evidence="": {"ok": True, "claim": claim, "evidence": evidence},
        )
    )

    # Click id="4" on Login, click id="4" on Bills, then finish
    script = [
        {"tool_call": {"name": "browser_click", "args": {"id": "4"}}},
        {"tool_call": {"name": "browser_click", "args": {"id": "4"}}},
        {"tool_call": {"name": "finish", "args": {"claim": "Navigated across pages"}}},
    ]

    llm = FakeLLM(script)
    res = run_agent("Test task", registry, llm, state, trace, max_steps=10)

    assert res.status == "finished"
    assert res.steps == 3
    records = Trace.read(tmp_path / "trace.jsonl")
    assert "WARNING" not in records[1]["observation"]

def test_tool_error_appears_in_next_model_input(tmp_path):
    trace = Trace(tmp_path / "trace.jsonl")
    state = AgentState()
    registry = build_default_registry(browser=None, state=state, workspace_dir=tmp_path)

    script = [
        {
            "tool_call": {
                "name": "read_file",
                "args": {"thought": "Reading nonexistent", "path": "does_not_exist.txt"},
            }
        },
        {
            "tool_call": {
                "name": "finish",
                "args": {"thought": "Handling error", "claim": "Detected missing file", "evidence": "Error received"},
            }
        },
    ]

    llm = FakeLLM(script)
    res = run_agent("Test task", registry, llm, state, trace, max_steps=5)

    assert res.status == "finished"
    assert len(llm.history_received) == 2

    # In turn 2, check that the previous error was sent as tool observation
    turn_2_messages = llm.history_received[1]
    tool_msg = next(m for m in turn_2_messages if m.get("role") == "tool")
    assert "does not exist" in tool_msg["content"].lower()

def test_thought_param_stripped_and_recorded(tmp_path):
    trace_file = tmp_path / "trace.jsonl"
    trace = Trace(trace_file)
    state = AgentState()

    # Create a tool that strictly rejects unexpected arguments
    def strict_tool(data: str):
        return f"processed {data}"

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="strict_tool",
            description="Strict tool without thought parameter",
            parameters={
                "type": "object",
                "properties": {"data": {"type": "string"}},
                "required": ["data"],
            },
            fn=strict_tool,
        )
    )
    registry.register(
        Tool(
            name="finish",
            description="Finish tool",
            parameters={
                "type": "object",
                "properties": {"claim": {"type": "string"}, "evidence": {"type": "string"}},
                "required": ["claim"],
            },
            fn=lambda claim, evidence="": {"ok": True, "claim": claim, "evidence": evidence},
        )
    )

    script = [
        {
            "tool_call": {
                "name": "strict_tool",
                "args": {"thought": "Specific internal reasoning", "data": "payload"},
            }
        },
        {
            "tool_call": {
                "name": "finish",
                "args": {"thought": "Done", "claim": "Success", "evidence": "Ran strict tool"},
            }
        },
    ]

    llm = FakeLLM(script)
    res = run_agent("Test task", registry, llm, state, trace, max_steps=5)

    assert res.status == "finished"

    # Verify trace recorded thought and args stripped
    records = Trace.read(trace_file)
    assert records[0]["thought"] == "Specific internal reasoning"
    assert "thought" not in records[0]["args"]
    assert records[0]["args"] == {"data": "payload"}
    assert "processed payload" in records[0]["observation"]
