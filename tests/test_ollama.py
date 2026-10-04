import json
import pytest
from typing import Any
from agent.llm import OllamaLLM, LLM, LLMError, LLMResponse, make_llm

class FakeResponse:
    def __init__(self, status_code: int = 200, json_data: Any = None, text: str = "", headers: dict = None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text or json.dumps(self._json_data)
        self.headers = headers or {}

    def json(self):
        return self._json_data

class FakeHttpClient:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.requests_received = []

    def post(self, url: str, headers: dict = None, json: dict = None, timeout: int = 60):
        self.requests_received.append({"url": url, "headers": headers, "json": json})
        if not self.responses:
            return FakeResponse(500, text="No more scripted responses")
        return self.responses.pop(0)

def test_message_conversion():
    client = FakeHttpClient([
        FakeResponse(200, {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "test_tool", "arguments": {"x": 1}}}],
            }
        })
    ])

    llm = OllamaLLM(api_key="test-key", model="test-model", http_client=client)

    system = "You are a test assistant."
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Running tool",
            "tool_call": {"name": "my_action", "args": {"val": 42}},
        },
        {"role": "tool", "name": "my_action", "content": "Action output"},
    ]
    tool_schemas = [
        {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
        }
    ]

    llm.chat(system=system, messages=messages, tool_schemas=tool_schemas)

    assert len(client.requests_received) == 1
    req = client.requests_received[0]
    payload = req["json"]

    assert payload["model"] == "test-model"
    ollama_msgs = payload["messages"]
    assert ollama_msgs[0] == {"role": "system", "content": "You are a test assistant."}
    assert ollama_msgs[1] == {"role": "user", "content": "Hello"}
    assert ollama_msgs[2]["role"] == "assistant"
    assert ollama_msgs[2]["tool_calls"][0]["function"] == {"name": "my_action", "arguments": {"val": 42}}
    assert ollama_msgs[3] == {"role": "tool", "name": "my_action", "content": "Action output"}

    assert payload["tools"][0]["function"]["name"] == "test_tool"
    assert req["headers"]["Authorization"] == "Bearer test-key"

def test_tool_call_is_parsed():
    # If multiple tool calls are returned, use the first one only
    client = FakeHttpClient([
        FakeResponse(200, {
            "message": {
                "role": "assistant",
                "content": "Thinking completed",
                "tool_calls": [
                    {"function": {"name": "first_action", "arguments": {"arg1": "val1"}}},
                    {"function": {"name": "second_action", "arguments": {"arg2": "val2"}}},
                ],
            }
        })
    ])

    llm = OllamaLLM(api_key="test-key", model="test-model", http_client=client)
    res = llm.chat("System", [{"role": "user", "content": "Do it"}], [{"name": "first_action", "parameters": {}}])

    assert isinstance(res, LLMResponse)
    assert res.text == "Thinking completed"
    assert res.tool_call == {"name": "first_action", "args": {"arg1": "val1"}}

def test_text_only_response_triggers_one_retry_then_error():
    # Attempt 1: text only
    # Attempt 2 (retry with reminder): text only -> raises LLMError
    client = FakeHttpClient([
        FakeResponse(200, {"message": {"role": "assistant", "content": "Just text 1", "tool_calls": []}}),
        FakeResponse(200, {"message": {"role": "assistant", "content": "Just text 2", "tool_calls": []}}),
    ])

    llm = OllamaLLM(api_key="test-key", model="test-model", http_client=client)

    with pytest.raises(LLMError) as exc_info:
        llm.chat("System", [{"role": "user", "content": "Run action"}], [{"name": "action", "parameters": {}}])

    assert "Ollama model failed to return a tool call after reminder" in str(exc_info.value)
    assert len(client.requests_received) == 2

    # Check that retry request contained the reminder message
    retry_msgs = client.requests_received[1]["json"]["messages"]
    assert retry_msgs[-1]["content"] == "respond with exactly one tool call; use finish when done"

def test_text_only_response_triggers_retry_and_succeeds():
    # Attempt 1: text only
    # Attempt 2: returns tool call -> succeeds!
    client = FakeHttpClient([
        FakeResponse(200, {"message": {"role": "assistant", "content": "Clarifying...", "tool_calls": []}}),
        FakeResponse(200, {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "finish", "arguments": {"claim": "done"}}}],
            }
        }),
    ])

    llm = OllamaLLM(api_key="test-key", model="test-model", http_client=client)
    res = llm.chat("System", [{"role": "user", "content": "Run action"}], [{"name": "finish", "parameters": {}}])

    assert res.tool_call == {"name": "finish", "args": {"claim": "done"}}
    assert len(client.requests_received) == 2

def test_429_handling_and_retry():
    sleeps = []
    fake_sleep = lambda s: sleeps.append(s)

    client = FakeHttpClient([
        FakeResponse(429, text="Rate limit exceeded", headers={"Retry-After": "2"}),
        FakeResponse(200, {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "act", "arguments": {}}}],
            }
        }),
    ])

    llm = OllamaLLM(api_key="test-key", model="test-model", http_client=client, sleep_fn=fake_sleep)
    res = llm.chat("System", [{"role": "user", "content": "Run"}], [{"name": "act", "parameters": {}}])

    assert res.tool_call["name"] == "act"
    assert any(s == 3.0 for s in sleeps)  # 2s + 1s buffer

def test_429_daily_quota_fails_fast():
    client = FakeHttpClient([
        FakeResponse(429, text="daily quota exhausted for test-model, resets in ~12h"),
    ])

    llm = OllamaLLM(api_key="test-key", model="test-model", http_client=client)
    with pytest.raises(LLMError) as exc_info:
        llm.chat("System", [{"role": "user", "content": "Run"}], [{"name": "act", "parameters": {}}])

    assert "daily quota exhausted" in str(exc_info.value)
    # Failed on first try without retrying 6 times
    assert len(client.requests_received) == 1

def test_factory_picks_right_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OLLAMA_API_KEY", "fake-ollama-key")

    # Explicit flag
    g_llm = make_llm(provider="gemini", model="gemini-2.5-flash")
    assert isinstance(g_llm, LLM)
    assert g_llm.model_name == "gemini-2.5-flash"

    o_llm = make_llm(provider="ollama", model="gpt-oss:20b")
    assert isinstance(o_llm, OllamaLLM)
    assert o_llm.model_name == "gpt-oss:20b"

    # Default without env is gemini
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    default_llm = make_llm()
    assert isinstance(default_llm, LLM)

    # Provider from env
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    env_llm = make_llm()
    assert isinstance(env_llm, OllamaLLM)
