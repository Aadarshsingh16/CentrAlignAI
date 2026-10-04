from types import SimpleNamespace
from typing import Any
import pytest
from agent.llm import LLM, LLMError, extract_retry_delay

class FakeClock:
    def __init__(self, start_time: float = 1000.0):
        self.current_time = start_time
        self.sleep_history: list[float] = []

    def time(self) -> float:
        return self.current_time

    def sleep(self, seconds: float) -> None:
        self.sleep_history.append(seconds)
        self.current_time += seconds

class FakeModels:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.call_count = 0

    def generate_content(self, model: str, contents: Any, config: Any):
        self.call_count += 1
        if not self.responses:
            raise RuntimeError("No more mocked responses")
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

class FakeGenAIClient:
    def __init__(self, responses: list[Any]):
        self.models = FakeModels(responses)

def make_valid_response(fn_name="test_tool", fn_args=None):
    if fn_args is None:
        fn_args = {}
    fc = SimpleNamespace(name=fn_name, args=fn_args)
    part = SimpleNamespace(text="reasoning", function_call=fc)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(
        candidates=[candidate],
        function_calls=[fc],
        text="reasoning",
    )

def test_extract_retry_delay():
    # 1. From details attribute
    e1 = Exception("Rate limit")
    e1.details = [{"@type": "RetryInfo", "retryDelay": "45s"}]
    assert extract_retry_delay(e1) == 45.0

    # 2. From string regex
    e2 = Exception("429 Resource exhausted: retry in 32.5s please")
    assert extract_retry_delay(e2) == 32.5

    # 3. From json string representation
    e3 = Exception("{'error': {'code': 429, 'details': [{'retryDelay': '18s'}]}}")
    assert extract_retry_delay(e3) == 18.0

def test_retry_delay_is_honored():
    clock = FakeClock()
    error_429 = Exception("429 RESOURCE_EXHAUSTED: {'details': [{'retryDelay': '25s'}]}")
    success_resp = make_valid_response()

    client = FakeGenAIClient([error_429, success_resp])
    llm = LLM(
        api_key="test-key",
        client=client,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
        max_rpm=0,  # disable throttle for this test to isolate retryDelay
    )

    resp = llm.chat(system="system", messages=[{"role": "user", "content": "hi"}], tool_schemas=[])

    assert resp.tool_call["name"] == "test_tool"
    assert client.models.call_count == 2
    # Expect 25s + 1s = 26s sleep
    assert len(clock.sleep_history) == 1
    assert clock.sleep_history[0] == 26.0

def test_throttle_spaces_calls():
    clock = FakeClock()
    # 4 RPM => 15s interval
    client = FakeGenAIClient([
        make_valid_response(),
        make_valid_response(),
        make_valid_response(),
    ])

    llm = LLM(
        api_key="test-key",
        client=client,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
        max_rpm=4.0,
    )

    # First call at t=1000: no throttle wait
    llm.chat(system="sys", messages=[{"role": "user", "content": "1"}], tool_schemas=[])
    assert client.models.call_count == 1
    assert len(clock.sleep_history) == 0

    # Fast forward clock only 3 seconds (t=1003)
    clock.current_time += 3.0

    # Second call at t=1003: needs to wait 15 - 3 = 12s
    llm.chat(system="sys", messages=[{"role": "user", "content": "2"}], tool_schemas=[])
    assert client.models.call_count == 2
    assert len(clock.sleep_history) == 1
    assert pytest.approx(clock.sleep_history[0], 0.01) == 12.0

    # Fast forward clock 5 seconds (t=1020)
    clock.current_time += 5.0

    # Third call: elapsed since last call (t=1015) is 5s, needs to wait 10s
    llm.chat(system="sys", messages=[{"role": "user", "content": "3"}], tool_schemas=[])
    assert client.models.call_count == 3
    assert len(clock.sleep_history) == 2
    assert pytest.approx(clock.sleep_history[1], 0.01) == 10.0

def test_non_429_error_is_not_retried_forever():
    clock = FakeClock()
    fatal_error = Exception("400 INVALID_ARGUMENT: Schema mismatch")
    client = FakeGenAIClient([fatal_error, fatal_error])

    llm = LLM(
        api_key="test-key",
        client=client,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
        max_rpm=0,
    )

    with pytest.raises(LLMError) as exc_info:
        llm.chat(system="sys", messages=[{"role": "user", "content": "err"}], tool_schemas=[])

    assert "400 INVALID_ARGUMENT" in str(exc_info.value)
    # Must fail on attempt 1 without retries
    assert client.models.call_count == 1
    assert len(clock.sleep_history) == 0
