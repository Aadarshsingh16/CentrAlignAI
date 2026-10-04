import os
import re
import sys
import time
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional
from dotenv import load_dotenv

load_dotenv()

class LLMError(Exception):
    """Raised when LLM provider encounters unrecoverable API errors."""
    pass

@dataclass
class LLMResponse:
    """Task-agnostic response structure from an LLM call."""
    text: str
    tool_call: Optional[dict[str, Any]]
    raw: Any

def extract_retry_delay(e: Exception) -> Optional[float]:
    """Extracts suggested retry delay in seconds from error details or error message string."""
    # 1. Inspect structured details if present
    details = getattr(e, "details", None)
    if not details and hasattr(e, "response_json") and isinstance(e.response_json, dict):
        details = e.response_json.get("error", {}).get("details", [])
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict) and "retryDelay" in item:
                val = str(item["retryDelay"]).rstrip("s")
                try:
                    return float(val)
                except ValueError:
                    pass

    # 2. Regex fallback for retryDelay pattern
    err_str = str(e)
    m = re.search(r"['\"]retryDelay['\"]\s*:\s*['\"](\d+(?:\.\d+)?)s?['\"]", err_str)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # 3. Regex fallback for 'Please retry in X.Xs'
    m2 = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
    if m2:
        try:
            return float(m2.group(1))
        except ValueError:
            pass

    return None

class LLM:
    """
    Single adapter for Gemini LLM using google-genai SDK.
    Handles message formatting, tool schema conversion, forced tool calls,
    client-side rate-limit throttling, and retry backoff.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client: Any = None,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
        max_rpm: Optional[float] = None,
    ):
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.model_name = (model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn

        # Client-side throttling: default 4 RPM (space calls >= 15s apart)
        if max_rpm is not None:
            self.max_rpm = float(max_rpm)
        else:
            rpm_str = os.getenv("LLM_MAX_RPM", "4").strip()
            self.max_rpm = float(rpm_str) if rpm_str else 4.0

        self.min_interval = (60.0 / self.max_rpm) if self.max_rpm > 0 else 0.0
        self._last_call_time = 0.0

        if client is not None:
            self.client = client
        else:
            if not self.api_key:
                raise LLMError("GEMINI_API_KEY environment variable is missing or empty.")
            from google import genai
            self.client = genai.Client(api_key=self.api_key)

    def _apply_throttle(self) -> None:
        """Enforces minimum interval between consecutive API calls."""
        if self.min_interval <= 0:
            return
        now = self.time_fn()
        elapsed = now - self._last_call_time
        if self._last_call_time > 0 and elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            if wait_time > 0.05:
                print(f"[rate limit] waiting {int(wait_time + 0.99)}s (throttling {self.max_rpm} RPM)...")
                self.sleep_fn(wait_time)
        self._last_call_time = self.time_fn()

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        """
        Converts neutral role messages (user, assistant, tool) into google-genai Content structures.
        Preserves raw assistant content to retain model thoughts and function call signatures.
        """
        from google.genai import types

        contents = []
        for msg in messages:
            role = msg.get("role")
            if role == "assistant":
                if msg.get("raw") is not None:
                    contents.append(msg["raw"])
                else:
                    contents.append(
                        types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=msg.get("content", ""))],
                        )
                    )
            elif role == "tool":
                # In Gemini API, function responses are provided with role='user'
                name = msg.get("name", "tool")
                result_content = msg.get("content", "")
                part = types.Part.from_function_response(
                    name=name,
                    response={"result": result_content},
                )
                contents.append(types.Content(role="user", parts=[part]))
            else:
                # User message
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=msg.get("content", ""))],
                    )
                )
        return contents

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        max_retries: int = 6,
    ) -> LLMResponse:
        """
        Executes a chat turn with forced function calling (mode ANY).
        Applies client-side throttling and exponential/retryDelay backoff on 429 and 5xx errors.
        """
        from google.genai import types

        tools = [{"function_declarations": tool_schemas}]
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=tools,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.0,
        )

        contents = self._convert_messages(messages)

        last_error = None
        for attempt in range(max_retries):
            # Enforce client throttle before dispatching request
            self._apply_throttle()

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

                # Extract response text cleanly
                text = ""
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    text_parts = [p.text for p in response.candidates[0].content.parts if getattr(p, "text", None)]
                    text = "".join(text_parts)

                tool_call = None
                if response.function_calls:
                    fc = response.function_calls[0]
                    tool_call = {
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {},
                    }

                raw_content = None
                if response.candidates and response.candidates[0].content:
                    raw_content = response.candidates[0].content

                return LLMResponse(text=text, tool_call=tool_call, raw=raw_content)

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_429 = "429" in err_str or "resource_exhausted" in err_str
                is_5xx = (
                    "500" in err_str
                    or "502" in err_str
                    or "503" in err_str
                    or "504" in err_str
                    or "unavailable" in err_str
                )

                if (is_429 or is_5xx) and attempt < max_retries - 1:
                    if is_429:
                        delay = extract_retry_delay(e)
                        if delay is not None:
                            wait_sec = min(delay + 1.0, 90.0)
                        else:
                            wait_sec = min((2 ** (attempt + 1)) + random.uniform(1.0, 3.0), 90.0)
                    else:
                        # 5xx error backoff
                        wait_sec = min((2 ** (attempt + 1)) + random.uniform(1.0, 3.0), 90.0)

                    print(f"[rate limit] waiting {int(wait_sec)}s (attempt {attempt + 1}/{max_retries})...")
                    self.sleep_fn(wait_sec)
                    continue
                else:
                    # Non-retryable error or exhausted retries
                    raise LLMError(f"LLM request failed after {attempt + 1} attempt(s): {e}") from e

        raise LLMError(f"LLM request failed after {max_retries} attempts: {last_error}")

def list_available_models():
    """Prints models available to current GEMINI_API_KEY supporting generateContent."""
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        print("GEMINI_API_KEY is not set in environment or .env")
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=key)
    print("Available Gemini models supporting generateContent:")
    count = 0
    for m in client.models.list():
        supported = getattr(m, "supported_actions", []) or []
        if "generateContent" in supported:
            print(f"  - {m.name}")
            count += 1
    print(f"Total: {count} models found.")

if __name__ == "__main__":
    if "--list-models" in sys.argv:
        list_available_models()
    else:
        print("Usage: python -m agent.llm --list-models")
