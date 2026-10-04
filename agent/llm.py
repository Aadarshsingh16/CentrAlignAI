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

                if is_429:
                    delay = extract_retry_delay(e)
                    is_per_day = "perday" in err_str
                    if is_per_day or (delay is not None and delay > 120.0):
                        hours_msg = f"~{max(1, round(delay / 3600))}h" if delay else "tomorrow"
                        raise LLMError(f"daily quota exhausted for {self.model_name}, resets in {hours_msg}") from e

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

GeminiLLM = LLM

class OllamaLLM:
    """
    Adapter for Ollama LLM provider (supporting local or hosted cloud API).
    Handles message formatting, tool schema conversion, tool call parsing,
    tool call enforcement retry, client-side rate-limit throttling, and retry backoff.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        http_client: Any = None,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
        max_rpm: Optional[float] = None,
    ):
        raw_key = api_key or os.getenv("OLLAMA_API_KEY") or os.getenv("ollamapi") or os.getenv("OLLAMA_KEY") or ""
        self.api_key = raw_key.strip()

        raw_base = base_url or os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
        self.base_url = raw_base.strip().rstrip("/")
        if self.base_url.endswith("/api"):
            self.api_base = self.base_url
        else:
            self.api_base = f"{self.base_url}/api"

        self.chat_url = f"{self.api_base}/chat"
        self.tags_url = f"{self.api_base}/tags"

        self.model_name = (model or os.getenv("OLLAMA_MODEL", "gpt-oss:20b")).strip()
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn

        # Client-side throttling: default to 60 RPM if not explicitly configured
        if max_rpm is not None:
            self.max_rpm = float(max_rpm)
        else:
            rpm_str = os.getenv("LLM_MAX_RPM")
            self.max_rpm = float(rpm_str) if rpm_str else 60.0

        self.min_interval = (60.0 / self.max_rpm) if self.max_rpm > 0 else 0.0
        self._last_call_time = 0.0

        if http_client is not None:
            self.http_client = http_client
        else:
            import requests
            self.http_client = requests

    def _apply_throttle(self) -> None:
        """Enforces minimum interval between consecutive API calls."""
        if self.min_interval <= 0:
            return
        now = self.time_fn()
        elapsed = now - self._last_call_time
        if elapsed < self.min_interval:
            wait = self.min_interval - elapsed
            if wait > 1.0:
                print(f"[rate limit] waiting {int(wait)}s (throttling {self.max_rpm} RPM)...")
            self.sleep_fn(wait)
        self._last_call_time = self.time_fn()

    def _format_messages(self, system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        if system.strip():
            formatted.append({"role": "system", "content": system.strip()})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "assistant":
                tc = msg.get("tool_call")
                if tc:
                    call_name = tc.get("name", "")
                    call_args = tc.get("args", {})
                    formatted.append({
                        "role": "assistant",
                        "content": str(content or ""),
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": call_name,
                                    "arguments": call_args,
                                },
                            }
                        ],
                    })
                else:
                    formatted.append({"role": "assistant", "content": str(content or "")})
            elif role == "tool":
                formatted.append({
                    "role": "tool",
                    "name": str(msg.get("name", "")),
                    "content": str(content or ""),
                })
            else:
                formatted.append({"role": "user", "content": str(content or "")})

        return formatted

    def _format_tools(self, tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "parameters": s.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for s in tool_schemas
        ]

    def _post(self, payload: dict[str, Any], max_retries: int = 6) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error = None
        for attempt in range(max_retries):
            self._apply_throttle()
            try:
                resp = self.http_client.post(self.chat_url, headers=headers, json=payload, timeout=60)
                status = resp.status_code
                if status == 200:
                    return resp.json()

                is_429 = status == 429
                is_5xx = 500 <= status < 600

                delay = None
                retry_after = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        pass
                if delay is None:
                    delay = extract_retry_delay(Exception(resp.text))

                resp_text_lower = resp.text.lower()
                if is_429:
                    is_per_day = "perday" in resp_text_lower or "daily quota" in resp_text_lower
                    if is_per_day or (delay is not None and delay > 120.0):
                        hours_msg = f"~{max(1, round(delay / 3600))}h" if delay else "tomorrow"
                        raise LLMError(f"daily quota exhausted for {self.model_name}, resets in {hours_msg}")

                if (is_429 or is_5xx) and attempt < max_retries - 1:
                    if delay is not None:
                        wait_sec = min(delay + 1.0, 90.0)
                    else:
                        wait_sec = min((2 ** (attempt + 1)) + random.uniform(1.0, 3.0), 90.0)
                    print(f"[rate limit] waiting {int(wait_sec)}s (attempt {attempt + 1}/{max_retries})...")
                    self.sleep_fn(wait_sec)
                    continue
                else:
                    raise LLMError(f"Ollama request failed with status {status}: {resp.text}")

            except LLMError:
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_sec = min((2 ** (attempt + 1)) + 1.0, 90.0)
                    self.sleep_fn(wait_sec)
                    continue
                raise LLMError(f"Ollama connection error after {attempt + 1} attempts: {e}") from e

        raise LLMError(f"Ollama request failed after {max_retries} attempts: {last_error}")

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        max_retries: int = 6,
    ) -> LLMResponse:
        """
        Executes a chat turn via Ollama API.
        Enforces tool calls: if model returns text without a tool call, retries once
        with a reminder. If still no tool call, raises LLMError.
        If multiple tool calls are returned, selects the first one.
        """
        formatted_messages = self._format_messages(system, messages)
        tools = self._format_tools(tool_schemas)

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        resp_data = self._post(payload, max_retries=max_retries)
        msg = resp_data.get("message", {})
        text = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if tool_calls:
            first_tc = tool_calls[0]
            fn = first_tc.get("function", {})
            call_name = fn.get("name", "")
            call_args = fn.get("arguments", {})
            if isinstance(call_args, str):
                try:
                    import json
                    call_args = json.loads(call_args)
                except Exception:
                    pass
            return LLMResponse(
                text=text,
                tool_call={"name": call_name, "args": call_args},
                raw=resp_data,
            )

        if not tool_schemas:
            return LLMResponse(text=text, tool_call=None, raw=resp_data)

        # Retry once with short reminder message
        retry_messages = list(formatted_messages)
        retry_messages.append({"role": "assistant", "content": text})
        retry_messages.append({
            "role": "user",
            "content": "respond with exactly one tool call; use finish when done",
        })

        retry_payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": retry_messages,
            "tools": tools,
            "stream": False,
        }
        retry_resp = self._post(retry_payload, max_retries=max_retries)
        r_msg = retry_resp.get("message", {})
        r_text = r_msg.get("content", "")
        r_calls = r_msg.get("tool_calls", [])

        if r_calls:
            first_tc = r_calls[0]
            fn = first_tc.get("function", {})
            call_name = fn.get("name", "")
            call_args = fn.get("arguments", {})
            if isinstance(call_args, str):
                try:
                    import json
                    call_args = json.loads(call_args)
                except Exception:
                    pass
            return LLMResponse(
                text=r_text,
                tool_call={"name": call_name, "args": call_args},
                raw=retry_resp,
            )

        raise LLMError(f"Ollama model failed to return a tool call after reminder: {r_text}")

def make_llm(provider: Optional[str] = None, model: Optional[str] = None, **kwargs: Any) -> Any:
    """
    Factory function for instantiating LLM adapters.
    Provider resolves from: parameter -> LLM_PROVIDER env -> default 'gemini'.
    """
    load_dotenv()
    resolved_provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
    if resolved_provider == "ollama":
        return OllamaLLM(model=model, **kwargs)
    elif resolved_provider == "gemini":
        return LLM(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider '{resolved_provider}'. Supported: 'gemini', 'ollama'")

def list_available_models(provider: str = "gemini"):
    """Prints models available to the active key for the chosen provider."""
    load_dotenv()
    if provider == "ollama":
        key = (os.getenv("OLLAMA_API_KEY") or os.getenv("ollamapi") or os.getenv("OLLAMA_KEY") or "").strip()
        base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com").strip().rstrip("/")
        api_base = base_url if base_url.endswith("/api") else f"{base_url}/api"
        tags_url = f"{api_base}/tags"

        import requests
        headers = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            r = requests.get(tags_url, headers=headers, timeout=15)
            if r.status_code != 200:
                print(f"Error fetching Ollama models ({r.status_code}): {r.text}")
                sys.exit(1)
            models = r.json().get("models", [])
            print(f"Available Ollama models ({base_url}):")
            for m in models:
                print(f"  - {m.get('name')}")
            print(f"Total: {len(models)} models found.")
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            sys.exit(1)
    else:
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

def probe_model(model_name: str, provider: str = "gemini"):
    """Makes a probe call to a model. For Ollama, tests a tool-calling round trip."""
    load_dotenv()
    if provider == "ollama":
        llm = OllamaLLM(model=model_name)
        fake_tool = {
            "name": "add",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        }
        try:
            resp = llm.chat(
                system="You are a helpful calculation assistant. You must call the add tool.",
                messages=[{"role": "user", "content": "What is 2 + 2? Please call the add tool with a=2 and b=2."}],
                tool_schemas=[fake_tool],
            )
            if resp.tool_call and resp.tool_call.get("name") == "add":
                print(f"OK: Tool call succeeded -> {resp.tool_call['name']}({resp.tool_call['args']})")
                print(f"Model {model_name} successfully verified on Ollama.")
            else:
                print(f"Failed: Model did not return expected tool call: {resp}")
        except Exception as e:
            print(f"Error probing Ollama model {model_name}: {e}")
    else:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            print("GEMINI_API_KEY is not set in environment or .env")
            sys.exit(1)

        from google import genai
        client = genai.Client(api_key=key)
        try:
            res = client.models.generate_content(model=model_name, contents="ping")
            txt = res.text.strip() if res.text else "OK"
            print(f"OK ({txt})")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    provider = "gemini"
    if "--provider" in sys.argv:
        p_idx = sys.argv.index("--provider")
        if p_idx + 1 < len(sys.argv):
            provider = sys.argv[p_idx + 1].strip().lower()

    if "--list-models" in sys.argv:
        list_available_models(provider=provider)
    elif "--probe" in sys.argv:
        idx = sys.argv.index("--probe")
        if idx + 1 < len(sys.argv):
            probe_model(sys.argv[idx + 1], provider=provider)
        else:
            print("Usage: python -m agent.llm --probe MODEL [--provider PROVIDER]")
    else:
        print("Usage: python -m agent.llm [--list-models | --probe MODEL] [--provider PROVIDER]")
