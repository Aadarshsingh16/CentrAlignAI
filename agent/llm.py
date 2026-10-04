import os
import sys
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

# Load .env if present
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

class LLM:
    """
    Single adapter for Gemini LLM using google-genai SDK.
    Handles message formatting, tool schema conversion, forced tool calls, and retries.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY environment variable is missing or empty.")

        # Default model priority: GEMINI_MODEL env var -> gemini-2.5-flash
        self.model_name = (model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()
        
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

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
        max_retries: int = 4,
    ) -> LLMResponse:
        """
        Executes a chat turn with forced function calling (mode ANY).
        Implements exponential backoff on 429 and 5xx errors.
        """
        from google.genai import types
        from google.genai.errors import APIError

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
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

                # Extract response text and tool call
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
                is_retryable = (
                    "429" in err_str
                    or "500" in err_str
                    or "502" in err_str
                    or "503" in err_str
                    or "504" in err_str
                    or "resource_exhausted" in err_str
                    or "unavailable" in err_str
                )
                if is_retryable and attempt < max_retries - 1:
                    sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                    time.sleep(sleep_time)
                    continue
                else:
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
