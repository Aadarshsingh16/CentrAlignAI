import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional
from agent.env import Environment
from agent.llm import LLM, LLMError
from agent.state import AgentState
from agent.tools import ToolRegistry
from agent.trace import Trace
from agent.verifier import compute_diff, verify

def _hash_observation(obs: str) -> str:
    """Computes a stable hash of an observation string to capture situational state."""
    return hashlib.sha256(obs.encode("utf-8", errors="replace")).hexdigest()[:16]

@dataclass
class AgentResult:
    """Outcome of an agent execution loop."""
    status: str  # 'finished', 'max_steps', 'stuck', 'llm_error'
    claim: str
    evidence: str
    steps: int
    verified: Optional[bool] = None
    verdict: Optional[dict[str, Any]] = None
    diff: Optional[dict[str, Any]] = None

SYSTEM_PROMPT = """You are an autonomous worker designed to accomplish user goals using the provided tools.
Follow these operational guidelines:
1. Inspect before acting: examine existing files, directories, or web pages before modifying them.
2. Read all observations and error messages attentively; adapt your actions immediately if an operation fails.
3. Never guess missing or ambiguous parameters; use the ask_user tool to clarify when needed. When more than one option plausibly matches the information you have (for example several similar entries in a list or dropdown), call ask_user to choose before acting; do not pick one yourself.
4. Store key facts and findings into working memory using the remember tool.
5. Some actions require human approval; if an action is denied, do not retry the same action, but adapt or report the refusal.
6. Only take actions needed for the user's request; if a step fails or can't be verified, report it honestly instead of trying broad or destructive workarounds (for example deleting or resetting existing data).
7. Declare completion by calling finish only when the user's goal has been fully achieved, supplying concrete evidence."""

def _augment_tool_schemas(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Generically adds a required 'thought' string parameter to every tool schema.
    Forces the model to reason about its next action before invoking the tool.
    """
    augmented = []
    for s in schemas:
        schema_copy = copy.deepcopy(s)
        params = schema_copy.get("parameters", {})
        if not params:
            params = {"type": "object", "properties": {}}
        
        props = params.get("properties", {})
        props["thought"] = {
            "type": "string",
            "description": "Short explanation of your rationale and why this action was chosen.",
        }
        params["properties"] = props

        req = params.get("required", [])
        if "thought" not in req:
            req.append("thought")
        params["required"] = req

        schema_copy["parameters"] = params
        augmented.append(schema_copy)
    return augmented

def _trim_older_observations(messages: list[dict[str, Any]], keep_recent: int = 8) -> list[dict[str, Any]]:
    """
    Keeps the most recent tool observations in full while truncating older ones.
    Prevents token bloat during long task execution trajectories.
    """
    trimmed = []
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    cutoff_index = tool_indices[-keep_recent] if len(tool_indices) > keep_recent else -1

    for idx, msg in enumerate(messages):
        if msg.get("role") == "tool" and idx < cutoff_index:
            m_copy = dict(msg)
            content = str(m_copy.get("content", ""))
            if len(content) > 200:
                m_copy["content"] = content[:200] + "... [earlier observation truncated]"
            trimmed.append(m_copy)
        else:
            trimmed.append(msg)
    return trimmed

def run_agent(
    task: str,
    registry: ToolRegistry,
    llm: Any,  # LLM or test FakeLLM
    state: AgentState,
    trace: Trace,
    context: str = "",
    max_steps: int = 40,
    env: Optional[Environment] = None,
) -> AgentResult:
    """
    Executes the task-agnostic autonomous agent loop.
    Enforces loop safety: max steps, repeated action detection, and structured trace logging.
    Verifies declared task outcomes independently against environment ground truth.
    """
    full_system = SYSTEM_PROMPT
    if context.strip():
        full_system += f"\n\nContext Information:\n{context.strip()}"

    tool_schemas = _augment_tool_schemas(registry.schemas())

    initial_prompt = f"Goal:\n{task}\n\nInitial State:\n{state.to_prompt()}"
    messages: list[dict[str, Any]] = [{"role": "user", "content": initial_prompt}]

    action_history: list[tuple[str, str, str]] = []
    last_observation: str = ""
    step_count = 0

    before_state = None
    if env is not None:
        before_state = env.ground_truth()

    verification_rounds = 0

    while step_count < max_steps:
        step_count += 1

        # Prepare messages with trimmed older observations
        active_messages = _trim_older_observations(messages, keep_recent=8)

        # Call LLM
        try:
            response = llm.chat(
                system=full_system,
                messages=active_messages,
                tool_schemas=tool_schemas,
            )
        except Exception as e:
            trace.append(
                step=step_count,
                thought="LLM execution failed",
                action="error",
                args={},
                observation=f"LLM Error: {e}",
                state=state.to_dict(),
            )
            return AgentResult(
                status="llm_error",
                claim="",
                evidence=f"Encountered LLM error: {e}",
                steps=step_count,
            )

        tool_call = response.tool_call
        if not tool_call:
            trace.append(
                step=step_count,
                thought="No tool call returned",
                action="none",
                args={},
                observation="Model did not invoke a tool.",
                state=state.to_dict(),
            )
            return AgentResult(
                status="stuck",
                claim="Model failed to return a tool call.",
                evidence="",
                steps=step_count,
            )

        action_name = tool_call.get("name", "")
        raw_args = tool_call.get("args", {}) or {}
        args = dict(raw_args)

        # Extract reasoning thought before tool execution
        thought = args.pop("thought", "").strip()
        if not thought and response.text:
            thought = response.text.strip()

        # Check for repetitive action loops (action + args + state before action)
        action_signature = (
            action_name,
            json.dumps(args, sort_keys=True),
            _hash_observation(last_observation),
        )
        repeat_count = 1
        for prev in reversed(action_history):
            if prev == action_signature:
                repeat_count += 1
            else:
                break
        action_history.append(action_signature)

        if repeat_count >= 3:
            msg = f"Loop detected: action '{action_name}' with identical arguments repeated 3 times."
            trace.append(
                step=step_count,
                thought=thought,
                action=action_name,
                args=args,
                observation=msg,
                state=state.to_dict(),
            )
            return AgentResult(
                status="stuck",
                claim="Aborted due to repeated identical actions.",
                evidence=msg,
                steps=step_count,
            )

        # Record assistant turn in message history
        messages.append({
            "role": "assistant",
            "content": response.text or f"Invoking {action_name}",
            "raw": response.raw,
            "tool_call": tool_call,
        })

        # Execute tool
        exec_res = registry.execute(action_name, args)
        raw_observation = exec_res.get("observation", str(exec_res))
        observation = raw_observation

        # Handle finish verification flow
        if action_name == "finish" and exec_res.get("ok"):
            claim = exec_res.get("claim", "")
            evidence = exec_res.get("evidence", "")

            if env is None:
                trace.append(
                    step=step_count,
                    thought=thought,
                    action=action_name,
                    args=args,
                    observation=observation,
                    state=state.to_dict(),
                )
                return AgentResult(
                    status="finished",
                    claim=claim,
                    evidence=evidence,
                    steps=step_count,
                    verified=None,
                    verdict={"status": "skipped", "reasons": "No environment provided for verification."},
                    diff=None,
                )

            after_state = env.ground_truth()
            diff = compute_diff(before_state, after_state)
            run_summary = {
                "approvals": state.approvals,
                "denials": state.denials,
                "asks": state.asks,
                "denial_notes": list(state.denial_notes),
            }

            v = verify(
                task=task,
                claim=claim,
                evidence=evidence,
                before=before_state,
                after=after_state,
                diff=diff,
                run_summary=run_summary,
                llm=llm,
            )

            if not v.achieved or not v.claim_accurate:
                verification_rounds += 1
                obs_mismatch = (
                    f"Verification mismatch (round {verification_rounds}/2): {v.reasons}\n"
                    f"Ground-truth diff: {json.dumps(diff, default=str)}\n"
                    f"Please address this discrepancy or adjust your claim before finishing."
                )
                state.add_step(f"Step {step_count}: finish verification failed -> {v.reasons[:60]}")
                trace.append(
                    step=step_count,
                    thought=thought,
                    action=action_name,
                    args=args,
                    observation=obs_mismatch,
                    state=state.to_dict(),
                )

                if verification_rounds < 2:
                    obs_payload = f"{obs_mismatch}\n\nUpdated Memory:\n{state.to_prompt()}"
                    messages.append({
                        "role": "tool",
                        "name": action_name,
                        "content": obs_payload,
                    })
                    last_observation = obs_mismatch
                    continue

                # Stop after 2 failed rounds
                return AgentResult(
                    status="finished",
                    claim=claim,
                    evidence=evidence,
                    steps=step_count,
                    verified=False,
                    verdict=v.to_dict(),
                    diff=diff,
                )

            # Verification passed
            trace.append(
                step=step_count,
                thought=thought,
                action=action_name,
                args=args,
                observation=observation,
                state=state.to_dict(),
            )
            return AgentResult(
                status="finished",
                claim=claim,
                evidence=evidence,
                steps=step_count,
                verified=True,
                verdict=v.to_dict(),
                diff=diff,
            )

        # Inject warning into observation on second consecutive repeat
        if repeat_count == 2:
            observation += (
                "\n\n[WARNING: You have executed this exact same action with the same arguments twice. "
                "Do not repeat it again. Please adjust your parameters or choose a different approach.]"
            )

        # Update working state
        state.add_step(f"Step {step_count}: {action_name} -> {observation[:80]}")

        # Write to JSONL trace
        trace.append(
            step=step_count,
            thought=thought,
            action=action_name,
            args=args,
            observation=observation,
            state=state.to_dict(),
        )

        # Append tool observation with current memory snapshot
        obs_payload = f"{observation}\n\nUpdated Memory:\n{state.to_prompt()}"
        messages.append({
            "role": "tool",
            "name": action_name,
            "content": obs_payload,
        })
        last_observation = raw_observation

    # Max steps reached without finish
    after_state = env.ground_truth() if env is not None else None
    diff = compute_diff(before_state, after_state) if env is not None else None
    return AgentResult(
        status="max_steps",
        claim="Maximum steps reached before goal completion.",
        evidence="",
        steps=step_count,
        verified=False if env is not None else None,
        verdict={"achieved": False, "claim_accurate": False, "reasons": "Maximum execution steps reached before task completion."} if env is not None else None,
        diff=diff,
    )
