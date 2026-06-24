"""Dialogue orchestrator for the BankCo Voice Assistant.

Contains two core classes:

    SLMClient
        Stateless wrapper around an OpenAI-compatible inference server
        (llama.cpp, vLLM, Ollama, etc.) hosting the fine-tuned Qwen3-0.6B.

    TextOrchestrator
        Deterministic dialogue manager that sits between the user and the
        SLM.  Handles intent tracking, multi-turn slot elicitation, and
        simulated backend execution.
"""

import json
import random

from openai import OpenAI

from src.config import (
    FUNCTION_REQUIRED_ARGS,
    INDIVIDUAL_SLOT_PROMPTS,
    SUCCESS_TEMPLATES,
    SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# SLM Client — stateless wrapper around an OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
class SLMClient:
    """Lightweight client for a llama.cpp / Ollama / vLLM / remote server."""

    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def invoke(self, conversation_history: list[dict]) -> dict | str:
        """Send the full conversation history to the SLM.

        Returns a parsed function-call dict ``{"name": ..., "arguments": ...}``
        or an error string if no valid tool call could be extracted.
        """
        messages = [SYSTEM_PROMPT] + conversation_history

        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        response = chat_response.choices[0].message

        # --- Path A: proper tool_calls in the response ---
        if response.tool_calls:
            fn = response.tool_calls[0].function
            arguments = fn.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            return {"name": fn.name, "arguments": arguments}

        # --- Path B: model returned JSON in content (fallback) ---
        if response.content:
            try:
                parsed = json.loads(response.content.strip())
                if "name" in parsed:
                    args = parsed.get("arguments", parsed.get("parameters", {}))
                    if isinstance(args, str):
                        args = json.loads(args)
                    return {"name": parsed["name"], "arguments": args}
            except (json.JSONDecodeError, KeyError):
                pass

        return f"No valid tool call in SLM response, model returned {response}"


# ---------------------------------------------------------------------------
# Text Orchestrator — deterministic dialogue manager
# ---------------------------------------------------------------------------
class TextOrchestrator:
    """Deterministic dialogue manager sitting between the user and the SLM.

    Responsibilities:
        1. Forward user utterances to the SLM for intent + slot extraction.
        2. Track the active intent across turns (state machine).
        3. Elicit missing slots with targeted follow-up questions.
        4. Execute the backend action once all required slots are filled.
    """

    def __init__(self, slm_client: SLMClient, debug: bool = False):
        self.slm = slm_client
        self.debug = debug
        self.conversation_history: list[dict] = []

        # State tracker
        self.active_intent: str | None = None
        self.accumulated_slots: dict = {}

    def reset(self) -> None:
        """Clear conversation history and active intent state."""
        self.conversation_history = []
        self.active_intent = None
        self.accumulated_slots = {}

    def process_utterance(self, transcript: str) -> str | None:
        """Process a single user utterance and return the bot response.

        Returns ``None`` when the conversation should end (goodbye / quit).
        """
        if transcript.lower() in ("quit", "exit"):
            return None

        self.conversation_history.append({"role": "user", "content": transcript})
        function_call = self.slm.invoke(self.conversation_history)

        if self.debug:
            print(f"  [DEBUG] SLM returned: {function_call}")

        if isinstance(function_call, str):
            self.conversation_history.append({"role": "assistant", "content": ""})
            return self._generate_clarification_response()

        # --- State management engine ---
        extracted_intent = function_call["name"]
        extracted_args = function_call.get("arguments", {})

        # Ignore generic conversational intents when filling slots for a live action
        conversational_intents = ["greeting", "thank_you", "intent_unclear"]

        if self.active_intent and extracted_intent in conversational_intents:
            pass  # keep the active transaction alive
        else:
            if extracted_intent != self.active_intent:
                self.active_intent = extracted_intent
                self.accumulated_slots = {}

        # Merge newly extracted slots into session memory
        if isinstance(extracted_args, dict):
            for key, value in extracted_args.items():
                if value is not None:
                    self.accumulated_slots[key] = value

        # Format flat history turn for the model's next lookback
        json_text_output = json.dumps(
            {"name": self.active_intent, "arguments": self.accumulated_slots}
        )
        self.conversation_history.append(
            {"role": "assistant", "content": json_text_output}
        )

        # Route using persistent state
        return self._handle_function_call(
            {"name": self.active_intent, "arguments": self.accumulated_slots}
        )

    # -- Internal helpers ---------------------------------------------------

    def _handle_function_call(self, function_call: dict) -> str | None:
        name = function_call["name"]
        arguments = function_call.get("arguments", {})

        if name == "goodbye":
            return None

        if name == "intent_unclear":
            return self._generate_clarification_response()

        # Check for missing required args
        missing = self._get_missing_args(name, arguments)
        if missing:
            return self._generate_slot_elicitation(name, missing, arguments)

        # All slots filled — execute
        return self._execute_and_respond(name, arguments)

    def _get_missing_args(self, function_name: str, arguments: dict) -> list[str]:
        required = FUNCTION_REQUIRED_ARGS.get(function_name, [])
        return [arg for arg in required if arguments.get(arg) is None]

    def _generate_clarification_response(self) -> str:
        capabilities = [
            "check your balance",
            "transfer money",
            "cancel or replace cards",
            "pay bills",
            "report fraud",
            "or connect you to an agent",
        ]
        return (
            "I didn't quite understand that. Could you tell me what you need? "
            f"I can help you {', '.join(capabilities)}."
        )

    def _generate_slot_elicitation(
        self, function: str, missing_args: list[str], current_args: dict
    ) -> str:
        individual = INDIVIDUAL_SLOT_PROMPTS.get(function, {})
        questions = [
            individual.get(arg, f"the {arg.replace('_', ' ')}") for arg in missing_args
        ]
        if len(questions) == 1:
            return f"Could you provide {questions[0]}?"
        return f"Could you provide {', '.join(questions[:-1])}, and {questions[-1]}?"

    def _execute_and_respond(self, function: str, arguments: dict) -> str:
        api_result = self._call_backend_api(function, arguments)
        template = SUCCESS_TEMPLATES.get(function, "Done.")
        response_text = template.format(**arguments, **api_result)

        # Transaction complete — clear state
        self.active_intent = None
        self.accumulated_slots = {}

        return response_text

    def _call_backend_api(self, function: str, arguments: dict) -> dict:
        """Simulate a backend — returns extra data needed by templates."""
        if function == "check_balance":
            return {"balance": round(random.uniform(100, 25_000), 2)}
        return {}
