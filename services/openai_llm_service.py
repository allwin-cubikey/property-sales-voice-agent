"""
OpenAI LLM Service — GPT-4.1 Integration
Exposes the same interface as GroqLLMService so it can be swapped in with a
single config flag (LLM_PROVIDER=openai).

Added: stream_response() async generator for true LLM→TTS streaming pipeline.
"""
import json
import logging
import re
import time
from pydantic import BaseModel, Field, create_model
from typing import AsyncGenerator, Dict, Any, Optional, List, Tuple, Union, Type
from openai import AsyncOpenAI

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streaming chunk boundary constants — tune here without touching logic
# ---------------------------------------------------------------------------
STREAM_MIN_WORDS      = 5    # minimum words before firing on a strong boundary
STREAM_SOFT_MIN_WORDS = 15   # minimum words before firing on a soft boundary (,;)
STREAM_MAX_WORDS      = 25   # force-flush after this many words regardless of boundary
STREAM_FIRST_CHUNK_MAX_WORDS = 6 # force-flush first chunk early to start TTS
STREAM_SENTENCE_CHARS = frozenset(".!?")   # strong boundaries
STREAM_SOFT_CHARS     = frozenset(",;")    # soft boundaries

# Emotion tag pattern (shared with test_local.py / main.py)
_EMOTION_TAG_RE = re.compile(r"\[EMOTION:\s*([a-z]+)\]\s*$", re.IGNORECASE | re.MULTILINE)


class DynamicModelGenerator:
    """Generates Pydantic models dynamically based on field configuration."""

    @staticmethod
    def create_dynamic_model(
        dynamic_fields: Dict[str, Dict[str, Any]],
        base_model_name: str = "DynamicResponseModel",
    ) -> Type[BaseModel]:
        model_fields = {}
        if dynamic_fields:
            for field_name, field_config in dynamic_fields.items():
                field_type = {
                    "string": str,
                    "int": int,
                    "float": float,
                    "boolean": bool,
                }.get(field_config.get("type", "string").lower(), str)
                default = field_config.get("default", "none")
                description = field_config.get("description", "")
                model_fields[field_name] = (
                    Union[field_type, None],
                    Field(default=default, description=description),
                )
        if "response" not in model_fields and "assistant_text" not in model_fields:
            model_fields["response"] = (
                str,
                Field(default="", description="Conversational response to the user"),
            )
        return create_model(base_model_name, __base__=BaseModel, **model_fields)


class OpenAILLMService:
    """
    GPT-4.1 LLM service with the same public interface as GroqLLMService.

    Two generation modes:
      • generate_response() — awaitable, accumulates full stream, returns dict.
        Used by main.py WebSocket path and Groq-compatible callers.
      • stream_response()   — async generator, yields (chunk_text, is_final, emotion)
        tuples in real time as tokens arrive. Used by test_local.py for low-latency TTS.
    """

    # Mirrors GroqLLMService.PROPERTY_INFO_FIELDS
    PROPERTY_INFO_FIELDS = {
        "intent": {
            "type": "string",
            "description": "Intent classification",
            "default": "flow_progress",
        },
        "assistant_text": {
            "type": "string",
            "description": "Spoken response text",
            "default": "",
        },
        "preferred_bhk": {
            "type": "string",
            "description": "BHK preference",
            "default": "none",
        },
        "visit_date": {
            "type": "string",
            "description": "Visit date",
            "default": "none",
        },
        "visit_time": {
            "type": "string",
            "description": "Visit time",
            "default": "none",
        },
        "visit_confirmed": {
            "type": "string",
            "description": "Visit confirmed",
            "default": "no",
        },
        "callback_scheduled": {
            "type": "string",
            "description": "Callback scheduled",
            "default": "no",
        },
        "end_call": {
            "type": "string",
            "description": "End call flag",
            "default": "no",
        },
    }

    SYSTEM_PROMPT_TEMPLATE = None  # set during initialize()

    def __init__(self, api_key: str, max_history: int = 4):
        self.api_key = api_key
        self.max_history = max_history
        self.client: Optional[AsyncOpenAI] = None
        self.ResponseModel = None
        self.dynamic_fields = None
        self.system_prompt_template = None
        self.conversation_history: List[Dict[str, str]] = []
        self.current_model = config.OPENAI_MODEL
        self.fallback_active = False
        # Populated by stream_response() after the stream ends so callers can
        # retrieve intent, slots, should_end_call, etc. after iterating chunks.
        self.last_response_meta: Dict[str, Any] = {}
        logger.info("OpenAILLMService instance created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(
        self,
        dynamic_fields: Optional[Dict[str, Dict[str, Any]]] = None,
        system_prompt_template: Optional[str] = None,
    ) -> bool:
        """Initialize the service (creates AsyncOpenAI client and response model)."""
        try:
            self.client = AsyncOpenAI(api_key=self.api_key)
            self.dynamic_fields = dynamic_fields or self.PROPERTY_INFO_FIELDS
            self.system_prompt_template = (
                system_prompt_template or self.SYSTEM_PROMPT_TEMPLATE or ""
            )
            self.ResponseModel = DynamicModelGenerator.create_dynamic_model(
                self.dynamic_fields, "PropertyEnquiryResponseModel"
            )
            logger.info(
                f"[INIT] OpenAI LLM service initialized — model: {config.OPENAI_MODEL}"
            )
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize OpenAI LLM service: {e}", exc_info=True)
            return False

    async def close(self):
        """Close the service and release resources."""
        if self.client:
            await self.client.close()
            logger.info("[CLEANUP] OpenAI client closed")

    # ------------------------------------------------------------------
    # Prompt helpers (mirrors GroqLLMService interface)
    # ------------------------------------------------------------------

    def format_system_prompt(self, **format_values) -> str:
        """Format system prompt template with runtime values."""
        if not self.system_prompt_template or "{" not in self.system_prompt_template:
            return self.system_prompt_template or ""
        default_values = {}
        if hasattr(config, "AGENT_NAME"):
            default_values["agent_name"] = config.AGENT_NAME
        if hasattr(config, "COMPANY_NAME"):
            default_values["company_name"] = config.COMPANY_NAME
        merged = {**default_values, **format_values}
        try:
            return self.system_prompt_template.format(**merged)
        except KeyError as e:
            logger.error(f"[ERROR] Missing key in system prompt: {e}")
            return self.system_prompt_template

    def generate_system_prompt(self, formatted_system_prompt: str) -> str:
        """Pass through — prompt already includes output format."""
        return formatted_system_prompt

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-(self.max_history * 2):]
            logger.info(
                f"[HISTORY] Trimmed to {len(self.conversation_history)} messages"
            )

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return a copy of the current conversation history."""
        return self.conversation_history.copy()

    # ------------------------------------------------------------------
    # Core generation — non-streaming (generate_response)
    # Used by: main.py WebSocket path, Groq-compatible callers
    # ------------------------------------------------------------------

    async def generate_response(
        self,
        user_input: str,
        format_values: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response using GPT-4.1 with streaming.

        Accumulates the full token stream then parses JSON. Returns a dict
        with the same shape as GroqLLMService.generate_response().
        """
        if not self.client or not self.ResponseModel:
            raise ValueError("OpenAI LLM service not initialized")

        if format_values is None:
            format_values = {}

        generate_start = time.time()

        try:
            formatted_prompt = self.format_system_prompt(**format_values)
            system_prompt = self.generate_system_prompt(formatted_prompt)

            history = list(conversation_history or self.get_conversation_history())
            if len(history) > 6:
                history = history[-6:]

            messages = [
                {"role": "system", "content": system_prompt},
                *history,
                {"role": "user", "content": user_input},
            ]

            api_start = time.time()
            accumulated = ""

            stream = await self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                top_p=config.LLM_TOP_P,
                max_tokens=config.MAX_LLM_TOKENS,
                response_format={"type": "json_object"},
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    accumulated += delta

            api_time = time.time() - api_start
            logger.info(
                f"[TIMING] GPT-4.1 stream completed in {api_time:.3f}s "
                f"({len(accumulated)} chars)"
            )

            if not accumulated.strip():
                logger.warning("[OPENAI] Empty response — using default")
                return self._get_default_response()

            return self._parse_response(accumulated, generate_start)

        except Exception as e:
            logger.error(f"[ERROR] GPT-4.1 generation failed: {e}", exc_info=True)
            return self._get_default_response()

    # ------------------------------------------------------------------
    # Core generation — TRUE STREAMING (stream_response)
    # Used by: test_local.py for low-latency TTS pipeline
    # ------------------------------------------------------------------

    async def stream_response(
        self,
        user_input: str,
        format_values: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[Tuple[str, bool, Optional[str]], None]:
        """
        Stream handler that now accumulates the FULL response before yielding.
        Reverted from chunked streaming to full-response streaming for Deepgram WebSocket.
        
        Yields exactly once: (full_assistant_text, True, emotion)
        
        This keeps the interface compatible with test_local.py's loop,
        but removes the complexity of partial chunks.
        """
        if not self.client or not self.ResponseModel:
            logger.error("[STREAM] Service not initialized")
            yield "I apologize, I'm having a brief issue.", True, None
            return

        if format_values is None:
            format_values = {}

        self.last_response_meta = self._get_default_response()
        stream_start = time.time()

        try:
            formatted_prompt = self.format_system_prompt(**format_values)
            system_prompt = self.generate_system_prompt(formatted_prompt)

            history = list(conversation_history or self.get_conversation_history())
            if len(history) > 6:
                history = history[-6:]

            messages = [
                {"role": "system", "content": system_prompt},
                *history,
                {"role": "user", "content": user_input},
            ]

            full_accum = ""
            api_start = time.time()

            # We still use stream=True to get the first byte latency advantage 
            # and to monitor the connection health, but we accumulate everything.
            stream = await self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                top_p=config.LLM_TOP_P,
                max_tokens=config.MAX_LLM_TOKENS,
                response_format={"type": "json_object"},
                stream=True,
            )

            async for api_chunk in stream:
                delta = api_chunk.choices[0].delta.content
                if delta:
                    full_accum += delta

            # ── Stream finished ──
            api_time = time.time() - api_start
            logger.info(
                f"[TIMING] GPT-4.1 full stream received in {api_time:.3f}s "
                f"({len(full_accum)} chars)"
            )

            # Extract full text from JSON
            final_text = self._extract_assistant_text_from_json(full_accum)
            
            # Extract Emotion
            emotion: Optional[str] = None
            try:
                m = _EMOTION_TAG_RE.search(final_text)
                if m:
                    emotion = m.group(1).lower()
                    # Strip the tag
                    final_text = final_text[: m.start()].rstrip()
            except Exception as e:
                logger.warning(f"[llm] Emotion parse failed: {e}")

            logger.info(f"[LLM] Stream complete | Final text: {final_text[:50]}... | Emotion: {emotion}")

            # Yield the single, complete result
            yield final_text, True, emotion

            # ── Parse full JSON for meta (intent, slots, end_call, …) ──
            if full_accum.strip():
                try:
                    self.last_response_meta = self._parse_response(full_accum, stream_start)
                except Exception as parse_err:
                    logger.error(f"[STREAM] Meta parse failed: {parse_err}")

        except Exception as e:
            logger.error(f"[ERROR] stream_response failed: {e}", exc_info=True)
            self.last_response_meta = self._get_default_response()
            yield "I apologize, I'm having a brief issue.", True, None


    def _extract_assistant_text_from_json(self, raw: str) -> str:
        """
        Pull assistant_text value out of a complete (or near-complete) JSON string.
        Used as fallback when the streaming word_buffer is empty at stream end.
        """
        try:
            data = json.loads(raw.strip())
            return data.get("assistant_text", "")
        except json.JSONDecodeError:
            # Try regex extraction
            m = re.search(r'"assistant_text"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
            if m:
                return m.group(1).encode("raw_unicode_escape").decode("unicode_escape")
        return ""

    def _parse_response(self, raw: str, generate_start: float) -> Dict[str, Any]:
        """Parse JSON response into the standard return dict."""
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.error(f"[OPENAI] JSON parse failed: {raw[:200]}")
                    return self._get_default_response()
            else:
                logger.error(f"[OPENAI] No JSON found in response: {raw[:200]}")
                return self._get_default_response()

        for field_name, field_cfg in (self.dynamic_fields or {}).items():
            if field_name not in data:
                data[field_name] = field_cfg.get("default", "none")

        try:
            model_instance = self.ResponseModel(**data)
        except Exception as e:
            logger.warning(f"[OPENAI] Pydantic validation warning: {e}")
            model_instance = self.ResponseModel()

        spoken_text = (
            getattr(model_instance, "assistant_text", None)
            or getattr(model_instance, "response", "")
            or ""
        )
        intent = data.get("intent", "flow_progress")
        end_call_val = str(data.get("end_call", "no")).lower()
        should_end = end_call_val in ("yes", "true", "1")

        total_time = time.time() - generate_start
        logger.info(
            f"[OPENAI] Response parsed | intent={intent} | "
            f"end_call={end_call_val} | total={total_time:.3f}s"
        )

        return {
            "response": model_instance,
            "spoken_text": spoken_text,
            "intent": intent,
            "should_end_call": should_end,
            "raw_model_data": model_instance.model_dump(),
            "raw_response": raw,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model_used": config.OPENAI_MODEL,
            "was_fallback": False,
        }

    def _get_default_response(self) -> Dict[str, Any]:
        """Return a safe fallback response when the API fails."""
        fallback_text = "I apologize, I'm experiencing a brief issue. Could you please repeat that?"
        default_values = {}
        if self.dynamic_fields:
            for field in self.dynamic_fields:
                default_values[field] = "none"
        if "assistant_text" in (self.dynamic_fields or {}):
            default_values["assistant_text"] = fallback_text
            default_values["intent"] = "flow_progress"
            default_values["end_call"] = "no"
        else:
            default_values["response"] = fallback_text

        try:
            model_instance = self.ResponseModel(**default_values)
        except Exception:
            model_instance = self.ResponseModel()

        spoken = (
            getattr(model_instance, "assistant_text", None)
            or getattr(model_instance, "response", fallback_text)
        )
        return {
            "response": model_instance,
            "spoken_text": spoken,
            "intent": "flow_progress",
            "should_end_call": False,
            "raw_model_data": model_instance.model_dump(),
            "raw_response": "Error: OpenAI API failed",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model_used": "none",
            "was_fallback": False,
        }
