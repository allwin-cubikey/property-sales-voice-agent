"""
LLM Service - Groq Integration with Dynamic Field Extraction
Handles property enquiry conversations with intelligent response generation and structured data extraction
"""
import logging
import time
import json
import aiohttp
from pydantic import BaseModel, Field, create_model
from typing import Dict, Any, Optional, List, Union, Type
import config

logger = logging.getLogger(__name__)


class DynamicModelGenerator:
    """Generates Pydantic models dynamically based on field configuration"""
    
    @staticmethod
    def create_dynamic_model(
        dynamic_fields: Dict[str, Dict[str, Any]], 
        base_model_name: str = 'DynamicResponseModel'
    ) -> Type[BaseModel]:
        """
        Dynamically create a Pydantic model based on configuration.
        
        Args:
            dynamic_fields: Dictionary of field configurations
            base_model_name: Name for the generated model
            
        Returns:
            Dynamically created Pydantic model class
        """
        start_time = time.time()
        model_fields = {}
        
        # Only add fields if dynamic_fields is not empty
        if dynamic_fields:
            for field_name, field_config in dynamic_fields.items():
                # Determine field type
                field_type = {
                    'string': str,
                    'int': int,
                    'float': float,
                    'boolean': bool
                }.get(field_config.get('type', 'string').lower(), str)
                
                # Determine default value
                default = field_config.get('default', 'none')
                description = field_config.get('description', '')
                
                # Create field
                model_fields[field_name] = (
                    Union[field_type, None], 
                    Field(default=default, description=description)
                )
        
        # NOTE: 'assistant_text' is the conversational output field in the new schema
        # Only add default 'response' field if neither 'response' nor 'assistant_text' is in dynamic_fields
        if 'response' not in model_fields and 'assistant_text' not in model_fields:
            model_fields['response'] = (
                str, 
                Field(default='', description='Conversational response to the user')
            )
        
        # Dynamically create the model
        model = create_model(base_model_name, __base__=BaseModel, **model_fields)
        
        create_time = time.time() - start_time
        logger.info(f"[TIMING] Dynamic model creation took {create_time:.3f}s")
        return model


class GroqLLMService:
    """
    Enhanced LLM service with structured data extraction for property enquiry agent
    """
    
    # Define dynamic fields for property information extraction
    # NOTE: conversation_stage removed — backend is sole authority on stage
    # 'response' field renamed to 'assistant_text' for voice output
    PROPERTY_INFO_FIELDS = {
        "intent": {
            "type": "string",
            "description": "Intent classification",
            "default": "flow_progress"
        },
        "assistant_text": {
            "type": "string",
            "description": "Spoken response text",
            "default": ""
        },
        "preferred_bhk": {
            "type": "string",
            "description": "BHK preference",
            "default": "none"
        },
        "visit_date": {
            "type": "string",
            "description": "Visit date",
            "default": "none"
        },
        "visit_time": {
            "type": "string",
            "description": "Visit time",
            "default": "none"
        },
        "visit_confirmed": {
            "type": "string",
            "description": "Visit confirmed",
            "default": "no"
        },
        "callback_scheduled": {
            "type": "string",
            "description": "Callback scheduled",
            "default": "no"
        },
        "end_call": {
            "type": "string",
            "description": "End call flag",
            "default": "no"
        }
    }
    
    
    def __init__(self, api_key: str, max_history: int = 4):
        """
        Initialize Groq LLM service
        
        Args:
            api_key: Groq API key
            max_history: Maximum conversation history to maintain
        """
        self.api_key = api_key
        self.max_history = max_history
        self.session = None
        self.ResponseModel = None
        self.dynamic_fields = None
        self.system_prompt_template = None
        self.conversation_history = []  # In-memory conversation history
        self.current_model = None  # ADDED for fallback info
        self.fallback_active = False  # ADDED for fallback info
        
        logger.info("GroqLLMService instance created")
    
    async def initialize(
        self, 
        dynamic_fields: Optional[Dict[str, Dict[str, Any]]] = None,
        system_prompt_template: Optional[str] = None
    ) -> bool:
        """
        Initialize the LLM service with dynamic fields and system prompt
        
        Args:
            dynamic_fields: Optional custom fields (uses PATIENT_INFO_FIELDS if None)
            system_prompt_template: Optional custom prompt (uses default if None)
            
        Returns:
            True if initialization successful
        """
        init_start = time.time()
        try:
            # Create aiohttp session for API calls
            self.session = aiohttp.ClientSession()
            
            # Use provided fields or default property info fields
            self.dynamic_fields = dynamic_fields or self.PROPERTY_INFO_FIELDS
            
            # Use provided template or default property agent template
            self.system_prompt_template = system_prompt_template or self.SYSTEM_PROMPT_TEMPLATE
            
            # Create dynamic response model
            model_start = time.time()
            self.ResponseModel = DynamicModelGenerator.create_dynamic_model(
                self.dynamic_fields, 
                "PropertyEnquiryResponseModel"
            )
            model_time = time.time() - model_start
            logger.info(f"[TIMING] Response model creation took {model_time:.3f}s")
            
            init_time = time.time() - init_start
            logger.info(f"[INIT] LLM service initialized in {init_time:.3f}s")
            logger.info(f"[INIT] Dynamic fields: {list(self.dynamic_fields.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize Groq LLM service: {e}", exc_info=True)
            return False
    
    def format_system_prompt(self, **format_values) -> str:
        """
        Format system prompt template with configuration values
        
        Args:
            **format_values: Values to format into the template
            
        Returns:
            Formatted system prompt string
        """
        format_start = time.time()
        
        # Default values from config (only if they exist)
        default_values = {}
        
        # Add property-specific values if they exist in config
        if hasattr(config, 'AGENT_NAME'):
            default_values['agent_name'] = config.AGENT_NAME
        if hasattr(config, 'COMPANY_NAME'):
            default_values['company_name'] = config.COMPANY_NAME
        if hasattr(config, 'PROPERTY_TYPES'):
            default_values['property_types'] = ', '.join(config.PROPERTY_TYPES)
        
        # Merge with provided values (provided values take precedence)
        format_values = {**default_values, **format_values}
        
        # Check if template has placeholders
        if '{' not in self.system_prompt_template:
            # If it doesn't have placeholders, it might already be formatted
            return self.system_prompt_template
        
        try:
            # Format the template
            result = self.system_prompt_template.format(**format_values)
            
            format_time = time.time() - format_start
            if format_time > 0.01:
                logger.info(f"[TIMING] System prompt formatting took {format_time:.3f}s")
            
            return result
            
        except KeyError as e:
            logger.error(f"[ERROR] Missing key in system prompt template: {e}")
            return self.system_prompt_template
        except Exception as e:
            logger.error(f"[ERROR] Error formatting system prompt: {e}", exc_info=True)
            return self.system_prompt_template
    
    def generate_system_prompt(self, formatted_system_prompt: str) -> str:
        """
        Generate complete system prompt with JSON schema for structured output.
        
        The new prompt already defines the output format inline, so we only
        need to append a compact field reminder for the model.
        """
        schema_start = time.time()
        
        # The new prompt already has the JSON output format defined.
        # Just return the formatted prompt as-is — no schema injection needed.
        system_prompt = formatted_system_prompt
        
        schema_time = time.time() - schema_start
        logger.info(f"[TIMING] Schema generation took {schema_time:.3f}s")
        
        return system_prompt
    
    def add_to_history(self, role: str, content: str):
        """
        Add message to conversation history with size management
        
        Args:
            role: Message role (user/assistant)
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Trim history if it exceeds max_history
        if len(self.conversation_history) > self.max_history * 2:  # *2 for user+assistant pairs
            # Keep only recent messages
            self.conversation_history = self.conversation_history[-(self.max_history * 2):]
            logger.info(f"[HISTORY] Trimmed conversation history to {len(self.conversation_history)} messages")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        return self.conversation_history.copy()
    
    def reset_conversation(self):
        """Reset conversation history"""
        previous_count = len(self.conversation_history)
        self.conversation_history = []
        logger.info(f"[HISTORY] Reset conversation history (was {previous_count} messages)")
    
    async def generate_response(self, user_input, format_values=None, conversation_history=None):
        """Generate response with automatic fallback on rate limit"""
        generate_start = time.time()
        
        if not self.ResponseModel or not self.session:
            raise ValueError("LLM service not initialized")
        
        if format_values is None:
            format_values = {}
        
        try:
            # Try with primary model
            return await self._generate_with_model(
                user_input=user_input,
                format_values=format_values,
                conversation_history=conversation_history,
                model=config.GROQ_PRIMARY_MODEL,
                is_fallback=False
            )
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error or empty response or validation error
            # We want to fallback if the primary model fails in ANY way (rate limit, empty, garbage)
            should_fallback = config.GROQ_USE_FALLBACK and (
                "rate limit" in error_msg.lower() or 
                "empty response" in error_msg.lower() or
                "validation error" in error_msg.lower() or
                "json" in error_msg.lower() or
                "failed to generate" in error_msg.lower() or
                "timeout" in error_msg.lower() or
                "connection" in error_msg.lower() or
                True  # Fallback on ANY primary model failure
            )
            
            if should_fallback:
                logger.warning(f"[FALLBACK] Primary model failed ({error_msg}), switching to {config.GROQ_FALLBACK_MODEL}")
                
                try:
                    # Retry with fallback model
                    return await self._generate_with_model(
                        user_input=user_input,
                        format_values=format_values,
                        conversation_history=conversation_history,
                        model=config.GROQ_FALLBACK_MODEL,
                        is_fallback=True
                    )
                except Exception as fallback_error:
                    logger.error(f"[FALLBACK] Fallback model also failed: {fallback_error}")
                    # Return default response
                    return self._get_default_response()
            else:
                # Non-rate-limit error or fallback disabled
                logger.error(f"[ERROR] LLM error (no fallback): {error_msg}")
                return self._get_default_response()

    async def _generate_with_model(self, user_input, format_values, conversation_history, model, is_fallback=False):
        """Internal method to generate response with specific model"""
        
        self.current_model = model
        self.fallback_active = is_fallback
        
        if is_fallback:
            logger.info(f"[LLM] Using FALLBACK model: {model}")
        else:
            logger.info(f"[LLM] Using PRIMARY model: {model}")
            
        # Ensure user_name is preserved in format_values for prompt formatting
        if 'user_name' not in format_values and hasattr(config, 'AGENT_NAME'):
            # Try to get from config or use default
            # format_values['user_name'] = "User" # Don't overwrite if not needed
            pass

        # Format system prompt
        formatted_system_prompt = self.format_system_prompt(**format_values)
        system_prompt = self.generate_system_prompt(formatted_system_prompt)
        
        # Get conversation history
        # Always use internal history for consistency
        conversation_history = self.get_conversation_history()

        # Keep last 6 messages (3 exchanges)
        if len(conversation_history) > 6:
            conversation_history = conversation_history[-6:]
            logger.info(f"[OPTIMIZATION] Trimmed history to last 6 messages")
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_history,
            {"role": "user", "content": user_input}
        ]
        
        # Prepare API call
        api_payload = {
            "model": model,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
            "top_p": config.LLM_TOP_P,
            "max_tokens": config.MAX_LLM_TOKENS,
            "stream": False
        }
        
        # Use strict JSON for large/capable models: 70b Llama 3 family AND all Llama 4 variants
        is_large_model = "70b" in model or "llama-4" in model.lower()
        if is_large_model:
            api_payload["response_format"] = {"type": "json_object"}

        # For 8b-instant, we'll parse manually (it often fails strict JSON)
        
        # ANTI-REFLECTION FIX for smaller models and GPT-OSS only
        # Llama 4 Maverick/Scout are capable models — skip this injection for them
        is_small_model = (
            ("8b" in model or "9b" in model or "gemma" in model.lower() or "gpt-oss" in model)
            and "llama-4" not in model.lower()
        )
        if is_small_model:
            # Force conversational response, not schema - be very explicit
            # Append this to the LAST message to ensure recency bias works in our favor
            json_instruction = """

        CRITICAL INSTRUCTION:
        You must respond in JSON format with actual conversation data, NOT the schema definition.

        CORRECT example:
        {"intent": "flow_progress", "assistant_text": "Hi! Are you looking for 3 BHK or 4 BHK?", "preferred_bhk": "none", "visit_date": "none", "visit_time": "none", "visit_confirmed": "no", "callback_scheduled": "no", "end_call": "no"}

        WRONG (do NOT do this):
        {"type": "object", "properties": {"assistant_text": {"type": "string"}}}

        Respond NOW with actual data."""
            
            # Check if last message is user or system and append
            if messages:
                last_msg = messages[-1]
                if last_msg["role"] == "user":
                    last_msg["content"] += json_instruction
                else:
                    # If for some reason last msg isn't user (e.g. function?), append a system reminder
                    messages.append({"role": "system", "content": json_instruction.strip()})
            
            # Reduce complexity for smaller models only
            if "8b" in model or "9b" in model or "gemma" in model.lower():
                api_payload["max_tokens"] = 80
                api_payload["temperature"] = 0.5
            elif "gpt-oss" in model:
                # gpt-oss can handle more but keep temperature stable
                api_payload["temperature"] = 0.3
                # Keep max_tokens from config
            
            # For GPT-OSS specifically, sometimes it needs a nudge in the 'system' slot too if the above fails,
            # but usually recency is key. Let's stick to modifying the last message first.
            
        # Make API call
        api_start = time.time()
        
        async with self.session.post(
            config.GROQ_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json=api_payload
        ) as response:
            result = await response.json()
            api_time = time.time() - api_start
            
            # Check for errors
            if 'error' in result:
                error_msg = result['error'].get('message', 'Unknown error')
                logger.error(f"[ERROR] Groq API error: {error_msg}")
                raise Exception(f"Groq API error: {error_msg}")

            response_content = result["choices"][0]["message"]["content"]
            token_usage = result.get("usage", {})

            # Handle empty response
            if not response_content or response_content.strip() == '':
                logger.warning(f"[LLM] Empty response from {model} (tokens: {token_usage.get('total_tokens', 0)})")
                
                # If this is the primary model, RAISE error to trigger fallback
                if not is_fallback:
                    raise ValueError("Empty response from LLM")
                
                # If this is ALREADY the fallback model, return safe default
                logger.warning(f"[LLM] Fallback model also failed with empty response. Using safe default.")
                return self._get_default_response()

            logger.info(f"[LLM] Model: {model}, Tokens: {token_usage.get('total_tokens', 0)}, Time: {api_time:.3f}s")
             
            # (Redundant check removed)
            
            # Parse response
            try:
                parsed_response = self.ResponseModel.model_validate_json(response_content)
                
                # Get the spoken text — new schema uses assistant_text, old used response
                spoken_text = getattr(parsed_response, 'assistant_text', None) or getattr(parsed_response, 'response', '')
                
                # SPECIAL CHECK: If response is empty but text looks like schema, force fallback
                if not spoken_text and ('"properties"' in response_content or "'properties'" in response_content):
                     logger.warning("[VALIDATION] Empty response with schema keywords - forcing fallback parsing")
                     raise ValueError("Possible schema reflection detected - empty response")
                     
            except Exception as validation_error:
                logger.error(f"[ERROR] Response validation error: {validation_error}")
                # Fallback parsing
                parsed_response = self._parse_fallback_response(response_content)
                spoken_text = getattr(parsed_response, 'assistant_text', None) or getattr(parsed_response, 'response', '')
            
            # Add assistant response to history (use clean spoken text)
            self.add_to_history("assistant", spoken_text)
            
            # Check for call end — prefer explicit end_call field from new schema
            should_end_call = False
            
            # 1. Check end_call field (new architecture)
            if hasattr(parsed_response, 'end_call') and parsed_response.end_call:
                should_end_call = str(parsed_response.end_call).lower() == 'yes'
            
            # 2. Check callback_scheduled field
            if hasattr(parsed_response, 'callback_scheduled') and parsed_response.callback_scheduled:
                if str(parsed_response.callback_scheduled).lower() == 'yes':
                    should_end_call = True
            
            # 3. Fallback: detect farewell phrases in spoken text
            if not should_end_call and spoken_text:
                should_end_call = any(
                    phrase in spoken_text.lower() 
                    for phrase in [
                        "have a wonderful day", "have a great day",
                        "goodbye", "bye", "thank you for your time",
                        "take care", "pleasure speaking"
                    ]
                )
            
            # Log intent if available
            intent = getattr(parsed_response, 'intent', 'unknown')
            logger.info(f"[LLM] Intent: {intent} | End call: {should_end_call}")
            
            return {
                "response": parsed_response,
                "spoken_text": spoken_text,
                "intent": intent,
                "should_end_call": should_end_call,
                "raw_model_data": parsed_response.model_dump(),
                "raw_response": response_content,
                "token_usage": token_usage,
                "model_used": model,
                "was_fallback": is_fallback
            }

    def _get_default_response(self):
        """Return safe default response when all models fail"""
        fallback_text = "I apologize, I'm experiencing technical difficulties. Could you please repeat that?"
        
        default_values = {}
        if self.dynamic_fields:
            for field in self.dynamic_fields.keys():
                default_values[field] = 'none'
        
        # Set the spoken text field (new schema uses assistant_text)
        if 'assistant_text' in (self.dynamic_fields or {}):
            default_values['assistant_text'] = fallback_text
            default_values['intent'] = 'flow_progress'
            default_values['end_call'] = 'no'
        else:
            default_values['response'] = fallback_text
        
        default_response = self.ResponseModel(**default_values)
        
        spoken_text = getattr(default_response, 'assistant_text', None) or getattr(default_response, 'response', fallback_text)
        
        return {
            "response": default_response,
            "spoken_text": spoken_text,
            "intent": "flow_progress",
            "should_end_call": False,
            "raw_model_data": default_response.model_dump(),
            "raw_response": "Error: All models failed",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model_used": "none",
            "was_fallback": False
        }

    def _parse_fallback_response(self, response_content):
        """Attempt to parse response when validation fails or strict mode is off"""
        # Determine which field holds the spoken text
        uses_assistant_text = 'assistant_text' in (self.dynamic_fields or {})
        text_field = 'assistant_text' if uses_assistant_text else 'response'
        fallback_text = "Could you repeat that? I want to make sure I understood correctly."
        
        try:
            # First, try to repair potential truncation (EOF errors)
            text = self._repair_truncated_json(response_content.strip())
            
            # Try to find JSON block if it's embedded in text
            if '{' in text and '}' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                json_str = text[start:end]
                raw_json = json.loads(json_str)
            else:
                raw_json = json.loads(text)
            
            # Check for schema reflection (model returning the schema itself)
            if "properties" in raw_json and "type" in raw_json and raw_json["type"] == "object":
                logger.warning("[EXTRACTOR] Detected schema reflection in response")
                props = raw_json.get("properties", {})
                
                # Check if it's a PURE SCHEMA HALLUCINATION
                response_prop = props.get(text_field) or props.get('response')
                if isinstance(response_prop, dict) and 'type' in response_prop:
                    logger.warning("[EXTRACTOR] Detected PURE schema hallucination (no data). Using safe fallback.")
                    default_values = {text_field: fallback_text}
                    if self.dynamic_fields:
                        for field in self.dynamic_fields.keys():
                            if field != text_field:
                                default_values[field] = 'none'
                    return self.ResponseModel(**default_values)

                raw_json = props
            
            # Handle nested slots object from new schema
            if 'slots' in raw_json and isinstance(raw_json['slots'], dict):
                slots = raw_json['slots']
                # Flatten slots into top-level fields
                for slot_key in ['preferred_bhk', 'visit_date', 'visit_time', 'visit_confirmed', 'callback_scheduled']:
                    if slot_key in slots and slot_key not in raw_json:
                        val = slots[slot_key]
                        raw_json[slot_key] = str(val) if val is not None else 'none'
                
            default_values = {}
            if self.dynamic_fields:
                for field in self.dynamic_fields.keys():
                    val = raw_json.get(field, 'none')
                    if isinstance(val, (dict, list)):
                         val = 'none'
                    if val is None:
                        val = 'none'
                    default_values[field] = str(val)
            
            # Extract spoken text safely
            resp_val = raw_json.get(text_field) or raw_json.get('response') or raw_json.get('assistant_text') or fallback_text
            if isinstance(resp_val, (dict, list)):
                resp_val = fallback_text
                
            default_values[text_field] = str(resp_val)
            return self.ResponseModel(**default_values)
            
        except Exception as e:
            logger.warning(f"[EXTRACTOR] JSON parsing failed: {e}")
            
            # Check if the content looks like a schema
            if ('"type"' in response_content and '"object"' in response_content and 
                '"properties"' in response_content):
                logger.error("[EXTRACTOR] Schema reflection detected in failed parse - using safe fallback")
                default_values = {text_field: "I didn't catch that. Could you repeat?"}
            else:
                default_values = {text_field: response_content.strip()}
                
            # Set other fields to defaults
            if self.dynamic_fields:
                for field in self.dynamic_fields.keys():
                    if field != text_field:
                        default_values[field] = 'none'

            return self.ResponseModel(**default_values)

    def _repair_truncated_json(self, json_str: str) -> str:
        """Attempts to fix truncated JSON strings from a cut-off LLM response"""
        if not json_str or not json_str.startswith("{"):
            return json_str
            
        # Count braces
        open_braces = json_str.count("{")
        close_braces = json_str.count("}")
        
        # If open == close, maybe it's just a validation error, not truncation
        if open_braces <= close_braces:
            return json_str
            
        # If we have an open quote at the end, close it
        if json_str.count('"') % 2 != 0:
            json_str += '"'
            
        # Add missing close braces
        while open_braces > close_braces:
            json_str += ' }'
            close_braces += 1
            
        logger.warning(f"[REPAIR] Successfully closed truncated JSON: {json_str[-20:]}")
        return json_str
    
    async def close(self):
        """Close LLM service and cleanup resources"""
        close_start = time.time()
        
        if self.session:
            await self.session.close()
            close_time = time.time() - close_start
            logger.info(f"[TIMING] LLM service closed in {close_time:.3f}s")
            logger.info(f"[CLEANUP] Session closed, conversation history cleared")


# Convenience function for quick initialization
async def create_llm_service(
    api_key: Optional[str] = None,
    max_history: int = 4
) -> GroqLLMService:
    """
    Create and initialize a Groq LLM service instance
    
    Args:
        api_key: Groq API key (uses config.GROQ_API_KEY if None)
        max_history: Maximum conversation history length
        
    Returns:
        Initialized GroqLLMService instance
    """
    api_key = api_key or config.GROQ_API_KEY
    
    service = GroqLLMService(api_key=api_key, max_history=max_history)
    await service.initialize()
    
    logger.info("[FACTORY] Created and initialized LLM service")
    return service


# Compatibility alias
LLMService = GroqLLMService
