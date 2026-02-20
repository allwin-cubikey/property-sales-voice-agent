"""
STT Service - Deepgram Integration with Real-time Transcription
Supports Flux (v2) and Nova-2 (v1) models via SDK v5.
"""
import logging
import asyncio
import json
import time

# SDK v5 imports
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import (
    ListenV2TurnInfoEvent,
    ListenV2ConnectedEvent,
    ListenV2FatalErrorEvent,
    ListenV1ResultsEvent,
)

from services.stt_base import BaseSTTService
import config

logger = logging.getLogger(__name__)

# Flux uses v2 API with built-in turn detection; Nova uses v1 with manual endpointing
_USE_FLUX = config.DEEPGRAM_MODEL.startswith("flux")


class DeepgramSTTService(BaseSTTService):
    """
    Enhanced STT service with real-time streaming transcription.
    Supports Flux (v2 with auto turn-detection) and Nova-2 (v1 with manual endpointing).
    Uses AsyncDeepgramClient with background tasks for connection lifecycle.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.callback_function = None
        
        # Connection state
        self.dg_connection = None
        self._is_connected = False
        self._connection_event = asyncio.Event()
        self._lifecycle_task = None
        self.deepgram_client = None
        
        # V1 Buffer (Nova)
        self.is_finals = []
        self.processing_lock = asyncio.Lock()
        
        # Session info
        self.session_start_time = None
        self.ai_currently_speaking = False
        
        # Audio config
        self._encoding = "linear16"
        self._sample_rate = 16000
        self._last_turn_index = -1
        
        logger.info("[STT] DeepgramSTTService instance created")

    @property
    def is_connected(self) -> bool:
        """Check if the STT service is currently connected."""
        return self._is_connected and self.dg_connection is not None

    async def initialize(self, api_key: str, callback=None, encoding: str = "linear16", sample_rate: int = None):
        """
        Initialize STT service by starting the background connection task.
        """
        init_start = time.time()
        try:
            self._encoding = encoding
            if sample_rate:
                self._sample_rate = sample_rate
            elif encoding == "mulaw":
                self._sample_rate = 8000
            else:
                self._sample_rate = 16000

            logger.info(f"[STT] Initializing Deepgram: model={config.DEEPGRAM_MODEL}, "
                        f"encoding={self._encoding}, sample_rate={self._sample_rate}")
            
            if callback:
                self.callback_function = callback
            
            # Initialize Async Client
            self.deepgram_client = AsyncDeepgramClient(api_key=self.api_key or api_key)
            self._connection_event.clear()
            
            # Start background lifecycle task (handles async with)
            if _USE_FLUX:
                self._lifecycle_task = asyncio.create_task(self._flux_lifecycle())
            else:
                self._lifecycle_task = asyncio.create_task(self._nova_lifecycle())
            
            # Wait for connection to open
            try:
                await asyncio.wait_for(self._connection_event.wait(), timeout=5.0)
                self._is_connected = True
                self.session_start_time = time.time()
                
                init_time = time.time() - init_start
                logger.info(f"[STT] Deepgram initialized in {init_time:.3f}s (model={config.DEEPGRAM_MODEL})")
                return True
            except asyncio.TimeoutError:
                logger.error("[ERROR] Deepgram connection timed out during initialization")
                await self.close()
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize Deepgram STT: {e}", exc_info=True)
            self._is_connected = False
            return False

    async def process_audio(self, audio_chunk: bytes) -> bool:
        """Process audio chunk by sending to Deepgram."""
        if not self.is_connected:
            return False
            
        try:
            # SDK v5: public `send_media` expects object, but `_send` handles bytes directly.
            # Using _send to bypass Pydantic model construction for raw audio performance.
            if self.dg_connection:
                await self.dg_connection._send(audio_chunk)
            return True
        except Exception as e:
            # Only log error if we haven't already marked as disconnected to avoid flooding
            if self._is_connected:
                logger.error(f"[ERROR] Audio send error: {e}")
                self._is_connected = False
            return False

    async def start_stream(self, callback) -> bool:
        if not self.is_connected:
            return False
        self.callback_function = callback
        logger.info("[STT] Stream started with callback")
        return True

    async def close(self) -> bool:
        """Close STT connection and cleanup."""
        logger.info("[CLEANUP] Closing Deepgram STT service")
        self._is_connected = False
        self.callback_function = None
        self.is_finals.clear()
        self._last_turn_index = -1
        
        try:
            # 1. Close the connection explicitly if it exists
            if self.dg_connection:
                # Use a short timeout for connection close
                # This might trigger the keepalive error if already unstable, so we suppress it
                try:
                    await asyncio.wait_for(self.dg_connection.finish(), timeout=1.0)
                except Exception:
                    # Ignore errors during close (connection might already be dead)
                    pass
            
            # 2. Cancel lifecycle task
            if self._lifecycle_task and not self._lifecycle_task.done():
                self._lifecycle_task.cancel()
                try:
                    await asyncio.wait_for(self._lifecycle_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    # Log other errors but don't crash
                    logger.debug(f"[CLEANUP] Lifecycle task error: {e}")
                finally:
                    self._lifecycle_task = None
                    
        except Exception as e:
            logger.error(f"[ERROR] STT Close error: {e}")
            
        self.dg_connection = None
        self.deepgram_client = None
        return True

    # ──────────────────────────────────────────────────────────────
    # Flux v2 Lifecycle (Async Context Manager)
    # ──────────────────────────────────────────────────────────────
    async def _flux_lifecycle(self):
        """Manages Flux v2 connection lifecycle."""
        try:
            # async with handles open/close automatically
            async with self.deepgram_client.listen.v2.connect(
                model=config.DEEPGRAM_MODEL,
                encoding=self._encoding,
                sample_rate=str(self._sample_rate),
                keyterm=["BHK", "crore", "Yelahanka", "Brigade", "Eternia"],
                eot_threshold="0.5"
            ) as connection:
                
                self.dg_connection = connection
                
                # Register events
                connection.on(EventType.OPEN, self._on_open)
                connection.on(EventType.MESSAGE, self._on_flux_message)
                connection.on(EventType.ERROR, self._on_error)
                connection.on(EventType.CLOSE, self._on_close)
                
                # Signal ready
                self._connection_event.set()
                logger.info("[STT] Flux v2 connection established")
                
                # Block until closed (start_listening runs the loop)
                await connection.start_listening()
                
        except asyncio.CancelledError:
            logger.info("[STT] Flux lifecycle cancelled")
        except Exception as e:
            logger.error(f"[ERROR] Flux lifecycle error: {e}", exc_info=True)
            self._is_connected = False
        finally:
            logger.info("[STT] Flux lifecycle ended")

    async def _on_flux_message(self, result, **kwargs):
        """Handle Flux message."""
        try:
            if isinstance(result, ListenV2TurnInfoEvent):
                transcript = result.transcript.strip()
                confidence = result.end_of_turn_confidence
                
                if not transcript:
                    return

                # Deduplicate by turn_index
                if result.turn_index <= self._last_turn_index:
                    logger.debug(f"[STT] Skipping duplicate turn index {result.turn_index}")
                    return

                # Flux Turn Detection
                if confidence >= 0.45:
                    if self.callback_function:
                        logger.info(f"[STT] Turn {result.turn_index} (conf={confidence:.2f}): \"{transcript}\" → accepted")
                        self._last_turn_index = result.turn_index
                        asyncio.create_task(self.callback_function(transcript))
                else:
                    logger.info(f"[STT] Turn {result.turn_index} (conf={confidence:.2f}): \"{transcript}\" → below threshold, discarded")
                    # Interim/Low confidence -> Potential Interruption?
                    if (
                        self.callback_function 
                        and getattr(self, "ai_currently_speaking", False)
                        and len(transcript) > 1
                    ):
                        asyncio.create_task(self.callback_function("__FORCE_STOP__"))
                        
        except Exception as e:
             logger.error(f"[ERROR] Flux message handler: {e}")

    # ──────────────────────────────────────────────────────────────
    # Nova v1 Lifecycle (Legacy)
    # ──────────────────────────────────────────────────────────────
    async def _nova_lifecycle(self):
        """Manages Nova v1 connection lifecycle."""
        try:
            # v1.connect is also an async context manager in SDK v5
            async with self.deepgram_client.listen.v1.connect(
                model=config.DEEPGRAM_MODEL,
                encoding=self._encoding,
                sample_rate=str(self._sample_rate),
                smart_format="true",
                interim_results="true",
                utterance_end_ms="1000",
                endpointing="300",
                keywords=["BHK:3", "crore:3"]
            ) as connection:
                
                self.dg_connection = connection
                
                connection.on(EventType.OPEN, self._on_open)
                connection.on(EventType.MESSAGE, self._on_nova_message)
                connection.on(EventType.ERROR, self._on_error)
                connection.on(EventType.CLOSE, self._on_close)
                
                self._connection_event.set()
                logger.info("[STT] Nova v1 connection established")
                
                await connection.start_listening()
                
        except asyncio.CancelledError:
            logger.info("[STT] Nova lifecycle cancelled")
        except Exception as e:
            logger.error(f"[ERROR] Nova lifecycle error: {e}", exc_info=True)
            self._is_connected = False

    async def _on_nova_message(self, result, **kwargs):
        """Handle Nova message."""
        try:
            if isinstance(result, ListenV1ResultsEvent):
                if not result.channel.alternatives:
                    return
                
                alt = result.channel.alternatives[0]
                sentence = alt.transcript.strip()
                if not sentence:
                    return

                is_final = result.is_final
                is_speech_final = getattr(result, "speech_final", False)

                # Interruption check
                if (
                    not is_final
                    and self.callback_function
                    and getattr(self, "ai_currently_speaking", False)
                ):
                    asyncio.create_task(self.callback_function("__FORCE_STOP__"))
                    return

                if is_final:
                    async with self.processing_lock:
                        self.is_finals.append(sentence)
                        if is_speech_final and self.is_finals:
                            utterance = " ".join(self.is_finals).strip()
                            self.is_finals.clear()
                            if utterance and self.callback_function:
                                logger.info(f"[STT] Nova Speech Final: \"{utterance}\"")
                                asyncio.create_task(self.callback_function(utterance))
                                
        except Exception as e:
            logger.error(f"[ERROR] Nova message handler: {e}")

    # ──────────────────────────────────────────────────────────────
    # Shared Events
    # ──────────────────────────────────────────────────────────────
    async def _on_open(self, *args, **kwargs):
        logger.info("[STT] WebSocket Opened")

    async def _on_close(self, *args, **kwargs):
        logger.info("[STT] WebSocket Closed")

    async def _on_error(self, error, **kwargs):
        logger.error(f"[ERROR] WebSocket Error: {error}")

async def create_stt_service(api_key: str, callback=None) -> DeepgramSTTService:
    service = DeepgramSTTService(api_key=api_key)
    await service.initialize(api_key=api_key, callback=callback)
    return service
