"""
Deepgram Aura-2 WebSocket TTS Service
Implements BaseTTSService for ultra-low latency streaming
"""
import logging
import asyncio
import json
import websockets
import base64
import time
import threading
from typing import Optional, Dict, Any, Callable
from services.tts_base import BaseTTSService
import config

logger = logging.getLogger(__name__)

# Configurable voice - change here if needed
DEEPGRAM_VOICE = "aura-asteria-en"  # Warm, professional US female (default) or check available models

class DeepgramTTSService(BaseTTSService):
    """
    Deepgram TTS service using WebSocket streaming.
    Maintains a persistent connection for the duration of a turn.
    """
    def __init__(self, api_key: str, voice_id: Optional[str] = None):
        self.api_key = api_key or config.DEEPGRAM_API_KEY
        # Use voice_id if provided, otherwise default constant
        self.voice = voice_id or DEEPGRAM_VOICE 
        self.websocket = None
        self._receive_task = None
        self._connection_active = False
        self._first_byte_time = None
        self._first_byte_time = None
        self._connect_time = None
        self.last_text = ""
        self.speed = "normal"
        self._last_send_time = 0
        self._first_byte_received = False

    async def connect(self):
        """
        Open the WebSocket connection to Deepgram.
        Should be called before sending any text.
        """
        try:
            url = (
                f"wss://api.deepgram.com/v1/speak?"
                f"model={self.voice}&"
                f"encoding=linear16&"
                f"sample_rate=16000&"
                f"container=none"
            )
            headers = {
                "Authorization": f"Token {self.api_key}"
            }
            
            logger.info(f"[DEEPGRAM] Connecting to {url.split('?')[0]}...")
            self._connect_time = time.time()
            self.websocket = await websockets.connect(url, additional_headers=headers)
            self._connection_active = True
            
            elapsed = (time.time() - self._connect_time) * 1000
            logger.info(f"[DEEPGRAM] WebSocket connected in {elapsed:.1f}ms")
            
            # Reset timing metrics
            self._first_byte_time = None

        except Exception as e:
            logger.error(f"[DEEPGRAM] Connection failed: {e}")
            self._connection_active = False
            raise

    async def send_text(self, text: str):
        """Send text to Deepgram, reconnecting if necessary."""
        if not self.websocket or not self._connection_active:
            logger.info("[DEEPGRAM] Connection lost or not open, reconnecting...")
            await self.connect()
        
        try:
            # Track send time for accurate TTFB
            self._last_send_time = time.time()
            self._first_byte_received = False
            
            msg = json.dumps({"type": "Speak", "text": text})
            await self.websocket.send(msg)
            # logger.info(f"[DEEPGRAM] Text sent ({len(text)} chars)")
        except Exception as e:
            logger.error(f"[DEEPGRAM] Error sending text: {e}")
            self._connection_active = False
            # Try once to reconnect and resend
            try:
                logger.info("[DEEPGRAM] Attempting to reconnect and resend...")
                await self.connect()
                self._last_send_time = time.time()
                await self.websocket.send(json.dumps({"type": "Speak", "text": text}))
            except Exception as re:
                logger.error(f"[DEEPGRAM] Reconnect failed: {re}")

    async def flush(self):
        """Send Flush command to force synthesis of buffered text."""
        if self.websocket and self._connection_active:
            try:
                await self.websocket.send(json.dumps({"type": "Flush"}))
                logger.info("[DEEPGRAM] Flush sent")
            except Exception as e:
                logger.error(f"[DEEPGRAM] Error sending flush: {e}")

    async def keep_alive(self):
        """Send KeepAlive message to prevent timeout."""
        if self.websocket and self._connection_active:
             try:
                 await self.websocket.send(json.dumps({"type": "KeepAlive"}))
             except Exception as e:
                 logger.error(f"[DEEPGRAM] KeepAlive failed: {e}")
                 self._connection_active = False

    async def close(self):
        """
        Close the WebSocket connection.
        """
        self._connection_active = False
        if self.websocket:
            try:
                await self.websocket.send(json.dumps({"type": "Close"}))
                await self.websocket.close()
                logger.info("[DEEPGRAM] WebSocket closed")
            except Exception as e:
                logger.debug(f"[DEEPGRAM] Close error (ignorable): {e}")
            finally:
                self.websocket = None

    async def receive_audio(self, callback: Callable[[bytes], None]):
        """
        Receive audio stream until Metadata message (end of turn).
        Does NOT close the connection.
        """
        if not self.websocket:
            return

        try:
            total_bytes = 0
            # Iterate over messages. receive_audio returns when it hits Metadata (Request Completed)
            async for message in self.websocket:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        if data.get("type") == "Metadata":
                            # This signals end of the flushed audio segment
                            break
                        if data.get("type") == "Error":
                             logger.error(f"[DEEPGRAM] Error from server: {message}")
                             break
                    except:
                        pass
                
                elif isinstance(message, bytes):
                    # Audio data
                    if not self._first_byte_received:
                        self._first_byte_received = True
                        ttfb = (time.time() - self._last_send_time) * 1000
                        logger.info(f"[DEEPGRAM] First audio bytes received — synthesis TTFB: {ttfb:.1f}ms")
                    
                    total_bytes += len(message)
                    if callback:
                         if asyncio.iscoroutinefunction(callback):
                             await callback(message)
                         else:
                             callback(message)
            
            logger.info(f"[DEEPGRAM] Audio stream complete — total bytes: {total_bytes}")
            
        except Exception as e:
            logger.error(f"[DEEPGRAM] Error receiving audio: {e}")
            self._connection_active = False

    async def close(self):
        """
        Close the WebSocket connection.
        """
        self._connection_active = False
        if self.websocket:
            try:
                await self.websocket.send(json.dumps({"type": "Close"}))
                await self.websocket.close()
                logger.info("[DEEPGRAM] WebSocket closed")
            except Exception as e:
                logger.debug(f"[DEEPGRAM] Close error (ignorable): {e}")
            finally:
                self.websocket = None

    # Standard synthesis interface (fallback/non-streaming)
    async def synthesize(self, text: str, emotion: str = None) -> Optional[bytes]:
        """
        Fallback implementation for one-shot synthesis.
        For true streaming, use connect/send_text/receive_audio.
        """
        # We can implement this by open/send/receive/close loop if needed,
        # but the request is for streaming pipeline. 
        # For now, return None or implement simple wrapper.
        logger.warning("[DEEPGRAM] synthesize() called - prefer streaming pipeline")
        return None

    # ─── Abstract Methods Implementation ───────────────────────────

    async def initialize(self) -> bool:
        """Initialize resource (Deepgram specific setup)."""
        logger.info("[DEEPGRAM] Service initialized")
        return True

    async def stop(self):
        """Stop current playback/stream."""
        # For persistent connection, 'stop' might imply clearing audio or interrupting?
        # But here we just close for safety if requested
        await self.close()

    def set_speed(self, speed: str):
        """Set speed (stored but requires URL rebuild to take effect if changed)."""
        self.speed = speed

    async def get_last_spoken_text(self) -> str:
        """Return last text sent."""
        return self.last_text

    async def synthesize(self, text: str, send_audio_callback: Callable, speed: str = None) -> bool:
        """
        One-shot synthesis wrapper for compatibility.
        """
        try:
            # Ensure connected
            if not self.websocket or not self._connection_active:
                 await self.connect()
            
            await self.send_text(text)
            self.last_text = text
            await self.flush()
            
            # Note: For one-shot synthesize, we might want to Close?
            # But if we want persistence, we keep it open.
            # However, this method is usually for legacy/one-off.
            # We will use receive_audio which waits for Metadata.
            
            await self.receive_audio(send_audio_callback)
            
            # We do NOT close here anymore if we want persistance.
            # But synthesize() is typically "do it and done".
            # The caller usually closes if they want.
            return True
        except Exception as e:
            logger.error(f"[DEEPGRAM] One-shot synthesis failed: {e}")
            return False
