"""
Sarvam AI Speech-to-Text Service
Implements BaseSTTService for Sarvam AI's speech-to-text-translate API
"""
import logging
import asyncio
import aiohttp
from typing import Callable, Optional
from services.stt_base import BaseSTTService
import config

logger = logging.getLogger(__name__)


class SarvamSTTService(BaseSTTService):
    """
    Sarvam AI STT service with real-time WebSocket transcription
    Supports speech-to-text with translation capabilities
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Sarvam STT service
        
        Args:
            api_key: Sarvam AI API key
        """
        self.api_key = api_key
        self.callback_function = None
        self.ws = None
        self._ws_session = None
        self._is_connected = False
        self._listen_task = None
        self._processing_lock = asyncio.Lock()
        
        logger.info("[STT] SarvamSTTService instance created")
    
    async def initialize(self, api_key: str, callback: Optional[Callable] = None) -> bool:
        """
        Initialize STT service with callback function for transcription events.
        
        Args:
            api_key: Sarvam API key (can override instance key)
            callback: Async callback function to handle transcribed text
            
        Returns:
            True if initialization successful
        """
        try:
            # Update API key if provided
            if api_key:
                self.api_key = api_key
            
            # Store callback if provided
            if callback:
                self.callback_function = callback
            
            logger.info("[STT] Sarvam STT service initialized")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Sarvam STT initialization failed: {e}", exc_info=True)
            return False
    
    async def start_stream(self, callback: Callable) -> bool:
        """
        Start the audio streaming session with a callback for results.
        
        Args:
            callback: Function to call with transcription results
            
        Returns:
            True if stream started successfully
        """
        try:
            self.callback_function = callback
            
            # Create WebSocket session
            self._ws_session = aiohttp.ClientSession()
            
            # Connect to Sarvam WebSocket endpoint
            ws_url = config.SARVAM_STT_URL
            headers = {
                "api-subscription-key": self.api_key  # Sarvam uses this, not Bearer
            }
            
            logger.info(f"[STT] Connecting to Sarvam WebSocket: {ws_url}")
            
            self.ws = await self._ws_session.ws_connect(
                ws_url,
                headers=headers,
                heartbeat=30.0
            )
            
            self._is_connected = True
            logger.info("[STT] Sarvam WebSocket connected successfully")
            
            # Start listening task
            self._listen_task = asyncio.create_task(self._listen())
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to start Sarvam stream: {e}", exc_info=True)
            self._is_connected = False
            return False
    
    async def _listen(self):
        """
        Listen for transcription results from Sarvam WebSocket
        """
        try:
            logger.info("[STT] Started listening for Sarvam transcriptions")
            
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = msg.json()
                        
                        # Extract transcription text
                        text = data.get("text", "")
                        is_final = data.get("is_final", False)
                        
                        if text:
                            if is_final:
                                # Final transcription - send to callback
                                logger.info(f"[TRANSCRIPTION] Final: {text}")
                                if self.callback_function:
                                    await self.callback_function(text)
                            else:
                                # Interim result - send force stop signal
                                logger.debug(f"[TRANSCRIPTION] Interim: {text}")
                                if self.callback_function:
                                    await self.callback_function("__FORCE_STOP__")
                    
                    except Exception as e:
                        logger.error(f"[ERROR] Error processing Sarvam message: {e}")
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"[ERROR] WebSocket error: {self.ws.exception()}")
                    break
                    
        except asyncio.CancelledError:
            logger.info("[STT] Sarvam listen task cancelled")
        except Exception as e:
            logger.error(f"[ERROR] Error in Sarvam listen loop: {e}", exc_info=True)
        finally:
            self._is_connected = False
    
    async def process_audio(self, audio_chunk: bytes) -> bool:
        """
        Process an audio chunk and send it to Sarvam STT service.

        Args:
            audio_chunk: Raw PCM audio bytes (pcm_s16le, 16kHz) for Sarvam streaming
        """

        if not self._is_connected or not self.ws:
            logger.warning("[STT] Cannot process audio - not connected")
            return False
        
        try:
            # Send audio as binary data
            await self.ws.send_bytes(audio_chunk)
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error sending audio to Sarvam: {e}")
            return False
    
    async def close(self) -> bool:
        """
        Close the STT service and clean up resources.
        
        Returns:
            True if cleanup successful
        """
        try:
            logger.info("[CLEANUP] Closing Sarvam STT service")
            
            self._is_connected = False
            
            # Cancel listen task
            if self._listen_task and not self._listen_task.done():
                self._listen_task.cancel()
                try:
                    await self._listen_task
                except asyncio.CancelledError:
                    pass
            
            # Close WebSocket
            if self.ws and not self.ws.closed:
                await self.ws.close()
                logger.info("[CONNECTION] Sarvam WebSocket closed")
            
            # Close session
            if self._ws_session and not self._ws_session.closed:
                await self._ws_session.close()
                logger.info("[CONNECTION] Sarvam session closed")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error closing Sarvam STT service: {e}", exc_info=True)
            return False
    
    @property
    def is_connected(self) -> bool:
        """
        Check if the STT service is currently connected and ready.
        
        Returns:
            True if connected
        """
        return self._is_connected


# Convenience function for quick initialization
async def create_sarvam_stt_service(api_key: str, callback: Optional[Callable] = None) -> SarvamSTTService:
    """
    Create and initialize a Sarvam STT service instance.
    
    Args:
        api_key: Sarvam AI API key
        callback: Optional callback for transcriptions
        
    Returns:
        Initialized SarvamSTTService instance
    """
    service = SarvamSTTService(api_key=api_key)
    await service.initialize(api_key=api_key, callback=callback)
    
    logger.info("[FACTORY] Created and initialized Sarvam STT service")
    return service
