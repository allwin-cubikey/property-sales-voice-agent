"""
TTS Service Factory
Provides factory pattern for creating Text-to-Speech service instances
"""
import logging
from typing import Dict, Type
from services.tts_base import BaseTTSService
from services.tts_service import CartesiaTTSService
from services.sarvam_tts_service import SarvamTTSService
from services.deepgram_tts_service import DeepgramTTSService
from services.smallest_tts_service import SmallestTTSService

logger = logging.getLogger(__name__)


class TTSServiceFactory:
    """
    Factory class for creating TTS service instances.
    Supports multiple providers with runtime selection.
    """
    
    _providers: Dict[str, Type[BaseTTSService]] = {
        'cartesia': CartesiaTTSService,
        'sarvam': SarvamTTSService,
        'deepgram': DeepgramTTSService,
        'smallest': SmallestTTSService
    }
    
    @classmethod
    def create(cls, provider: str, api_key: str, voice_id: str, **kwargs) -> BaseTTSService:
        """
        Create a TTS service instance for the specified provider.
        
        Args:
            provider: Provider name ('cartesia', 'sarvam')
            api_key: API key for the provider
            voice_id: Voice ID to use
            **kwargs: Additional provider-specific arguments
                     For Cartesia: model_id, speed
                     For Sarvam: language, speed
                     For Deepgram: voice_id
            
        Returns:
            Instance of BaseTTSService
            
        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower()
        
        if provider not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(
                f"Unsupported TTS provider: '{provider}'. "
                f"Available providers: {available}"
            )
        
        service_class = cls._providers[provider]
        service = service_class(api_key=api_key, voice_id=voice_id, **kwargs)
        
        if provider == "smallest":
            logger.info(f"[TTS FACTORY] Provider: Smallest.ai Lightning v2 (voice={voice_id}, 24000Hz)")
        else:
            logger.info(f"[FACTORY] Created {provider} TTS service instance")
        return service
    
    @classmethod
    def register_provider(cls, name: str, service_class: Type[BaseTTSService]):
        """
        Register a new TTS provider.
        
        Args:
            name: Provider name (lowercase)
            service_class: Service class implementing BaseTTSService
            
        Raises:
            TypeError: If service_class doesn't inherit from BaseTTSService
        """
        if not issubclass(service_class, BaseTTSService):
            raise TypeError(
                f"Service class must inherit from BaseTTSService, "
                f"got {service_class.__name__}"
            )
        
        name = name.lower()
        cls._providers[name] = service_class
        logger.info(f"[FACTORY] Registered TTS provider: {name}")
    
    @classmethod
    def list_providers(cls) -> list:
        """
        Get list of available TTS providers.
        
        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
