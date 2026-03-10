from agents import Agent, ModelSettings, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel

from dataclasses import dataclass
from dotenv import load_dotenv
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from .base import BaseInteraction

load_dotenv()
set_tracing_disabled(True)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class UserConfig:
    temperature: float = 0.6
    top_p: float = 0.95

    max_iterations: int = 4
    enable_verification: bool = False


class User:
    def __init__(self, config: UserConfig = UserConfig()):
        base_url = os.environ.get('USER_BASE_URL')
        api_key = os.environ.get('USER_API_KEY')
        model = os.environ.get("USER_MODEL")    
        self.model = LitellmModel(
            model = f"openai/{model}",
            api_key = api_key,
            base_url = base_url,
        )
        self.model_settings = ModelSettings(
            temperature = config.temperature,
            top_p = config.top_p,
        )
        self.user = Agent(
            name = "User",
            model = self.model,
            model_settings = self.model_settings,
            instructions = "",
        )
    
    async def verify(self):
        pass

    async def interact(self, messages: list[dict[str, Any]]):
        pass


class UserInteraction(BaseInteraction):
    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.user = User()

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": 0.0,
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Generates a response for the current turn of interaction."""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                prompt = item.get("content")
                break
        
        user_response = await self.user.interact(prompt)