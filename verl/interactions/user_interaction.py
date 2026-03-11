from dataclasses import dataclass
from dotenv import load_dotenv
import logging
import os
from typing import Any, Optional
from uuid import uuid4
from openai import AsyncOpenAI

from .base import BaseInteraction
from MCPFactory.configs.system_prompt import USER_SYSTEM_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class UserConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    presence_penalty: float = 1.5


class User:
    def __init__(self, config: Optional[UserConfig] = UserConfig()):
        self.base_url = os.environ.get('USER_BASE_URL')
        self.api_key = os.environ.get('USER_API_KEY')
        self.model = os.environ.get("USER_MODEL")
        self.config = config or UserConfig()
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def interact(self, messages: list[dict[str, Any]]) -> str:
        """Interact with the user model and get a response."""
        messages = [{"role": "system", "content": USER_SYSTEM_PROMPT}] + messages
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            presence_penalty=self.config.presence_penalty,
            stream=False
        )
        
        return response.choices[0].message.content


class UserInteraction(BaseInteraction):
    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.user = User()
        self.max_turns = config.get("max_turns", 5)

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:
        """Initialize a new interaction session."""
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": 0.0,
            "turn_count": 0,
            "history": [],
        }
        
        logger.info(f"Started interaction {instance_id}")
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """
        Interaction with the user.
        
        Returns:
            should_terminate: True if the user output stop tokens or reach max turns.
            response_text: The user response.
            reward_score: 1.0 for success, 0.0 for failure, 0.5 otherwise.
            metadata: Additional information.
        """
        if instance_id not in self._instance_dict:
            raise ValueError(f"Unknown instance_id: {instance_id}")
        
        instance = self._instance_dict[instance_id]

        # Extract the latest assistant message
        prompt: Optional[str] = None
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                prompt = item.get("content")
                break
        
        if prompt is None:
            logger.warning(f"No assistant message found for {instance_id}")
            return True, "", 0.0, {"error": "no_assistant_message"}
        
        instance["history"].append({"role": "assistant", "content": prompt})
        
        # Interact with the user
        user_response = await self.user.interact(instance["history"])
        
        # Add to history
        instance["history"].append({"role": "user", "content": user_response})
        instance["turn_count"] += 1

        should_terminate = (
            instance["turn_count"] >= self.max_turns or
            "###SUCCESS###" in user_response or
            "###FAILURE###" in user_response
        )

        reward = await self.calculate_score(instance_id)
        
        logger.info(f"Interaction {instance_id} turn {instance['turn_count']}: "
                   f"terminate={should_terminate}, reward={reward}")
        
        return should_terminate, user_response, reward, {}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        if instance_id not in self._instance_dict:
            return 0.0
        
        instance = self._instance_dict[instance_id]
        user_response = instance["history"][-1]["content"]
        if "###SUCCESS###" in user_response:
            return 1.0
        elif "###FAILURE###" in user_response:
            return 0.0
        else:
            return 0.5

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]