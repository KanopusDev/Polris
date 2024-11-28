
import redis
from typing import List, Dict, Optional
import json
import time
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class ConversationContext:
    user_id: str
    language: str
    code_context: str
    last_interaction: float
    history: List[Dict[str, str]]

class ConversationManager:
    def __init__(self, redis_url: str, max_history: int = 10, 
                 context_window: int = 2048, ttl: int = 3600):
        self.redis_client = redis.from_url(redis_url)
        self.max_history = max_history
        self.context_window = context_window
        self.ttl = ttl

    def get_context(self, user_id: str) -> ConversationContext:
        """Retrieve conversation context for a user."""
        key = f"conversation:{user_id}"
        data = self.redis_client.get(key)
        
        if data:
            data = json.loads(data)
            return ConversationContext(**data)
        
        return ConversationContext(
            user_id=user_id,
            language="",
            code_context="",
            last_interaction=time.time(),
            history=[]
        )

    def add_interaction(self, user_id: str, 
                       user_input: str, 
                       assistant_response: str,
                       code_context: Optional[str] = None):
        """Add new interaction to conversation history."""
        context = self.get_context(user_id)
        
        # Update context
        context.last_interaction = time.time()
        if code_context:
            context.code_context = code_context

        # Add to history
        context.history.append({
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": time.time()
        })

        # Maintain max history
        if len(context.history) > self.max_history:
            context.history = context.history[-self.max_history:]

        # Save to Redis
        self._save_context(context)

    def _save_context(self, context: ConversationContext):
        """Save context to Redis with TTL."""
        key = f"conversation:{context.user_id}"
        self.redis_client.setex(
            key,
            self.ttl,
            json.dumps(context.__dict__)
        )

    @contextmanager
    def get_conversation_window(self, user_id: str) -> str:
        """Get formatted conversation history within context window."""
        context = self.get_context(user_id)
        
        # Format conversation history
        formatted_history = []
        total_length = 0
        
        for interaction in reversed(context.history):
            current_length = len(interaction['user']) + len(interaction['assistant'])
            if total_length + current_length > self.context_window:
                break
                
            formatted_history.insert(0, 
                f"User: {interaction['user']}\nAssistant: {interaction['assistant']}")
            total_length += current_length

        yield "\n".join(formatted_history)