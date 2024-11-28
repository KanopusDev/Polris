
from typing import Dict, Any, Optional
import torch
from collections import OrderedDict
import threading
import time

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self._lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
            
    def put(self, key: str, value: Any):
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

class ModelOutputCache:
    def __init__(self, model_name: str, cache_size: int = 1000):
        self.embedding_cache = LRUCache(cache_size)
        self.attention_cache = LRUCache(cache_size)
        self.model_name = model_name
        self.stats = {'hits': 0, 'misses': 0}
        
    def cache_forward_pass(self, input_ids: torch.Tensor, outputs: Any):
        """Cache the results of a forward pass."""
        cache_key = self._get_cache_key(input_ids)
        self.embedding_cache.put(f"{cache_key}_emb", outputs.hidden_states)
        if hasattr(outputs, 'attentions'):
            self.attention_cache.put(f"{cache_key}_att", outputs.attentions)
            
    def get_cached_outputs(self, input_ids: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve cached outputs for input if available."""
        cache_key = self._get_cache_key(input_ids)
        hidden_states = self.embedding_cache.get(f"{cache_key}_emb")
        attentions = self.attention_cache.get(f"{cache_key}_att")
        
        if hidden_states is not None:
            self.stats['hits'] += 1
            return {
                'hidden_states': hidden_states,
                'attentions': attentions
            }
        
        self.stats['misses'] += 1
        return None
        
    def _get_cache_key(self, input_ids: torch.Tensor) -> str:
        """Generate a unique cache key for input tensor."""
        return f"{self.model_name}_{hash(input_ids.cpu().numpy().tobytes())}"
        
    def get_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        return self.stats.copy()