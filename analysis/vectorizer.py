"""
Event Vectorization Service
Transforms raw events into numerical vectors for sequence analysis
"""

import json
import numpy as np
import config

try:
    from sentence_transformers import SentenceTransformer
    HAS_LOCAL_EMBEDDINGS = True
except ImportError:
    HAS_LOCAL_EMBEDDINGS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class EventVectorizer:
    """Vectorizes events for sequence analysis"""
    
    def __init__(self):
        self.event_type_map = {}
        self.embedding_model = None
        self.embedding_cache = {}
        
        # Initialize embedding model
        if config.EMBEDDING_SERVICE == 'local' and HAS_LOCAL_EMBEDDINGS:
            try:
                self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            except Exception:
                self.embedding_model = None
    
    def build_event_type_encoder(self, events: List[Dict]):
        """Build one-hot encoding map for event types"""
        event_types = set()
        for event in events:
            if event.get('type'):
                event_types.add(event['type'])
        
        self.event_type_map = {et: idx for idx, et in enumerate(sorted(event_types))}
    
    def one_hot_encode_event_type(self, event_type: str) -> np.ndarray:
        """One-hot encode event type"""
        if not self.event_type_map:
            return np.array([])
        
        vector = np.zeros(len(self.event_type_map))
        if event_type in self.event_type_map:
            vector[self.event_type_map[event_type]] = 1.0
        return vector
    
    def embed_text(self, text: str) -> np.ndarray | None:
        """Embed text using configured service (synchronous)"""
        if not text or not text.strip():
            return None
        
        # Check cache
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        embedding = None
        
        # Try local model first
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            except Exception:
                pass
        
        # Fallback to API (synchronous for now)
        if embedding is None:
            if config.EMBEDDING_SERVICE == 'openrouter' and config.OPENROUTER_API_KEY and HAS_REQUESTS:
                embedding = self._embed_via_openrouter(text)
            elif config.EMBEDDING_SERVICE == 'huggingface' and config.HF_TOKEN and HAS_REQUESTS:
                embedding = self._embed_via_huggingface(text)
        
        if embedding is not None:
            self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def _embed_via_openrouter(self, text: str) -> np.ndarray | None:
        """Embed via OpenRouter API"""
        try:
            response = requests.post(
                'https://openrouter.ai/api/v1/embeddings',
                headers={
                    'Authorization': f'Bearer {config.OPENROUTER_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': config.EMBEDDING_MODEL,
                    'input': text
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return np.array(data['data'][0]['embedding'])
        except Exception:
            return None
    
    def _embed_via_huggingface(self, text: str) -> np.ndarray | None:
        """Embed via Hugging Face API"""
        try:
            endpoint = config.HF_ENDPOINT or f'https://api-inference.huggingface.co/pipeline/feature-extraction/{config.EMBEDDING_MODEL}'
            response = requests.post(
                endpoint,
                headers={
                    'Authorization': f'Bearer {config.HF_TOKEN}',
                    'Content-Type': 'application/json'
                },
                json={'inputs': text},
                timeout=10
            )
            response.raise_for_status()
            return np.array(response.json())
        except Exception:
            return None
    
    def vectorize_event(self, event: dict, prompt_text: str | None = None) -> dict:
        """Vectorize a single event"""
        # One-hot encode event type
        event_type_vector = self.one_hot_encode_event_type(event.get('type', ''))
        
        # Embed prompt text (synchronous for now, can be made async)
        prompt_embedding = None
        if prompt_text:
            # For now, use a placeholder - in production this would be async
            # For batch processing, we'll handle embeddings separately
            pass
        
        # Combine vectors
        combined_vector = event_type_vector.tolist()
        if prompt_embedding is not None:
            combined_vector.extend(prompt_embedding.tolist())
        else:
            # Placeholder zeros for prompt embedding (768 dim for all-mpnet-base-v2)
            combined_vector.extend([0.0] * 768)
        
        return {
            'event_id': event.get('id'),
            'timestamp': event.get('timestamp'),
            'event_type': event.get('type'),
            'event_type_vector': event_type_vector.tolist(),
            'prompt_text': prompt_text,
            'combined_vector': combined_vector
        }
    
    def vectorize_sequence(self, events: list[dict], prompts_map: dict[int, dict] = None) -> list[dict]:
        """Vectorize a sequence of events"""
        if prompts_map is None:
            prompts_map = {}
        
        sequence = []
        for event in events:
            prompt_id = event.get('prompt_id')
            prompt_text = None
            if prompt_id and prompt_id in prompts_map:
                prompt_text = prompts_map[prompt_id].get('text')
            
            vectorized = self.vectorize_event(event, prompt_text)
            sequence.append(vectorized)
        
        return sequence
