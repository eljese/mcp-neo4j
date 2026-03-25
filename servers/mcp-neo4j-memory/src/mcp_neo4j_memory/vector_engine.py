import logging
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger("mcp_neo4j_memory")


class VectorEngine(ABC):
    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        pass


class GeminiVectorEngine(VectorEngine):
    def __init__(self, api_key: str, model: str = "models/gemini-embedding-001"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent?key={api_key}"

    async def get_embedding(self, text: str) -> list[float]:
        payload = {"model": self.model, "content": {"parts": [{"text": text}]}}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.url, json=payload, timeout=10.0)
                if response.status_code == 200:
                    return response.json().get("embedding", {}).get("values", [])
                else:
                    logger.error(
                        f"Gemini API Error: {response.status_code} {response.text}"
                    )
                    return []
        except Exception as e:
            logger.error(f"Gemini VectorEngine Exception: {e}")
            return []


class OllamaVectorEngine(VectorEngine):
    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"
    ):
        self.base_url = base_url
        self.model = model

    async def get_embedding(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=30.0)
                if response.status_code == 200:
                    return response.json().get("embedding", [])
                else:
                    logger.error(
                        f"Ollama API Error: {response.status_code} {response.text}"
                    )
                    return []
        except Exception as e:
            logger.error(f"Ollama VectorEngine Exception: {e}")
            return []
