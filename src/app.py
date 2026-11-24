from src.clients.decision_client import DecisionManager
from src.clients.encoder_client import EncoderManager
from src.clients.llm_client import LLMManager
from src.clients.qdrant_client import QdrantManager


async def init_app():
    await EncoderManager.init_client()
    await LLMManager.init_client()
    await QdrantManager.init_client()
    await DecisionManager.init_client()
