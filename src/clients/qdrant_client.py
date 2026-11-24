import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from src.clients.encoder_client import EncoderManager
from src.core.config import get_settings

logger = logging.getLogger(__name__)


class QdrantManager:
    QDRANT_URL: str | None = get_settings().QDRANT_URL
    MAX_RETRIES: int = 5
    VECTOR_TYPE: str = "dense"

    _client: AsyncQdrantClient | None = None

    @classmethod
    async def init_client(cls) -> AsyncQdrantClient:
        if cls._client is None:
            if cls.QDRANT_URL:
                cls._client = AsyncQdrantClient(
                    url=cls.QDRANT_URL, timeout=15, prefer_grpc=True
                )
            else:
                cls._client = AsyncQdrantClient(
                    path="./.qdrant_storage", timeout=15, prefer_grpc=False
                )
        return cls._client

    @classmethod
    async def close_client(cls) -> None:
        if cls._client:
            await cls._client.close()
            cls._client = None

    @classmethod
    async def create_collection(cls, collection_name: str) -> bool:
        client = await cls.init_client()
        exists = await cls.is_collection_exists(collection_name)
        if not exists:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    cls.VECTOR_TYPE: VectorParams(
                        size=get_settings().ENCODER_MODEL_DIM,
                        distance=Distance.COSINE,
                    )
                },
            )
        return True

    @classmethod
    async def is_collection_exists(cls, collection_name: str) -> bool:
        client = await cls.init_client()
        return await client.collection_exists(collection_name)

    @classmethod
    async def query(
        cls, query_text: str, collection_name: str, limit: int = 10
    ) -> list[ScoredPoint]:
        """Semantic search: embed the query and search the named vector."""
        client = await cls.init_client()
        vector = await EncoderManager.embed(query_text)
        response = await client.query_points(
            collection_name=collection_name,
            query=vector,
            using=cls.VECTOR_TYPE,
            limit=limit,
            with_payload=True,
        )
        return list(response.points)

    @classmethod
    async def get_point_by_ids(
        cls, collection_name: str, point_ids: list[int | str]
    ) -> list[str]:
        """Return the `content` payload of each existing point, in order."""
        client = await cls.init_client()
        records = await client.retrieve(
            collection_name=collection_name,
            ids=point_ids,
            with_payload=True,
        )
        return [str((record.payload or {}).get("content", "")) for record in records]

    @classmethod
    async def get_collections(cls) -> list[dict[str, Any]]:
        client = await cls.init_client()
        collections_response = await client.get_collections()
        result: list[dict[str, Any]] = []
        for col in collections_response.collections:
            collection_info = await client.get_collection(collection_name=col.name)
            info_dict = collection_info.model_dump()
            result.append(
                {
                    "name": col.name,
                    "status": info_dict.get("status"),
                    "metadata": info_dict.get("config", {}).get("metadata"),
                }
            )
        return result

    @classmethod
    async def create_point(
        cls,
        payload: dict,
        vector: list[float],
        collection_name: str,
    ) -> None:
        if cls._client:
            await cls._client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=uuid.uuid4().hex,
                        vector={
                            "dense": vector,
                        },
                        payload=payload,
                    )
                ],
                wait=False,
            )
