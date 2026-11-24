import sys

from openai import APITimeoutError, AsyncOpenAI

from src.core.config import get_settings

ENCODER_MAX_TOKEN = 8192


class EncoderManager:
    _client: AsyncOpenAI | None = None

    @classmethod
    async def init_client(cls) -> AsyncOpenAI:
        if not cls._client:
            cls._client = AsyncOpenAI(
                api_key=get_settings().OPENAI_API_KEY,
                base_url=get_settings().ENCODER_BASE_URL,
                timeout=2.1,
            )
            try:
                _ = await cls._client.models.list()
                print("llm client is ready!")
            except APITimeoutError:
                print("llm client is out of hand!")
                sys.exit()
        return cls._client

    @classmethod
    async def embed(cls, text: str | list[str]) -> list[list[float]]:
        if not cls._client:
            cls._client = await cls.init_client()
        if isinstance(text, str):
            text = [text]
        model_name = get_settings().ENCODER_MODEL_NAME
        response = await cls._client.embeddings.create(
            model=model_name,
            input=text,
            extra_body={"truncate_prompt_tokens": ENCODER_MAX_TOKEN},
        )
        embeddings = response.data
        return [res.embedding for res in embeddings]
