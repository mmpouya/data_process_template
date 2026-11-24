import logging
import sys

from openai import APIConnectionError, AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from src.core.config import get_settings
from src.schema.llm import UserPromptSchema

logger = logging.getLogger(__name__)


class LLMManager:
    _client: AsyncOpenAI | None = None
    # _chunker_client: AsyncOpenAI | None = None

    @classmethod
    async def init_client(cls) -> AsyncOpenAI:
        if not cls._client:  # or not cls._chunker_client:
            cls._client = AsyncOpenAI(
                api_key=get_settings().OPENAI_API_KEY,
                base_url=get_settings().OPENAI_BASE_URL,
                timeout=3.1,
            )
            try:
                await cls._client.models.list()
                print("llm client is ready!")
            except APIConnectionError:
                print("llm client is out of hand!")
                sys.exit()
        return cls._client

    @classmethod
    async def get_completion(
        cls,
        messages: list[UserPromptSchema],
        # user_message: str,
        # system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
    ) -> str:
        """Generate an async completion using the OpenAI API."""
        if not cls._client:
            cls._client = await cls.init_client()
        try:
            response: ChatCompletion = await cls._client.chat.completions.create(
                model=get_settings().OPENAI_MODEL_NAME,
                temperature=temperature,
                messages=messages,  # ty:ignore[invalid-argument-type]
                # messages=[
                #     {"role": "system", "content": system_message},
                #     {"role": "user", "content": user_message},
                # ],
            )
            if response.usage:
                tokens = response.usage.total_tokens
            else:
                tokens = 0
            logger.debug(f"token used: {tokens}")
            res = response.choices[0].message.content
            return res if res else ""
        except Exception as e:
            print("Error calling OpenAI API:", e)
            raise

    @classmethod
    async def get_parsed_result[OutputFormat: BaseModel](
        cls,
        system_prompt: str,
        user_message: str,
        output_format: type[OutputFormat],
        temperature: float = 0.7,
    ) -> OutputFormat | None:
        """Generate an async completion using the OpenAI API and parse the result.
        using structured output format."""
        if not cls._client:
            cls._client = await cls.init_client()
        max_retry = 10
        for _ in range(max_retry):
            try:
                response = await cls._client.beta.chat.completions.parse(
                    model=get_settings().OPENAI_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    response_format=output_format,
                    temperature=temperature,
                )
                if response.usage:
                    tokens = response.usage.total_tokens
                else:
                    tokens = 0
                logger.debug(f"token used: {tokens}")
                result = response.choices[0].message.parsed
                print(f"Parsed result: {result}")
                if result:
                    # return (result, tokens)
                    return result
                else:
                    # return (None, 0)
                    return None
            except Exception as e:
                print("Error calling OpenAI API:", e)

                raise
