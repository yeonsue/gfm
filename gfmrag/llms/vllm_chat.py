import logging
import os
import time

import dotenv
import tiktoken
from openai import OpenAI

from .base_language_model import BaseLanguageModel

logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

dotenv.load_dotenv()

os.environ["TIKTOKEN_CACHE_DIR"] = "./tmp"


def get_token_limit(model: str = "", default_limit: int = 128000) -> int:
    """Return a token limit for the configured vLLM-served model."""
    known_limits = {
        "gpt-4": 8192,
        "gpt-4-0613": 8192,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-3.5-turbo-16k-0613": 16384,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-0613": 4096,
    }
    return known_limits.get(model, default_limit)


class VLLMChat(BaseLanguageModel):
    """Chat wrapper for models served behind a vLLM OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_name_or_path: str,
        api_base: str,
        retry: int = 5,
        api_key: str | None = None,
        request_timeout: int = 60,
        token_limit: int | None = None,
        tiktoken_model_name: str | None = None,
    ):
        self.retry = retry
        self.model_name = model_name_or_path
        self.request_timeout = request_timeout
        self.tiktoken_model_name = tiktoken_model_name
        self.maximun_token = token_limit or get_token_limit(self.model_name)

        resolved_api_key = (
            api_key
            or os.environ.get("VLLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY"
        )
        self.client = OpenAI(api_key=resolved_api_key, base_url=api_base)

    def token_len(self, text: str) -> int:
        try:
            model_for_encoding = self.tiktoken_model_name or self.model_name
            encoding = tiktoken.encoding_for_model(model_for_encoding)
        except KeyError:
            logger.warning(
                "Tokenizer for model '%s' not found. Falling back to cl100k_base.",
                self.tiktoken_model_name or self.model_name,
            )
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def generate_sentence(
        self, llm_input: str | list, system_input: str = ""
    ) -> str | Exception:
        if isinstance(llm_input, list):
            message = llm_input
        else:
            message = []
            if system_input:
                message.append({"role": "system", "content": system_input})
            message.append({"role": "user", "content": llm_input})

        cur_retry = 0
        num_retry = self.retry
        message_string = "\n".join([m["content"] for m in message])
        input_length = self.token_len(message_string)
        if input_length > self.maximun_token:
            logger.warning(
                "Input length %s exceeds token limit %s; truncating raw input.",
                input_length,
                self.maximun_token,
            )
            llm_input = llm_input[: self.maximun_token]

        error = Exception("Failed to generate sentence")
        while cur_retry <= num_retry:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    timeout=self.request_timeout,
                    temperature=0.0,
                )
                result = response.choices[0].message.content
                return result.strip() if result else ""
            except Exception as e:
                logger.error("Message: %s", llm_input)
                logger.error("Number of token: %s", self.token_len(message_string))
                logger.error(e)
                time.sleep(30)
                cur_retry += 1
                error = e
        return error
