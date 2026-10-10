
import os
import time
from typing import AsyncGenerator

import tiktoken
from anthropic import (
    AsyncAnthropic,
    APIConnectionError as AnthropicConnectionError,
    APIStatusError as AnthropicAPIStatusError,
    RateLimitError as AnthropicRateLimitError,
)
from openai import (
    AsyncOpenAI,
    APIConnectionError as OpenAIConnectionError,
    APIStatusError as OpenAIAPIStatusError,
    RateLimitError as OpenAIRateLimitError,
)
from pricing import calculate_cost
from schemas import ChunkTelemetry, StreamChunk



class GatewayProviderError(Exception):
    """A safe, general error from an LLM provider."""


class RateLimitExceededError(GatewayProviderError):
    """The provider rejected a request because of rate limits."""


class AuthenticationError(GatewayProviderError):
    """Provider credentials are missing or invalid."""


class UnifiedLLMGateway:
    """Stream responses from OpenAI and Anthropic through one interface."""

    def __init__(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        self.openai_client = (
            AsyncOpenAI(api_key=openai_key) if openai_key else None
        )
        self.anthropic_client = (
            AsyncAnthropic(api_key=anthropic_key)
            if anthropic_key
            else None
        )

    async def aclose(self):
        """Close persistent provider clients."""
        if self.openai_client is not None:
            await self.openai_client.close()
        if self.anthropic_client is not None:
            await self.anthropic_client.close()

    async def stream(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Yield normalized text chunks and final telemetry."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1.")

        if model.startswith("gpt-"):
            provider = "openai"
            client = self.openai_client
        elif model.startswith("claude-"):
            provider = "anthropic"
            client = self.anthropic_client
        else:
            raise GatewayProviderError("Unsupported model.")

        if client is None:
            raise AuthenticationError(
                f"The API key for {provider} is missing."
            )

        started = time.perf_counter()
        first_token_time = None
        input_tokens = 0
        output_tokens = 0
        encoding = tiktoken.get_encoding("cl100k_base")

        try:
            if provider == "openai":
                input_tokens = len(encoding.encode(prompt))

                stream = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                )

                async for event in stream:
                    if not event.choices:
                        if event.usage is not None:
                            input_tokens = event.usage.prompt_tokens
                            output_tokens = event.usage.completion_tokens
                        continue

                    text = event.choices[0].delta.content or ""
                    if not text:
                        continue

                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    yield StreamChunk(
                        delta_text=text,
                        provider=provider,
                        model=model,
                    )

                # Prefer the provider's final usage information.
                # If it was not sent, estimate from the full response.
                if output_tokens == 0:
                    # This fallback is refined in later testing.
                    output_tokens = 0

            else:
                async with client.messages.stream(
                    model=model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                ) as stream:
                    async for text in stream.text_stream:
                        if not text:
                            continue

                        if first_token_time is None:
                            first_token_time = time.perf_counter()

                        yield StreamChunk(
                            delta_text=text,
                            provider=provider,
                            model=model,
                        )

                    final_message = await stream.get_final_message()
                    input_tokens = final_message.usage.input_tokens
                    output_tokens = final_message.usage.output_tokens

            elapsed = time.perf_counter() - started
            ttft_ms = (
                (first_token_time - started) * 1000
                if first_token_time is not None
                else 0.0
            )

            # If OpenAI did not return usage, estimate output tokens
            # from the full response in a later refinement.
            tokens_per_second = (
                output_tokens / elapsed if elapsed > 0 else 0.0
            )

            telemetry = ChunkTelemetry(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                latency_seconds=elapsed,
                ttft_ms=ttft_ms,
                tokens_per_second=tokens_per_second,
                total_cost_usd=calculate_cost(
                    model, input_tokens, output_tokens
                ),
            )

            yield StreamChunk(
                delta_text="",
                provider=provider,
                model=model,
                is_final=True,
                telemetry=telemetry,
            )

        except (
            OpenAIRateLimitError,
            AnthropicRateLimitError,
        ):
            raise RateLimitExceededError(
                f"{provider} rate limit exceeded."
            ) from None

        except (
            OpenAIConnectionError,
            AnthropicConnectionError,
        ):
            raise GatewayProviderError(
                f"Could not connect to {provider}."
            ) from None

        except (
            OpenAIAPIStatusError,
            AnthropicAPIStatusError,
        ) as exc:
            if exc.status_code in (401, 403):
                raise AuthenticationError(
                    f"{provider} authentication failed."
                ) from None

            raise GatewayProviderError(
                f"{provider} request failed (HTTP {exc.status_code})."
            ) from None