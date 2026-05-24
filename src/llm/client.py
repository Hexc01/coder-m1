import asyncio
import json
from typing import Any

import anthropic
from loguru import logger

from src.config import settings
from src.exceptions import LLMError


class ClaudeClient:
    """Wrapper around Anthropic SDK for mimo API calls."""

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model or settings.model_name
        self.max_tokens = max_tokens or settings.max_tokens
        self.client = anthropic.Anthropic(
            api_key=api_key or settings.anthropic_api_key,
            base_url=base_url or settings.anthropic_base_url,
        )

    async def chat(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Send a message to the LLM and return the response.

        Returns:
            {
                "content": str,
                "tool_calls": list[dict] | None,
                "usage": {"input_tokens": int, "output_tokens": int},
            }
        """
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=self.model,
                    max_tokens=max_tokens or self.max_tokens,
                    system=system,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                )
                return self._parse_response(response)
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/3)")
                await asyncio.sleep(wait)
            except anthropic.APIError as e:
                logger.error(f"API error on attempt {attempt + 1}/3: {e}")
                if attempt == 2:
                    raise LLMError(f"API error after 3 attempts: {e}") from e
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise LLMError(f"Unexpected error: {e}") from e

        raise LLMError("Failed after 3 retries")

    async def chat_structured(
        self,
        messages: list[dict],
        system: str = "",
        response_schema: dict | None = None,
        temperature: float = 0.0,
    ) -> Any:
        """Call LLM and parse JSON output.

        If response_schema is provided, append schema instructions to system prompt.
        Returns parsed JSON (dict or list).
        """
        full_system = system
        if response_schema:
            schema_hint = (
                "\n\nYou MUST respond with valid JSON only. "
                "No markdown fences, no explanation. "
                f"Schema: {json.dumps(response_schema)}"
            )
            full_system = system + schema_hint

        result = await self.chat(
            messages=messages,
            system=full_system,
            temperature=temperature,
        )

        content = result["content"].strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (```json and ```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed, retrying with correction prompt: {e}")
            correction_messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": "Your response was not valid JSON. Please respond with ONLY valid JSON, no markdown."},
            ]
            result2 = await self.chat(
                messages=correction_messages,
                system=full_system,
                temperature=0.0,
            )
            try:
                return json.loads(result2["content"].strip())
            except json.JSONDecodeError:
                raise LLMError(
                    f"Failed to parse JSON response after correction. "
                    f"Got: {result2['content'][:200]}"
                )

    def _parse_response(self, response: anthropic.types.Message) -> dict[str, Any]:
        """Parse an Anthropic Message into our standard format."""
        content_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return {
            "content": "\n".join(content_parts),
            "tool_calls": tool_calls if tool_calls else None,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }
