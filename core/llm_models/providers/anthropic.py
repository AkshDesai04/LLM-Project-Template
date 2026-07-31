import inspect
import time
from typing import Optional, List, Any, Union

from pydantic import BaseModel
from anthropic import Anthropic

from utils.logger import get_logger
from utils.env_ops import get_local_secret
from ..base_provider import LLMProvider, JudgeResult
from ..cost_tracker import cost_tracker
from ..utils.media_utils import (
    extract_text_from_pdf_bytes,
    process_video_frames,
    encode_image_base64
)
from core.modules.base import Base as BaseModule

logger = get_logger("AnthropicProvider")

# The Messages API rejects requests without an explicit output cap.
DEFAULT_MAX_TOKENS = 4096

# Base.reasoning_budget allows "minimal", which Anthropic's effort scale lacks.
EFFORT_LEVELS = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}

# Extended thinking needs headroom for the response on top of the budget.
THINKING_OUTPUT_HEADROOM = 1024

STRUCTURED_TOOL_NAME = "emit_structured_response"


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: Optional[str], base: BaseModule):
        api_key = api_key or get_local_secret("ANTHROPIC_KEY")
        super().__init__(api_key, base)
        self.client = Anthropic(api_key=api_key)

    @staticmethod
    def _build_structured_tool(structure: Any) -> Optional[dict]:
        """
        Anthropic has no JSON response format, so a schema is enforced by forcing
        a single tool call and reading the arguments back.
        """
        if not (inspect.isclass(structure) and issubclass(structure, BaseModel)):
            return None

        return {
            "name": STRUCTURED_TOOL_NAME,
            "description": (
                structure.__doc__
                or "Return the response using this schema."
            ).strip(),
            "input_schema": structure.model_json_schema(),
        }

    @staticmethod
    def _to_content_blocks(prompt: str, files: List[Any]) -> List[dict]:
        """
        Converts the OpenAI-shaped payloads produced by media_utils into
        Anthropic content blocks.
        """
        image_blocks = []
        text_attachments = []

        for item in files:
            if isinstance(item, str):
                text_attachments.append(item)
                continue

            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue

            url = item.get("image_url", {}).get("url", "")
            if not url.startswith("data:"):
                continue

            header, _, data = url.partition(",")
            media_type = header.split(";")[0][len("data:"):]

            image_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            })

        full_prompt = prompt
        if text_attachments:
            full_prompt += "\n\n" + "\n\n".join(
                f"[Attached Content]:\n{t}" for t in text_attachments
            )

        return [{"type": "text", "text": full_prompt}] + image_blocks

    @staticmethod
    def _extract_usage(usage: Any) -> tuple:
        """Anthropic reports cached reads separately from input_tokens."""
        if not usage:
            return 0, 0, 0

        def get_val(attr):
            return getattr(usage, attr, 0) or 0

        return (
            get_val('input_tokens'),
            get_val('output_tokens'),
            get_val('cache_read_input_tokens'),
        )

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
        prompt = getattr(module, 'prompt', "")
        structure = kwargs.get('schema') or kwargs.get('structure') or getattr(module, 'structure', None)
        model = kwargs.get('model', self.model_name)

        temperature = kwargs.get('temperature', getattr(module, 'temperature', self.temperature))
        top_p = kwargs.get('top_p', getattr(module, 'top_p', self.top_p))
        top_k = kwargs.get('top_k', getattr(module, 'top_k', self.top_k))
        system_prompt = kwargs.get('system_prompt', getattr(module, 'system_prompt', self.system_prompt))
        max_tokens = kwargs.get('max_tokens', getattr(module, 'max_tokens', self.max_tokens)) or DEFAULT_MAX_TOKENS
        stream = kwargs.get('stream', getattr(module, 'stream', self.stream))
        tools = kwargs.get('tools') or kwargs.get('function') or getattr(module, 'tools', self.tools)

        reasoning_budget = (
            kwargs.get('reasoning_budget')
            or kwargs.get('reasoning_effort')
            or getattr(module, 'reasoning_budget', None)
        )

        stop_sequences = kwargs.get('stop_sequences') or kwargs.get('stop') or getattr(module, 'stop_sequences', self.stop_sequences)
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]

        files = []
        if uploaded_file:
            if isinstance(uploaded_file, list):
                if any(isinstance(i, list) for i in uploaded_file):
                    files = [item for sublist in uploaded_file for item in sublist]
                else:
                    files = uploaded_file
            else:
                files = [uploaded_file]

        content_blocks = self._to_content_blocks(prompt, files)

        call_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": content_blocks}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if top_k is not None:
            call_kwargs["top_k"] = top_k

        if system_prompt:
            call_kwargs["system"] = system_prompt

        if stop_sequences:
            call_kwargs["stop_sequences"] = stop_sequences

        structured_tool = self._build_structured_tool(structure)
        if structured_tool:
            call_kwargs["tools"] = [structured_tool]
            call_kwargs["tool_choice"] = {"type": "tool", "name": STRUCTURED_TOOL_NAME}
        elif tools:
            call_kwargs["tools"] = tools

        if reasoning_budget:
            if isinstance(reasoning_budget, str):
                call_kwargs["effort"] = EFFORT_LEVELS.get(
                    reasoning_budget.lower(), reasoning_budget.lower()
                )
            else:
                # Explicit thinking budgets require temperature 1 and no nucleus sampling.
                call_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": reasoning_budget,
                }
                call_kwargs["temperature"] = 1
                call_kwargs.pop("top_p", None)
                call_kwargs.pop("top_k", None)
                call_kwargs["max_tokens"] = max(
                    max_tokens, reasoning_budget + THINKING_OUTPUT_HEADROOM
                )

        last_exception = None
        max_retries = kwargs.get('max_retries', 3)

        logger.info(f"Attempting generation with model: {model}")
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model}")
                start_time = time.time()

                if stream:
                    response_stream = self.client.messages.create(**call_kwargs, stream=True)

                    def stream_wrapper():
                        input_tokens = 0
                        output_tokens = 0
                        cached_tokens = 0
                        try:
                            for event in response_stream:
                                event_type = getattr(event, 'type', '')

                                if event_type == 'message_start':
                                    usage = getattr(getattr(event, 'message', None), 'usage', None)
                                    input_tokens, _, cached_tokens = self._extract_usage(usage)

                                elif event_type == 'message_delta':
                                    _, delta_output, _ = self._extract_usage(
                                        getattr(event, 'usage', None)
                                    )
                                    output_tokens = delta_output or output_tokens

                                yield event
                        finally:
                            total_duration = time.time() - start_time
                            costs = cost_tracker.calculate_cost(
                                model, input_tokens, output_tokens, cached_tokens
                            )
                            cost_tracker.record_transaction(
                                type(module).__name__,
                                model,
                                costs,
                                total_duration,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                cached_tokens=cached_tokens,
                            )
                            logger.info(
                                f"Anthropic Stream Transaction Recorded: "
                                f"${costs['total_cost']:.6f} total cost"
                            )

                    return stream_wrapper()

                response = self.client.messages.create(**call_kwargs)
                total_duration = time.time() - start_time

                text_parts = []
                tool_payload = None

                for block in getattr(response, 'content', []) or []:
                    block_type = getattr(block, 'type', '')

                    if block_type == 'text':
                        text_parts.append(getattr(block, 'text', ''))
                    elif block_type == 'tool_use' and getattr(block, 'name', '') == STRUCTURED_TOOL_NAME:
                        tool_payload = getattr(block, 'input', None)

                output_content = "\n".join(part for part in text_parts if part).strip()

                if not output_content and tool_payload is None:
                    raise ValueError("Received an empty response from Anthropic.")

                input_tokens, output_tokens, cached_tokens = self._extract_usage(
                    getattr(response, 'usage', None)
                )

                if input_tokens or output_tokens:
                    costs = cost_tracker.calculate_cost(
                        model, input_tokens, output_tokens, cached_tokens
                    )
                    cost_tracker.record_transaction(
                        type(module).__name__,
                        model,
                        costs,
                        total_duration,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cached_tokens=cached_tokens,
                    )
                    logger.info(
                        f"Anthropic Transaction Recorded: "
                        f"${costs['total_cost']:.6f} total cost"
                    )

                if tool_payload is not None:
                    if hasattr(structure, 'model_validate'):
                        return structure.model_validate(tool_payload)
                    return tool_payload

                return output_content

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Anthropic response failed on attempt {attempt + 1} for model {model}: {e}"
                )
                time.sleep(2)
                continue

        raise RuntimeError(
            f"Failed to get response from Anthropic model {model} after {max_retries} attempts."
        ) from last_exception

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        try:
            if mime_type == 'application/pdf':
                logger.info("Processing PDF for Anthropic (Local Extraction)...")
                return extract_text_from_pdf_bytes(file_bytes)

            elif mime_type.startswith('image/'):
                logger.info(f"Processing {mime_type} for Anthropic Vision...")
                return encode_image_base64(file_bytes, mime_type)

            elif mime_type.startswith('video/'):
                logger.info(f"Processing {mime_type} for Anthropic by extracting frames...")
                return process_video_frames(file_bytes)

            else:
                logger.info(f"Treating {mime_type} as plain text...")
                return file_bytes.decode('utf-8', errors='ignore')

        except Exception as e:
            logger.error(f"Anthropic media processing failed: {e}")
            raise RuntimeError(f"Failed to process {mime_type} for Anthropic: {e}")

    def embed_content(self, input_content: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. "
            "Use the OpenAI or Gemini provider for embeddings."
        )

    def evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult:
        judge_prompt = f"""
        You are an impartial judge evaluating the quality of an AI-generated response.
        [Original Prompt]: {input_prompt}
        [AI Generated Response]: {generated_output}
        [Evaluation Rubric]: {rubric if rubric else "Evaluate based on accuracy, clarity, and adherence to the prompt."}
        Please provide a score from 1-10, your reasoning, and any suggestions for improvement.
        """

        class JudgeModule(BaseModule):
            prompt: str = judge_prompt
            structure: Any = JudgeResult
            model: str = "claude-haiku-4-5"

        return self.model_response(JudgeModule())
