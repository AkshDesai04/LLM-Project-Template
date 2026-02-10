import inspect
import time
from typing import Optional, List, Any, Union

from openai import OpenAI
from pydantic import BaseModel

from utils.logger import get_logger
from .llm_model_base import LLMModels
from ..modules.base import Base as BaseModule

logger = get_logger("OpenAI")


class OpenAIModel(LLMModels):
    def __init__(self, api_key: str, base: BaseModule):
        super().__init__(api_key, base)
        self.client = OpenAI(api_key=api_key)

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None) -> Any:
        try:
            prompt = module.prompt
            structure = module.structure
            model = module.model or self.model_name
            temperature = getattr(module, 'temperature', self.temperature)
            top_p = getattr(module, 'top_p', self.top_p)
            reasoning_effort = getattr(module, 'reasoning_budget', None)

            messages = [{"role": "user", "content": prompt}]

            if uploaded_file:
                if isinstance(uploaded_file, str):
                    messages[0]["content"] += f"\n\n[Attached Content]:\n{uploaded_file}"

            logger.info(f"Starting generation with model: {model}")
            start_time = time.time()

            # For models that support structured output natively (v1.40.0+)
            # We use the beta.chat.completions.parse to get automatic Pydantic validation
            parsed_object = None

            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
            }

            # Reasoning models (o1, o3, gpt-5, etc.) often don't support temperature or top_p.
            # They might also require 'reasoning_effort' instead.
            is_reasoning_model = any(
                x in model.lower() for x in ["o1-", "o3-", "gpt-5"]) or reasoning_effort is not None

            if is_reasoning_model:
                kwargs.pop("temperature", None)
                kwargs.pop("top_p", None)
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort.lower()

            if structure and inspect.isclass(structure) and issubclass(structure, BaseModel):
                # Use the 'parse' method for automatic conversion to Pydantic model
                response = self.client.beta.chat.completions.parse(
                    **kwargs,
                    response_format=structure
                )
                parsed_object = response.choices[0].message.parsed
                output_content = response.choices[0].message.content
            else:
                # Standard completion
                if structure:  # If it's just 'json' or similar but not a BaseModel class
                    kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**kwargs)
                output_content = response.choices[0].message.content

            usage = response.usage

            end_time = time.time()
            total_duration = end_time - start_time

            if not output_content and not parsed_object:
                raise ValueError("Received an empty response from OpenAI.")

            # --- Log Metadata ---
            log_msg = ["\n" + "=" * 30, "OPENAI METADATA REPORT", "=" * 30, f"Model: {model}",
                       f"Total Wall-Clock Time: {total_duration:.4f}s"]

            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)

                cached_tokens = 0
                prompt_details = getattr(usage, 'prompt_tokens_details', None)
                if prompt_details:
                    cached_tokens = getattr(prompt_details, 'cached_tokens', 0)

                costs = self._calculate_cost(model, prompt_tokens, completion_tokens, cached_tokens)
                self.record_transaction(type(module).__name__, model, costs, total_duration)

                log_msg.append(f"Cached Input Tokens: {cached_tokens}")
                log_msg.append(f"Cached Cost: ${costs['cached_cost']:.6f}")
                log_msg.append("")
                log_msg.append(f"Completion Token Count: {completion_tokens}")

                completion_details = getattr(usage, 'completion_tokens_details', None)
                if completion_details:
                    reasoning_tokens = getattr(completion_details, 'reasoning_tokens', 0)
                    log_msg.append(f"Reasoning/Thinking Tokens: {reasoning_tokens}")

                log_msg.append(f"Output Cost: ${costs['output_cost']:.6f}")
                log_msg.append("")
                log_msg.append(f"Prompt Tokens Count: {prompt_tokens}")
                log_msg.append(f"Input Cost: ${costs['input_cost']:.6f}")
                log_msg.append("")
                log_msg.append(f"Total Token Count: {getattr(usage, 'total_tokens', 0)}")
                log_msg.append(f"Total Estimated Cost: ${costs['total_cost']:.6f}")

            log_msg.append("=" * 30 + "\n")
            logger.info("\n".join(log_msg))

            # Structured logging for costs
            if usage:
                logger.info({
                    "transaction_type": "generation",
                    "model": model,
                    "module": type(module).__name__,
                    "costs": costs,
                    "duration": total_duration
                })
            # ----------------------

            self.print_final_summary()

            if parsed_object:
                return parsed_object

            return output_content

        except Exception as e:
            logger.error(f"OpenAI response failed: {e}")
            raise

    def upload_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extracts text from PDF bytes to be used as context in the OpenAI prompt.
        """
        try:
            logger.info("Processing PDF for OpenAI (Local Extraction)...")
            text = self._extract_text_from_pdf_bytes(pdf_bytes)
            return text
        except Exception as e:
            logger.error(f"OpenAI PDF processing failed: {e}")
            raise RuntimeError(f"Failed to process PDF for OpenAI: {e}")

    def embed_content(self, text: Union[str, List[str]], model="text-embedding-3-small", **kwargs) -> Union[
        List[float], List[List[float]]]:
        try:
            logger.info(f"Starting embedding with model: {model}")
            start_time = time.time()

            input_data = [text] if isinstance(text, str) else text

            response = self.client.embeddings.create(
                model=model,
                input=input_data,
                **kwargs
            )

            end_time = time.time()
            total_duration = end_time - start_time

            # --- Log Metadata ---
            log_msg = ["\n" + "=" * 30, "OPENAI EMBEDDING METADATA REPORT", "=" * 30, f"Model: {model}",
                       f"Total Wall-Clock Time: {total_duration:.4f}s"]

            usage = getattr(response, 'usage', None)
            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                costs = self._calculate_cost(model, prompt_tokens, 0, 0)
                self.record_transaction("Embedding", model, costs, total_duration)
                log_msg.append(f"Prompt Tokens Count: {prompt_tokens}")
                log_msg.append(f"Input Cost: ${costs['input_cost']:.6f}")
                log_msg.append(f"Total Estimated Cost: ${costs['total_cost']:.6f}")

            log_msg.append("=" * 30 + "\n")
            logger.info("\n".join(log_msg))

            # Structured logging for costs
            if usage:
                logger.info({
                    "transaction_type": "embedding",
                    "model": model,
                    "module": "Embedding",
                    "costs": costs,
                    "duration": total_duration
                })
            # ----------------------

            if isinstance(text, str):
                return response.data[0].embedding
            else:
                return [d.embedding for d in response.data]

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise


# Initializer
def initialize_openai(key, module) -> OpenAIModel:
    try:
        module_instance = LLMModels.prepare_module(module)
        openai_model = OpenAIModel(key, module_instance)
        return openai_model

    except Exception as e:
        logger.error(f"Error initializing OpenAI: {e}")
        raise
