import time
from io import BytesIO
from typing import Optional, List, Any

from google import genai
from google.genai import types
from google.genai.types import ThinkingLevel

from utils.logger import get_logger
from .llm_model_base import LLMModels, JudgeResult
from ..modules.base import Base as BaseModule

logger = get_logger("Gemini")


class GeminiModel(LLMModels):
    def __init__(self, api_key: str, base: BaseModule):
        super().__init__(api_key, base)
        self.client = genai.Client(api_key=api_key)

        self.gen_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            # ... rest of init ...
        )

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None) -> Any:
        try:
            prompt = module.prompt
            structure = module.structure
            model = module.model or self.model_name
            top_p = module.top_p
            top_k = module.top_k
            temperature = module.temperature
            reasoning_budget = module.reasoning_budget
            response_mime_type = module.response_mime_type

            # Prepare content list
            contents: List[Any] = [prompt]
            if uploaded_file:
                contents.append(uploaded_file)

            # Configure thinking/reasoning
            thinking_config_obj = None

            if reasoning_budget:
                if isinstance(reasoning_budget, str):
                    reasoning = ThinkingLevel(reasoning_budget)

                    # If reasoning_budget is a string (e.g., "HIGH"), pass it to thinking_level
                    thinking_config_obj = types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_level=reasoning
                    )
                else:
                    # Fallback if just True/int provided without level specification
                    thinking_config_obj = types.ThinkingConfig(include_thoughts=True)

            # Build GenerateContentConfig
            config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                response_mime_type=response_mime_type,
                response_schema=structure,
                thinking_config=thinking_config_obj,
            )

            logger.info(f"Starting generation with model: {model}")
            start_time = time.time()

            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            end_time = time.time()
            total_duration = end_time - start_time

            if not response.text:
                raise ValueError("Received an empty response from Gemini.")

            # --- Log Metadata ---
            log_msg = []
            log_msg.append("\n" + "=" * 30)
            log_msg.append("GEMINI METADATA REPORT")
            log_msg.append("=" * 30)
            log_msg.append(f"Model: {model}")
            log_msg.append(f"Total Wall-Clock Time: {total_duration:.4f}s")

            if response.usage_metadata:
                u = response.usage_metadata

                # Helper to safely get value or 0
                def get_val(obj, attr):
                    return getattr(obj, attr, 0) or 0

                prompt_tokens = get_val(u, 'prompt_token_count')
                candidate_tokens = get_val(u, 'candidates_token_count')
                cached_tokens = get_val(u, 'cached_content_token_count')

                costs = self._calculate_cost(model, prompt_tokens, candidate_tokens, cached_tokens)
                self.record_transaction(type(module).__name__, model, costs, total_duration)

                log_msg.append(f"Cache Tokens Details: {getattr(u, 'cache_tokens_details', 'N/A')}")
                log_msg.append(f"Cached Content Token Count: {cached_tokens}")
                log_msg.append(f"Cached Cost: ${costs['cached_cost']:.6f}")
                log_msg.append("")
                log_msg.append(f"Candidates Token Count: {candidate_tokens}")
                log_msg.append(f"Candidates Tokens Details: {getattr(u, 'candidates_tokens_details', 'N/A')}")
                log_msg.append(f"Output Cost: ${costs['output_cost']:.6f}")
                log_msg.append("")
                log_msg.append(f"Prompt Tokens Count: {prompt_tokens}")

                # Handle Prompt Token Details (Image vs Text breakdown)
                details = getattr(u, 'prompt_tokens_details', None)
                if details:
                    for detail in details:
                        modality = "UNKNOWN"
                        if hasattr(detail, 'modality'):
                            modality = getattr(detail.modality, 'name', str(detail.modality))
                        log_msg.append(f"Prompt Tokens Detail: {modality}: {detail.token_count}")

                log_msg.append(f"Input Cost: ${costs['input_cost']:.6f}")
                log_msg.append("")
                log_msg.append(f"Thoughts Token Count: {get_val(u, 'thoughts_token_count')}")
                log_msg.append("")
                log_msg.append(f"Tool Use Prompt Tokens Count: {get_val(u, 'tool_use_prompt_token_count')}")
                log_msg.append("")
                log_msg.append(f"Total Token Count: {get_val(u, 'total_token_count')}")
                log_msg.append(f"Total Estimated Cost: ${costs['total_cost']:.6f}")
                log_msg.append(f"Traffic Type: {getattr(u, 'traffic_type', 'N/A')}")

            log_msg.append("=" * 30 + "\n")
            logger.info("\n".join(log_msg))

            # Structured logging for costs
            if response.usage_metadata:
                logger.info({
                    "transaction_type": "generation",
                    "model": model,
                    "module": type(module).__name__,
                    "costs": costs,
                    "duration": total_duration
                })
            # ----------------------

            self.print_final_summary()

            if structure:
                return response.parsed

            return response.text

        except Exception as e:
            logger.error(f"Gemini response failed: {e}")
            raise

    def upload_media(self, file_bytes: bytes, mime_type: str) -> types.File:
        """
        Uploads media (PDF, Image, Audio, etc.) to Gemini.
        Args:
            file_bytes: The raw bytes of the file.
            mime_type: The MIME type of the file (e.g., 'application/pdf', 'image/png').
        """
        try:
            logger.info(f"Uploading {mime_type} to Gemini...")
            file_obj = BytesIO(file_bytes)
            file_obj.seek(0)

            uploaded_file = self.client.files.upload(
                file=file_obj,
                config=types.UploadFileConfig(mime_type=mime_type)
            )

            logger.info(f"File uploaded: {uploaded_file.name}")
            return uploaded_file

        except Exception as e:
            logger.error(f"Gemini upload failed: {e}")
            raise RuntimeError(f"Failed to upload {mime_type} to Gemini: {e}")

    def embed_content(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT", model="gemini-embedding-001",
                      dimensions=1536) -> List[float]:
        try:
            logger.info(f"Starting embedding with model: {model}")
            start_time = time.time()

            result = self.client.models.embed_content(
                model=model,
                contents=[text],
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=dimensions)
            )

            end_time = time.time()
            total_duration = end_time - start_time

            if not result.embeddings or not hasattr(result.embeddings[0], 'values'):
                raise ValueError("Invalid embedding response from Gemini.")

            # --- Log Metadata ---
            log_msg = []
            log_msg.append("\n" + "=" * 30)
            log_msg.append("GEMINI EMBEDDING METADATA REPORT")
            log_msg.append("=" * 30)
            log_msg.append(f"Model: {model}")
            log_msg.append(f"Total Wall-Clock Time: {total_duration:.4f}s")

            # Embedding API often doesn't return usage_metadata directly.
            # We check for it, or fall back to count_tokens for pricing estimation.
            usage = getattr(result, 'usage_metadata', None)
            if usage:
                prompt_tokens = getattr(usage, 'prompt_token_count', 0)
            else:
                try:
                    token_count_resp = self.client.models.count_tokens(model=model, contents=[text])
                    prompt_tokens = token_count_resp.total_tokens
                except Exception:
                    prompt_tokens = 0

            if prompt_tokens > 0:
                costs = self._calculate_cost(model, prompt_tokens, 0, 0)
                self.record_transaction("Embedding", model, costs, total_duration)
                log_msg.append(f"Prompt Tokens Count: {prompt_tokens}")
                log_msg.append(f"Input Cost: ${costs['input_cost']:.6f}")
                log_msg.append(f"Total Estimated Cost: ${costs['total_cost']:.6f}")
            else:
                log_msg.append("Usage Metadata: N/A")

            log_msg.append("=" * 30 + "\n")
            logger.info("\n".join(log_msg))

            # Structured logging for costs
            if prompt_tokens > 0:
                logger.info({
                    "transaction_type": "embedding",
                    "model": model,
                    "module": "Embedding",
                    "costs": costs,
                    "duration": total_duration
                })
            # ----------------------

            return result.embeddings[0].values

        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise

    def evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult:
        """Evaluates a response using Gemini as a Judge."""
        logger.info("Evaluating response with Gemini Judge...")

        judge_prompt = f"""
        You are an impartial judge evaluating the quality of an AI-generated response.

        [Original Prompt]:
        {input_prompt}

        [AI Generated Response]:
        {generated_output}

        [Evaluation Rubric]:
        {rubric if rubric else "Evaluate based on accuracy, clarity, and adherence to the prompt."}

        Please provide a score from 1-10, your reasoning, and any suggestions for improvement.
        """

        # Create a temporary module instance for the judge call
        class JudgeModule(BaseModule):
            prompt: str = judge_prompt
            structure: Any = JudgeResult
            model: str = "gemini-2.0-flash"  # Use a fast, capable model for judging
            response_mime_type: str = "application/json"

        return self.model_response(JudgeModule())


# Initializer
def initialize_gemini(key, module) -> GeminiModel:
    try:
        module_instance = LLMModels.prepare_module(module)
        gemini = GeminiModel(key, module_instance)
        return gemini

    except Exception as e:
        logger.error(f"Error initializing Gemini: {e}")
        raise