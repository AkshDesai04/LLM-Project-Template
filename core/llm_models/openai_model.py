import inspect
import time
import base64
from io import BytesIO
from typing import Optional, List, Any, Union
import os
import tempfile
import cv2


import PyPDF2
from openai import OpenAI
from pydantic import BaseModel

from utils.logger import get_logger
from .llm_model_base import LLMModels, JudgeResult
from ..modules.base import Base as BaseModule

logger = get_logger("OpenAI")


class OpenAIModel(LLMModels):
    def __init__(self, api_key: str, base: BaseModule):
        super().__init__(api_key, base)
        self.client = OpenAI(api_key=api_key)

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None) -> Any:
        prompt = module.prompt
        structure = module.structure
        model = self.model_name  # Use the model set during initialization
        temperature = getattr(module, 'temperature', self.temperature)
        top_p = getattr(module, 'top_p', self.top_p)
        reasoning_effort = getattr(module, 'reasoning_budget', None)

        # Construct messages payload once
        messages = []
        files = []
        if uploaded_file:
            if isinstance(uploaded_file, list):
                # If uploaded_file is a list, check if its elements are also lists (e.g. from video frames)
                if any(isinstance(i, list) for i in uploaded_file):
                    # Flatten the list
                    files = [item for sublist in uploaded_file for item in sublist]
                else:
                    files = uploaded_file
            else:
                files = [uploaded_file]


        image_contents = [f for f in files if isinstance(f, dict) and f.get("type") == "image_url"]
        text_contents = [f for f in files if isinstance(f, str)]

        if image_contents:
            logger.info(f"Preparing multimodal message (Text + {len(image_contents)} Image(s))...")
            content_block = [{"type": "text", "text": prompt}]
            content_block.extend(image_contents)
            if text_contents:
                extra_text = "\n\n" + "\n\n".join([f"[Attached Content]:\n{t}" for t in text_contents])
                content_block[0]["text"] += extra_text
            messages.append({"role": "user", "content": content_block})
        else:
            full_prompt = prompt
            if text_contents:
                full_prompt += "\n\n" + "\n\n".join([f"[Attached Content]:\n{t}" for t in text_contents])
            messages.append({"role": "user", "content": full_prompt})

        last_exception = None
        max_retries = 3

        logger.info(f"Attempting generation with model: {model}")
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model}")
                start_time = time.time()

                kwargs = {"model": model, "messages": messages, "temperature": temperature, "top_p": top_p}

                is_reasoning_model = any(x in model.lower() for x in ["o1-", "o3-", "gpt-5"])
                if is_reasoning_model:
                    kwargs.pop("temperature", None)
                    kwargs.pop("top_p", None)
                    if reasoning_effort and isinstance(reasoning_effort, str):
                        kwargs["reasoning_effort"] = reasoning_effort.lower()

                parsed_object = None
                if structure and inspect.isclass(structure) and issubclass(structure, BaseModel):
                    response = self.client.beta.chat.completions.parse(**kwargs, response_format=structure)
                    parsed_object = response.choices[0].message.parsed
                    output_content = response.choices[0].message.content
                else:
                    if structure:
                        kwargs["response_format"] = {"type": "json_object"}
                    response = self.client.chat.completions.create(**kwargs)
                    output_content = response.choices[0].message.content

                end_time = time.time()
                total_duration = end_time - start_time

                if not output_content and not parsed_object:
                    raise ValueError("Received an empty response from OpenAI.")

                # --- Success: Log Metadata and return ---
                usage = response.usage
                log_msg = [
                    "\n" + "=" * 30, "OPENAI METADATA REPORT", "=" * 30,
                    f"Model: {model}", f"Total Wall-Clock Time: {total_duration:.4f}s"
                ]

                if usage:
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0)
                    cached_tokens = getattr(getattr(usage, 'prompt_tokens_details', None), 'cached_tokens', 0)
                    costs = self._calculate_cost(model, prompt_tokens, completion_tokens, cached_tokens)
                    self.record_transaction(type(module).__name__, model, costs, total_duration)

                    log_msg.append(f"Cached Input Tokens: {cached_tokens}")
                    log_msg.append(f"Cached Cost: ${costs['cached_cost']:.6f}")
                    log_msg.append("")
                    log_msg.append(f"Completion Token Count: {completion_tokens}")
                    if getattr(usage, 'completion_tokens_details', None):
                        log_msg.append(
                            f"Reasoning/Thinking Tokens: {getattr(usage.completion_tokens_details, 'reasoning_tokens', 0)}")
                    log_msg.append(f"Output Cost: ${costs['output_cost']:.6f}")
                    log_msg.append("")
                    log_msg.append(f"Prompt Tokens Count: {prompt_tokens}")
                    log_msg.append(f"Input Cost: ${costs['input_cost']:.6f}")
                    log_msg.append("")
                    log_msg.append(f"Total Token Count: {getattr(usage, 'total_tokens', 0)}")
                    log_msg.append(f"Total Estimated Cost: ${costs['total_cost']:.6f}")

                log_msg.append("=" * 30 + "\n")
                logger.info("\n".join(log_msg))

                if usage:
                    logger.info({
                        "transaction_type": "generation", "model": model, "module": type(module).__name__,
                        "costs": costs, "duration": total_duration
                    })

                self.print_final_summary()

                return parsed_object if parsed_object else output_content

            except Exception as e:
                last_exception = e
                logger.warning(f"OpenAI response failed on attempt {attempt + 1} for model {model}: {e}")
                time.sleep(2)
                continue

        logger.error(f"All {max_retries} retries failed for model {model}.")
        raise RuntimeError(
            f"Failed to get response from OpenAI model {model} after {max_retries} attempts.") from last_exception

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        """
        Uploads/Processes media.
        - PDF: Extracts text locally (str).
        - Images: Encodes to Base64 for Vision API (dict).
        - Video: Extracts frames and encodes them as Base64.
        """
        try:
            if mime_type == 'application/pdf':
                logger.info("Processing PDF for OpenAI (Local Extraction)...")
                return self._extract_text_from_pdf_bytes(file_bytes)

            elif mime_type.startswith('image/'):
                logger.info(f"Processing {mime_type} for OpenAI Vision...")
                b64_str = base64.b64encode(file_bytes).decode('utf-8')
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_str}"
                    }
                }
            elif mime_type.startswith('video/'):
                logger.info(f"Processing {mime_type} for OpenAI by extracting frames...")
                return self._process_video_for_openai(file_bytes)
            else:
                # Fallback for plain text files
                logger.info(f"Treating {mime_type} as plain text...")
                return file_bytes.decode('utf-8', errors='ignore')

        except Exception as e:
            logger.error(f"OpenAI media upload failed: {e}")
            raise RuntimeError(f"Failed to process {mime_type} for OpenAI: {e}")

    def _process_video_for_openai(self, video_bytes: bytes, frames_per_second: int = 1) -> List[dict]:
        """
        Extracts frames from video bytes, encodes them, and returns a list of dictionaries.
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(video_bytes)
        temp_file.close()

        base64_frames = []
        video = cv2.VideoCapture(temp_file.name)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        frame_interval = int(fps / frames_per_second)
        
        try:
            for frame_num in range(0, total_frames, frame_interval):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                success, frame = video.read()
                if not success:
                    continue
                _, buffer = cv2.imencode(".jpeg", frame)
                base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        finally:
            video.release()
            os.unlink(temp_file.name)

        logger.info(f"Extracted {len(base64_frames)} frames from video.")

        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_frame}"
                }
            }
            for b64_frame in base64_frames
        ]

    def _extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """Helper to extract text from PDF bytes using PyPDF2."""
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return f"[Error extracting text from PDF: {e}]"

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

    def evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult:
        """Evaluates a response using OpenAI as a Judge."""
        logger.info("Evaluating response with OpenAI Judge...")

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
            model: str = "gpt-4o-mini"  # Use a cost-effective, capable model for judging
            response_mime_type: str = "application/json"

        return self.model_response(JudgeModule())


# Initializer
def initialize_openai(key, module) -> OpenAIModel:
    try:
        module_instance = LLMModels.prepare_module(module)
        openai_model = OpenAIModel(key, module_instance)
        return openai_model

    except Exception as e:
        logger.error(f"Error initializing OpenAI: {e}")
        raise
