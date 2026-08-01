import time
import json
import base64
from typing import Optional, List, Any, Union

import ollama
from ollama import Client

from utils.logging import get_logger
from utils.env import get_secret
from ..base_provider import LLMProvider, JudgeResult
from ..cost_tracker import cost_tracker
from ..reasoning import (
    ThinkTagStreamSplitter,
    build_result,
    join_reasoning,
    resolve_return_reasoning,
    split_think_tags,
)
from ..utils.media_utils import extract_text_from_pdf_bytes, process_video_frames
from core.modules.base import Base as BaseModule

logger = get_logger("OllamaProvider")

# Top-level Constants
DEFAULT_OLLAMA_URL: str = "http://localhost:11434"
DEFAULT_OLLAMA_KEY: str = "local-key"
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_SLEEP_SECONDS: float = 2.0
DEFAULT_FORMAT_JSON: str = "json"


class OllamaProvider(LLMProvider):
    def __init__(self, api_key: Optional[str], base: BaseModule):
        """
        api_key is not strictly required for Ollama but kept for interface consistency.
        OLLAMA_URL and OLLAMA_KEY should be set in .env if needed.
        """
        api_key = api_key or get_secret("OLLAMA_KEY", raise_error=False) or DEFAULT_OLLAMA_KEY
        super().__init__(api_key, base)
        ollama_url = get_secret("OLLAMA_URL", raise_error=False) or DEFAULT_OLLAMA_URL
        logger.info(f"Initializing Ollama client with host: {ollama_url}")
        self.client = Client(host=ollama_url)

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, system_prompt: Optional[str] = None, **kwargs) -> Any:
        prompt = getattr(module, 'prompt', "")
        structure = kwargs.get('schema') or kwargs.get('structure') or getattr(module, 'structure', None)
        model = kwargs.get('model', self.model_name)
        
        temperature = kwargs.get('temperature', getattr(module, 'temperature', self.temperature))
        top_p = kwargs.get('top_p', getattr(module, 'top_p', self.top_p))
        top_k = kwargs.get('top_k', getattr(module, 'top_k', self.top_k))
        
        final_system_prompt = system_prompt or kwargs.get('system_prompt', getattr(module, 'system_prompt', self.system_prompt))
        max_tokens = kwargs.get('max_tokens', getattr(module, 'max_tokens', self.max_tokens))
        stop = kwargs.get('stop') or kwargs.get('stop_sequences') or getattr(module, 'stop_sequences', self.stop_sequences)
        seed = kwargs.get('seed', getattr(module, 'seed', self.seed))
        stream = kwargs.get('stream', getattr(module, 'stream', self.stream))
        return_reasoning = resolve_return_reasoning(module, kwargs, self.return_reasoning)
        
        # Ollama specific options
        options = {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'num_predict': max_tokens,
            'stop': stop,
            'seed': seed,
        }
        # Remove None values
        options = {k: v for k, v in options.items() if v is not None}

        messages = []
        if final_system_prompt:
            messages.append({'role': 'system', 'content': final_system_prompt})
        
        images = []
        if uploaded_file:
            files = []
            if isinstance(uploaded_file, list):
                files = uploaded_file
            else:
                files = [uploaded_file]
                
            for f in files:
                if isinstance(f, bytes):
                    images.append(f)
                elif isinstance(f, dict) and f.get('type') == 'image_url':
                    url = f.get('image_url', {}).get('url', '')
                    if url.startswith('data:image'):
                        try:
                            b64_data = url.split(',')[1]
                            images.append(base64.b64decode(b64_data))
                        except Exception as e:
                            logger.error(f"Failed to decode base64 image: {e}")
                elif isinstance(f, str):
                    prompt += f"\n\n[Attached Content]:\n{f}"

        user_msg = {'role': 'user', 'content': prompt}
        if images:
            user_msg['images'] = images
        
        messages.append(user_msg)

        last_exception = None
        max_retries = kwargs.get('max_retries', DEFAULT_MAX_RETRIES)
        pull_attempted = False
        
        format_param = None
        response_mime_type = kwargs.get('response_mime_type', getattr(module, 'response_mime_type', self.response_mime_type))
        if structure or response_mime_type == "application/json":
            format_param = DEFAULT_FORMAT_JSON

        if stream and structure:
            logger.warning("Streaming is not supported with structured output in Ollama. Disabling streaming.")
            stream = False

        logger.info(f"Attempting generation with model: {model} (Ollama)")
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model}")
                start_time = time.time()
                
                if stream:
                    response_stream = self.client.chat(
                        model=model,
                        messages=messages,
                        options=options,
                        format=format_param,
                        stream=True
                    )
                    
                    def stream_wrapper():
                        splitter = ThinkTagStreamSplitter()

                        for chunk in response_stream:
                            if chunk.get('done'):
                                prompt_tokens = chunk.get('prompt_eval_count', 0)
                                completion_tokens = chunk.get('eval_count', 0)
                                total_duration = time.time() - start_time
                                costs = cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
                                cost_tracker.record_transaction(
                                    type(module).__name__,
                                    model,
                                    costs,
                                    total_duration,
                                    input_tokens=prompt_tokens,
                                    output_tokens=completion_tokens,
                                    cached_tokens=0,
                                )
                                logger.info(f"Ollama Stream Transaction Recorded: ${costs['total_cost']:.6f} total cost")

                            if not return_reasoning:
                                yield chunk
                                continue

                            message = chunk.get('message') or {}
                            text, thought = splitter.feed(message.get('content'))
                            thought = join_reasoning([
                                message.get('thinking'),
                                thought,
                            ]) or ""

                            if text or thought:
                                yield [text, thought]

                        if return_reasoning:
                            text, thought = splitter.flush()
                            if text or thought:
                                yield [text, thought]
                    return stream_wrapper()

                response = self.client.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    format=format_param
                )
                
                total_duration = time.time() - start_time
                message = response['message']
                output_content = message['content']
                field_reasoning = message.get('thinking')
                
                prompt_tokens = response.get('prompt_eval_count', 0)
                completion_tokens = response.get('eval_count', 0)
                
                costs = cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
                cost_tracker.record_transaction(
                    type(module).__name__,
                    model,
                    costs,
                    total_duration,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    cached_tokens=0,
                )
                logger.info(f"Ollama Transaction Recorded: ${costs['total_cost']:.6f} total cost")
                
                # Strip inline thoughts before parsing, or the <think> block
                # would make otherwise valid JSON unparseable.
                output_content, inline_reasoning = split_think_tags(output_content)
                reasoning = join_reasoning([field_reasoning, inline_reasoning])

                if structure:
                    try:
                        parsed = json.loads(output_content)
                        if hasattr(structure, 'model_validate'):
                             parsed = structure.model_validate(parsed)
                        return build_result(parsed, reasoning, return_reasoning)
                    except Exception as e:
                        logger.warning(f"Failed to parse Ollama JSON response: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(DEFAULT_RETRY_SLEEP_SECONDS)
                            continue
                
                return build_result(output_content, reasoning, return_reasoning)

            except ollama.ResponseError as e:
                last_exception = e
                if e.status_code == 404 and 'not found' in str(e).lower() and not pull_attempted:
                    logger.info(f"Model '{model}' not found locally. Attempting to pull from Ollama Hub...")
                    pull_attempted = True
                    try:
                        logger.info(f"Pulling '{model}'. This may take a while...")
                        self.client.pull(model)
                        logger.info(f"Successfully pulled model '{model}'. Retrying generation...")
                        continue
                    except Exception as pull_error:
                        logger.error(f"Failed to pull model '{model}' from Ollama Hub: {pull_error}")
                        raise RuntimeError(
                            f"Model '{model}' is not installed locally and could not be found or pulled from Ollama Hub. "
                            f"Ensure the model name is correct. Error: {pull_error}"
                        ) from pull_error
                else:
                    logger.warning(f"Ollama response failed on attempt {attempt + 1} for model {model}: {e}")
                    time.sleep(DEFAULT_RETRY_SLEEP_SECONDS)
                    continue

            except Exception as e:
                last_exception = e
                logger.warning(f"Ollama response failed on attempt {attempt + 1} for model {model}: {e}")
                time.sleep(DEFAULT_RETRY_SLEEP_SECONDS)
                continue
        
        raise RuntimeError(f"Failed to get response from Ollama after {max_retries} attempts.") from last_exception

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        try:
            if mime_type == 'application/pdf':
                logger.info("Processing PDF for Ollama (Local Extraction)...")
                return extract_text_from_pdf_bytes(file_bytes)

            elif mime_type.startswith('image/'):
                logger.info(f"Processing {mime_type} for Ollama Vision...")
                return file_bytes

            elif mime_type.startswith('video/'):
                logger.info("Ollama does not natively support video. Extracting frames...")
                return process_video_frames(file_bytes)

            else:
                logger.info(f"Treating {mime_type} as plain text...")
                return file_bytes.decode('utf-8', errors='ignore')

        except Exception as e:
            logger.error(f"Ollama media processing failed: {e}")
            raise RuntimeError(f"Failed to process {mime_type} for Ollama: {e}")

    def embed_content(self, text: Union[str, List[str]], model: Optional[str] = None, **kwargs) -> Union[List[float], List[List[float]]]:
        try:
            model = model or self.model_name
            input_texts = [text] if isinstance(text, str) else text
            embeddings = []
            start_time = time.time()
            
            for t in input_texts:
                resp = self.client.embeddings(model=model, prompt=t)
                embeddings.append(resp['embedding'])
            
            total_duration = time.time() - start_time
            
            costs = cost_tracker.calculate_cost(model, 0, 0)
            cost_tracker.record_transaction(
                "Embedding",
                model,
                costs,
                total_duration,
                input_tokens=0,
                output_tokens=0,
                cached_tokens=0,
            )
            
            if isinstance(text, str):
                return embeddings[0]
            return embeddings
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            raise

    def evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult:
        judge_prompt = f"""
        You are an impartial judge evaluating the quality of an AI-generated response.
        [Original Prompt]: {input_prompt}
        [AI Generated Response]: {generated_output}
        [Evaluation Rubric]: {rubric if rubric else "Evaluate based on accuracy, clarity, and adherence to the prompt."}
        
        Please provide a score from 1-10, your reasoning, and any suggestions for improvement.
        Return your response in JSON format with fields: score (int), reasoning (str), improvements (str).
        """

        class JudgeModule(BaseModule):
            prompt: str = judge_prompt
            structure: Any = JudgeResult
            model: str = self.model_name

        return self.model_response(JudgeModule())
