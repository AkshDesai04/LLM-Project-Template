import time
import json
import base64
from typing import Optional, List, Any, Union

import ollama
from ollama import Client

from utils.logger import get_logger
from utils.env_ops import get_local_secret
from ..base_provider import LLMProvider, JudgeResult
from ..cost_tracker import cost_tracker
from ..utils.media_utils import extract_text_from_pdf_bytes, process_video_frames
from core.modules.base import Base as BaseModule

logger = get_logger("OllamaProvider")

class OllamaProvider(LLMProvider):
    def __init__(self, api_key: str, base: BaseModule):
        """
        api_key is not strictly required for Ollama but kept for interface consistency.
        OLLAMA_URL should be set in .env.
        """
        super().__init__(api_key, base)
        ollama_url = get_local_secret("OLLAMA_URL", raise_error=False) or "http://localhost:11434"
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
        max_retries = kwargs.get('max_retries', 3)
        
        format_param = None
        if structure:
            format_param = 'json'

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
                        for chunk in response_stream:
                            if chunk.get('done'):
                                prompt_tokens = chunk.get('prompt_eval_count', 0)
                                completion_tokens = chunk.get('eval_count', 0)
                                total_duration = time.time() - start_time
                                costs = cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
                                cost_tracker.record_transaction(type(module).__name__, model, costs, total_duration)
                                logger.info(f"Ollama Stream Transaction Recorded: ${costs['total_cost']:.6f} total cost")
                            yield chunk
                    return stream_wrapper()

                response = self.client.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    format=format_param
                )
                
                total_duration = time.time() - start_time
                output_content = response['message']['content']
                
                prompt_tokens = response.get('prompt_eval_count', 0)
                completion_tokens = response.get('eval_count', 0)
                
                costs = cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
                cost_tracker.record_transaction(type(module).__name__, model, costs, total_duration)
                logger.info(f"Ollama Transaction Recorded: ${costs['total_cost']:.6f} total cost")
                
                if structure:
                    try:
                        parsed = json.loads(output_content)
                        if hasattr(structure, 'model_validate'):
                             return structure.model_validate(parsed)
                        return parsed
                    except Exception as e:
                        logger.warning(f"Failed to parse Ollama JSON response: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                
                return output_content

            except Exception as e:
                last_exception = e
                logger.warning(f"Ollama response failed on attempt {attempt + 1} for model {model}: {e}")
                time.sleep(2)
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
                logger.info(f"Ollama does not natively support video. Extracting frames...")
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
            cost_tracker.record_transaction("Embedding", model, costs, total_duration)
            
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
