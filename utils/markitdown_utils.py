import os
from typing import Optional, Any, Union
from markitdown import MarkItDown
from utils.logger import get_logger

logger = get_logger("MarkItDownUtils")

class MarkItDownUtils:
    """
    A comprehensive utility class for converting various file formats and data sources
    into Markdown using Microsoft's MarkItDown.
    
    Supported formats include:
    - Documents: PDF, DOCX, PPTX, XLSX, XLS, EPub
    - Data: CSV, JSON, XML
    - Media: Images (EXIF/OCR), Audio (Transcription)
    - Web: HTML, YouTube (Transcripts)
    - Archives: ZIP
    """

    def __init__(
        self, 
        llm_client: Optional[Any] = None, 
        llm_model: Optional[str] = None,
        docintel_endpoint: Optional[str] = None,
        enable_plugins: bool = True
    ):
        """
        Initializes the MarkItDown converter.
        
        Args:
            llm_client: Optional LLM client (e.g., OpenAI or Gemini) for image description or OCR.
            llm_model: Optional model name to use for LLM-based processing.
            docintel_endpoint: Optional Azure Document Intelligence endpoint for high-fidelity conversion.
            enable_plugins: Whether to enable plugins (like markitdown-ocr for scanned documents).
        """
        logger.info(f"Initializing MarkItDown utility (plugins={enable_plugins})...")
        self.md = MarkItDown(
            llm_client=llm_client, 
            llm_model=llm_model,
            docintel_endpoint=docintel_endpoint,
            enable_plugins=enable_plugins
        )

    def convert(self, source: str) -> str:
        """
        Generic conversion method for local files, URLs, or other supported strings.
        
        Args:
            source: Path to a local file, a URL, or a media source.
            
        Returns:
            The converted Markdown content as a string.
        """
        try:
            logger.info(f"Converting source: {source}")
            result = self.md.convert(source)
            return result.text_content
        except Exception as e:
            logger.error(f"Failed to convert source '{source}': {e}")
            raise

    def convert_local(self, file_path: str) -> str:
        """
        Converts a local file to Markdown. This is safer for untrusted environments.
        
        Args:
            file_path: The absolute or relative path to the local file.
            
        Returns:
            The converted Markdown content.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Local file not found: {file_path}")
            
            logger.info(f"Converting local file: {file_path}")
            result = self.md.convert_local(file_path)
            return result.text_content
        except Exception as e:
            logger.error(f"Error during local conversion of {file_path}: {e}")
            raise

    def convert_url(self, url: str) -> str:
        """
        Converts content from a URL (HTML or YouTube) to Markdown.
        
        Args:
            url: The URL to fetch and convert.
            
        Returns:
            The converted Markdown content.
        """
        try:
            logger.info(f"Converting URL: {url}")
            result = self.md.convert_url(url)
            return result.text_content
        except Exception as e:
            logger.error(f"Error during URL conversion of {url}: {e}")
            raise

    def convert_stream(self, stream: Any, file_extension: str) -> str:
        """
        Converts a binary stream to Markdown.
        
        Args:
            stream: A file-like object containing binary data.
            file_extension: The extension (e.g., '.pdf') to help determine the format.
            
        Returns:
            The converted Markdown content.
        """
        try:
            logger.info(f"Converting stream with extension: {file_extension}")
            result = self.md.convert_stream(stream, file_extension=file_extension)
            return result.text_content
        except Exception as e:
            logger.error(f"Error during stream conversion: {e}")
            raise

    def convert_image(self, image_path: str, describe: bool = True) -> str:
        """
        Specialized method for converting images, optionally using an LLM to describe them.
        
        Args:
            image_path: Path to the image file.
            describe: Whether to attempt generating a text description (requires LLM config).
            
        Returns:
            Markdown containing image metadata and/or description.
        """
        try:
            logger.info(f"Converting image: {image_path} (describe={describe})")
            # Note: The generic convert() handles images, but we highlight it here.
            result = self.md.convert(image_path)
            return result.text_content
        except Exception as e:
            logger.error(f"Error converting image {image_path}: {e}")
            raise

    def convert_audio(self, audio_path: str) -> str:
        """
        Specialized method for converting audio files to Markdown via transcription.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            Markdown containing audio metadata and transcript.
        """
        try:
            logger.info(f"Converting audio: {audio_path}")
            result = self.md.convert(audio_path)
            return result.text_content
        except Exception as e:
            logger.error(f"Error converting audio {audio_path}: {e}")
            raise

if __name__ == "__main__":
    utils = MarkItDownUtils()
    print(utils.convert("./tests/test_files/0.pdf"))
