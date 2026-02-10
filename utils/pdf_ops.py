from io import BytesIO
from typing import Optional, BinaryIO

from PyPDF2 import PdfReader, PdfWriter
from utils.logger import get_logger

logger = get_logger("PDFOps")


def split_pages(input_stream: BinaryIO, start_index: Optional[int] = None, end_index: Optional[int] = None) -> BytesIO:
    logger.info(f"Splitting PDF pages from index {start_index} to {end_index}.")
    pdf_reader = PdfReader(input_stream)
    pdf_writer = PdfWriter()
    total_pages = len(pdf_reader.pages)

    # Resolve default values and clamp to valid range
    start_idx = 0 if start_index is None else max(0, min(start_index, total_pages - 1))
    end_idx = total_pages - 1 if end_index is None else max(start_idx, min(end_index, total_pages - 1))

    for page_index in range(start_idx, end_idx + 1):
        pdf_writer.add_page(pdf_reader.pages[page_index])

    output_stream = BytesIO()
    pdf_writer.write(output_stream)
    output_stream.seek(0)

    logger.info(f"Successfully split {end_idx - start_idx + 1} pages.")
    return output_stream

def extract_pages(file_obj: BinaryIO, n: int = None, m: int = None) -> BytesIO:
    """
    Extract pages from n to m (inclusive) from a PDF file-like object
    and return a new BytesIO PDF.
    ...
    """
    logger.info(f"Extracting PDF pages: {n} to {m}")
    reader = PdfReader(file_obj)
    writer = PdfWriter()

    total = len(reader.pages)

    # default start
    if n is None:
        n = 0

    # clamp n
    n = max(0, min(n, total - 1))

    # default end
    if m is None:
        m = total - 1

    # clamp m
    m = max(n, min(m, total - 1))

    for i in range(n, m + 1):
        writer.add_page(reader.pages[i])

    output_stream = BytesIO()
    writer.write(output_stream)
    output_stream.seek(0)

    logger.info(f"Extracted {m - n + 1} pages successfully.")
    return output_stream

def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    logger.info("Extracting text from PDF bytes...")
    try:
        pdf_file_obj = BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file_obj)
        text = ""
        for i, page in enumerate(reader.pages):
            text += (page.extract_text() or "") + "\n"
        
        logger.info(f"Successfully extracted text from {len(reader.pages)} pages.")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF bytes: {e}")
        raise RuntimeError(f"Failed to extract text from PDF bytes: {e}")